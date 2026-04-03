from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.core.schedule import Event, EventType
from src.server.server import NCAServer
from src.server.trainer import NCATrainer, _switch_task_channel
from src.server.protocol import build_init_msg, build_run_model_msg, build_state_msg


_ROOT = Path(__file__).resolve().parents[1]


def _load_blender_client() -> tuple[type, types.ModuleType]:
    package = sys.modules.get("blender")
    if package is None:
        package = types.ModuleType("blender")
        package.__path__ = [str(_ROOT / "blender")]
        sys.modules["blender"] = package

    protocol_spec = importlib.util.spec_from_file_location(
        "blender.protocol", _ROOT / "blender" / "protocol.py"
    )
    assert protocol_spec is not None and protocol_spec.loader is not None
    protocol_module = importlib.util.module_from_spec(protocol_spec)
    sys.modules["blender.protocol"] = protocol_module
    protocol_spec.loader.exec_module(protocol_module)

    client_spec = importlib.util.spec_from_file_location(
        "blender.client", _ROOT / "blender" / "client.py"
    )
    assert client_spec is not None and client_spec.loader is not None
    client_module = importlib.util.module_from_spec(client_spec)
    sys.modules["blender.client"] = client_module
    client_spec.loader.exec_module(client_module)
    return client_module.NCAClient, client_module


NCAClient, _CLIENT_MODULE = _load_blender_client()


class _FakeSocket:
    def __init__(self) -> None:
        self.connect_calls: list[tuple[str, int]] = []
        self.timeout_values: list[float | None] = []
        self.shutdown_calls: list[int] = []
        self.closed = False

    def settimeout(self, value: float) -> None:
        self.timeout_values.append(value)

    def connect(self, addr: tuple[str, int]) -> None:
        self.connect_calls.append(addr)

    def shutdown(self, how: int) -> None:
        self.shutdown_calls.append(how)

    def close(self) -> None:
        self.closed = True


def test_nca_client_connect_send_and_disconnect(monkeypatch, tmp_path) -> None:
    fake_socket = _FakeSocket()
    monkeypatch.setattr(_CLIENT_MODULE.socket, "socket", lambda *args, **kwargs: fake_socket)

    sent_messages: list[dict] = []
    monkeypatch.setattr(_CLIENT_MODULE, "send_msg", lambda sock, msg: sent_messages.append(msg))

    client = NCAClient(host="example.com", port=6000)
    client.connect(timeout=1.5)

    assert client.connected
    assert fake_socket.connect_calls == [("example.com", 6000)]
    assert fake_socket.timeout_values[:2] == [1.5, 2.0]

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint-bytes")

    target = np.zeros((2, 2, 2, 1), dtype=np.float32)
    client.send_init({"mode": "demo"}, target)
    client.send_pause()
    client.send_resume()
    client.send_stop()
    client.send_schedule([Event(epoch=1, event_type=EventType.LEARNING_RATE, value=0.01).to_dict()])
    client.send_run_model(checkpoint.as_posix(), phase_steps=12, broadcast_every=3)

    assert sent_messages[0]["type"] == "init"
    assert sent_messages[0]["config"] == {"mode": "demo"}
    assert sent_messages[1] == {"type": "pause"}
    assert sent_messages[2] == {"type": "resume"}
    assert sent_messages[3] == {"type": "stop"}
    assert sent_messages[4]["type"] == "update_schedule"
    assert sent_messages[5]["type"] == "run_model"
    assert sent_messages[5]["phase_steps"] == 12
    assert sent_messages[5]["broadcast_every"] == 3
    assert client.connected

    client.disconnect()

    assert not client.connected
    assert fake_socket.shutdown_calls == [2]
    assert fake_socket.closed


def test_nca_client_reconnect_uses_fresh_socket(monkeypatch) -> None:
    sockets: list[_FakeSocket] = []

    def make_socket(*args, **kwargs):
        sock = _FakeSocket()
        sockets.append(sock)
        return sock

    monkeypatch.setattr(_CLIENT_MODULE.socket, "socket", make_socket)

    client = NCAClient(host="example.com", port=6001)
    client.connect(timeout=1.0)
    client.disconnect()
    client.connect(timeout=1.0)

    assert len(sockets) == 2
    assert sockets[0].closed
    assert client.connected
    client.disconnect()


def test_nca_client_listener_dispatches_messages(monkeypatch) -> None:
    client = NCAClient()
    client._running = True

    state = np.ones((1, 4, 2, 2, 2), dtype=np.float32)
    messages = [
        build_state_msg(state, epoch=5, loss=0.25),
        {"type": "error", "message": "boom"},
        {"type": "ack", "message": "ok"},
        None,
    ]

    def fake_recv_msg(sock):
        return messages.pop(0)

    seen_state: list[np.ndarray] = []
    seen_error: list[str] = []
    disconnected: list[bool] = []

    monkeypatch.setattr(_CLIENT_MODULE, "recv_msg", fake_recv_msg)

    client._listen_loop(seen_state.append, seen_error.append, lambda: disconnected.append(True))

    assert len(seen_state) == 1
    assert np.array_equal(seen_state[0], state)
    assert seen_error == ["boom"]
    assert disconnected == [True]


def test_server_handle_client_dispatches_protocol_messages(monkeypatch) -> None:
    server = NCAServer()

    init_msg = build_init_msg({"training": True}, np.zeros((2, 2, 2, 1), dtype=np.float32))
    run_msg = build_run_model_msg("Zg==", phase_steps=8, broadcast_every=2)
    schedule_msg = {"type": "update_schedule", "events": [Event(epoch=1, event_type=EventType.BATCH_SIZE, value=2.0).to_dict()]}
    bad_msg = {"type": "unknown"}
    messages = [init_msg, run_msg, {"type": "pause"}, {"type": "resume"}, {"type": "stop"}, schedule_msg, {"type": "ping"}, bad_msg, None]

    sent_messages: list[dict] = []

    class FakeTrainer:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple]] = []

        def init(self, config, target, send_fn):
            self.calls.append(("init", (config, target.shape, callable(send_fn))))

        def run_inference(self, model_bytes, phase_steps, broadcast_every, send_fn):
            self.calls.append(("run_inference", (model_bytes, phase_steps, broadcast_every, callable(send_fn))))

        def stop(self):
            self.calls.append(("stop", ()))

        def pause(self):
            self.calls.append(("pause", ()))

        def resume(self):
            self.calls.append(("resume", ()))

        def update_schedule(self, events):
            self.calls.append(("update_schedule", (events,)))

    fake_trainer = FakeTrainer()
    server.trainer = fake_trainer

    monkeypatch.setattr("src.server.server.recv_msg", lambda sock: messages.pop(0))
    monkeypatch.setattr("src.server.server.send_msg", lambda sock, msg: sent_messages.append(msg))

    server._handle_client(object())

    assert fake_trainer.calls[0][0] == "init"
    assert fake_trainer.calls[1][0] == "run_inference"
    assert fake_trainer.calls[2][0] == "pause"
    assert fake_trainer.calls[3][0] == "resume"
    assert fake_trainer.calls[4][0] == "stop"
    assert fake_trainer.calls[5][0] == "update_schedule"
    assert sent_messages[0] == {"type": "ack", "message": "Training started"}
    assert sent_messages[1] == {"type": "ack", "message": "Inference started"}
    assert sent_messages[2] == {"type": "ack", "message": "Paused"}
    assert sent_messages[3] == {"type": "ack", "message": "Resumed"}
    assert sent_messages[4] == {"type": "ack", "message": "Stopped"}
    assert sent_messages[5] == {"type": "ack", "message": "Schedule updated"}
    assert sent_messages[6] == {"type": "pong"}
    assert sent_messages[7]["type"] == "error"


def test_server_handle_client_stops_on_recv_error(monkeypatch) -> None:
    server = NCAServer()
    sent_messages: list[dict] = []

    monkeypatch.setattr(
        "src.server.server.recv_msg",
        lambda sock: (_ for _ in ()).throw(RuntimeError("socket failed")),
    )
    monkeypatch.setattr("src.server.server.send_msg", lambda sock, msg: sent_messages.append(msg))

    server._handle_client(object())

    assert sent_messages == []


def test_trainer_switch_channel_and_broadcast(monkeypatch) -> None:
    cell_cfg = SimpleNamespace(task_channels=2, visible_channels=1)
    state = torch.zeros(1, 4, 2, 2, 2)
    state[:, 2, ...] = 0.5

    switched = _switch_task_channel(state, 1, cell_cfg)

    assert switched.shape == state.shape
    assert torch.allclose(switched[:, 1], torch.zeros_like(switched[:, 1]))
    assert torch.allclose(switched[:, 2], torch.ones_like(switched[:, 2]))

    trainer = NCATrainer(verbose=False)
    trainer._send_fn = lambda msg: captured.append(msg)
    trainer._last_broadcast = 0.0

    captured: list[dict] = []
    monkeypatch.setattr("src.server.trainer.time.monotonic", lambda: 10.0)

    trainer._broadcast(state, epoch=3, loss=0.5, visible_channels=1)
    trainer._broadcast(state, epoch=4, loss=0.25, visible_channels=1)

    assert len(captured) == 1
    assert captured[0]["type"] == "state"
    assert captured[0]["epoch"] == 3
    assert captured[0]["loss"] == 0.5
    assert captured[0]["shape"] == [1, 1, 2, 2, 2]