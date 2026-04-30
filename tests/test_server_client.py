from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

from nca3d.core.runners import MorphRunner
from nca3d.core.schedule import Event, EventType
from nca3d.server.server import NCAServer
from nca3d.server.trainer import NCATrainer
from nca3d.server.protocol import build_init_msg, build_run_model_msg, build_state_msg


_ROOT = Path(__file__).resolve().parents[1]


def _load_blender_client() -> tuple[type, types.ModuleType]:
    package = sys.modules.get("blender")
    if package is None:
        package = types.ModuleType("blender")
        package.__path__ = [str(_ROOT / "blender")]
        sys.modules["blender"] = package

    for mod_name, filename in [
        ("blender.protocol", "protocol.py"),
        ("blender.client", "client.py"),
    ]:
        spec = importlib.util.spec_from_file_location(
            mod_name, _ROOT / "blender" / filename
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)

    return sys.modules["blender.client"].NCAClient, sys.modules["blender.client"]


NCAClient, _CLIENT_MODULE = _load_blender_client()


class _FakeSocket:
    def __init__(self):
        self.connect_calls: list = []
        self.timeout_values: list = []
        self.shutdown_calls: list = []
        self.closed = False

    def settimeout(self, v): self.timeout_values.append(v)
    def connect(self, addr): self.connect_calls.append(addr)
    def shutdown(self, how): self.shutdown_calls.append(how)
    def close(self): self.closed = True


# --- NCAClient tests ---

def test_nca_client_connect_send_and_disconnect(monkeypatch, tmp_path) -> None:
    fake_socket = _FakeSocket()
    monkeypatch.setattr(_CLIENT_MODULE.socket, "socket", lambda *a, **kw: fake_socket)

    sent: list[dict] = []
    monkeypatch.setattr(_CLIENT_MODULE, "send_msg", lambda s, m: sent.append(m))

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
    client.send_schedule(
        [Event(epoch=1, event_type=EventType.LEARNING_RATE, value=0.01).to_dict()]
    )
    client.send_run_model(checkpoint.as_posix(), phase_steps=12, broadcast_every=3)

    assert sent[0]["type"] == "init"
    assert sent[0]["config"] == {"mode": "demo"}
    assert sent[1] == {"type": "pause"}
    assert sent[2] == {"type": "resume"}
    assert sent[3] == {"type": "stop"}
    assert sent[4]["type"] == "update_schedule"
    assert sent[5]["type"] == "run_model"
    assert sent[5]["phase_steps"] == 12
    assert sent[5]["send_delay_ms"] == 40

    client.disconnect()
    assert not client.connected
    assert fake_socket.shutdown_calls == [2]
    assert fake_socket.closed


def test_nca_client_reconnect_uses_fresh_socket(monkeypatch) -> None:
    sockets: list[_FakeSocket] = []

    def make_socket(*a, **kw):
        s = _FakeSocket()
        sockets.append(s)
        return s

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

    monkeypatch.setattr(_CLIENT_MODULE, "recv_msg", lambda s: messages.pop(0))

    seen_state, seen_error, disconnected = [], [], []
    client._listen_loop(
        seen_state.append, seen_error.append, lambda: disconnected.append(True)
    )

    assert len(seen_state) == 1
    assert np.array_equal(seen_state[0], state)
    assert seen_error == ["boom"]
    assert disconnected == [True]


# --- NCAServer dispatch tests ---

def test_server_handle_client_dispatches_protocol_messages(monkeypatch) -> None:
    server = NCAServer()

    messages = [
        build_init_msg({"training": True}, np.zeros((2, 2, 2, 1), dtype=np.float32)),
        build_run_model_msg("Zg==", 8, 2),
        {"type": "pause"},
        {"type": "resume"},
        {"type": "stop"},
        {"type": "update_schedule",
         "events": [Event(epoch=1, event_type=EventType.BATCH_SIZE, value=2.0).to_dict()]},
        {"type": "ping"},
        {"type": "unknown"},
        None,
    ]
    sent: list[dict] = []

    class FakeTrainer:
        calls: list = []

        def init(self, c, t, fn): self.calls.append(("init",))
        def run_inference(self, b, ps, be, sd, fn): self.calls.append(("run_inference", sd))
        def stop(self): self.calls.append(("stop",))
        def pause(self): self.calls.append(("pause",))
        def resume(self): self.calls.append(("resume",))
        def update_schedule(self, e): self.calls.append(("update_schedule",))

    ft = FakeTrainer()
    server.trainer = ft

    monkeypatch.setattr("nca3d.server.server.recv_msg", lambda s: messages.pop(0))
    monkeypatch.setattr("nca3d.server.server.send_msg", lambda s, m: sent.append(m))

    server._handle_client(object())

    assert [c[0] for c in ft.calls] == [
        "init", "run_inference", "pause", "resume", "stop", "update_schedule"
    ]
    assert ft.calls[1][1] == 40
    assert sent[0] == {"type": "ack", "message": "Training started"}
    assert sent[6] == {"type": "pong"}
    assert sent[7]["type"] == "error"


def test_server_handle_client_stops_on_recv_error(monkeypatch) -> None:
    server = NCAServer()
    sent: list[dict] = []
    monkeypatch.setattr(
        "nca3d.server.server.recv_msg",
        lambda s: (_ for _ in ()).throw(RuntimeError("socket failed")),
    )
    monkeypatch.setattr("nca3d.server.server.send_msg", lambda s, m: sent.append(m))
    server._handle_client(object())
    assert sent == []


# --- NCATrainer tests ---

def test_trainer_accepts_custom_runner_factory() -> None:
    calls: list[str] = []

    class CountingRunner(MorphRunner):
        def __init__(self):
            super().__init__(verbose=False)
            calls.append("created")

    trainer = NCATrainer(verbose=False, runner_factory=CountingRunner)
    assert trainer._runner is None
    trainer.stop()
    assert calls == []


def test_trainer_broadcast_rate_limit(monkeypatch) -> None:
    state = torch.zeros(1, 4, 2, 2, 2)
    trainer = NCATrainer(verbose=False)
    captured: list[dict] = []
    trainer._send_fn = lambda m: captured.append(m)
    trainer._last_broadcast = 0.0

    monkeypatch.setattr("nca3d.server.trainer.time.monotonic", lambda: 10.0)

    trainer._broadcast(state, epoch=3, loss=0.5, visible_channels=1)
    trainer._broadcast(state, epoch=4, loss=0.25, visible_channels=1)

    assert len(captured) == 1
    assert captured[0]["type"] == "state"
    assert captured[0]["epoch"] == 3
    assert captured[0]["loss"] == 0.5
    assert captured[0]["shape"] == [1, 1, 2, 2, 2]


def test_trainer_parse_inference_checkpoint_accepts_nested_schema() -> None:
    trainer = NCATrainer(verbose=False)
    ckpt = {
        "config": {
            "cell": {
                "hidden_channels": 2,
                "visible_channels": 1,
                "alive_threshold": 0.1,
                "task_channels": 0,
            },
            "perception": {"kernel_radius": 1, "channel_groups": 3},
            "update": {"hidden_dim": 8, "stochastic_update": False, "fire_rate": 0.5},
            "grid": {"size": [4, 4, 4]},
        },
        "state_dict": {"perception.depthwise.weight": torch.zeros(9, 1, 3, 3, 3)},
    }

    cell_cfg, _perc_cfg, _upd_cfg, grid_cfg, state_dict = trainer._parse_inference_checkpoint(ckpt)

    assert cell_cfg.hidden_channels == 2
    assert cell_cfg.visible_channels == 1
    assert cell_cfg.task_channels == 0
    assert grid_cfg.size == (4, 4, 4)
    assert "perception.depthwise.weight" in state_dict


def test_trainer_parse_inference_checkpoint_accepts_flat_nca_schema() -> None:
    trainer = NCATrainer(verbose=False)
    ckpt = {
        "config": {
            "grid_size": [4, 4, 4],
            "hidden_channels": 2,
            "visible_channels": 1,
            "alive_threshold": 0.15,
            "task_channels": 3,
            "perception_kernel_radius": 1,
            "perception_channel_groups": 3,
            "update_hidden_dim": 8,
            "update_stochastic": False,
            "update_fire_rate": 0.5,
        },
        "state_dict": {"grid.perception.depthwise.weight": torch.zeros(18, 1, 3, 3, 3)},
    }

    cell_cfg, _perc_cfg, _upd_cfg, grid_cfg, state_dict = trainer._parse_inference_checkpoint(ckpt)

    assert cell_cfg.hidden_channels == 2
    assert cell_cfg.visible_channels == 1
    assert cell_cfg.task_channels == 3
    assert grid_cfg.size == (4, 4, 4)
    assert "grid.perception.depthwise.weight" not in state_dict
    assert "perception.depthwise.weight" in state_dict
