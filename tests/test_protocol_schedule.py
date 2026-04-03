from __future__ import annotations

import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.core.schedule import Event, EventType, NOW, Schedule
from src.server.protocol import (
    MAX_MSG_SIZE,
    b64_to_tensor,
    build_init_msg,
    build_run_model_msg,
    build_state_msg,
    decode_message,
    encode_message,
    parse_init_msg,
    parse_run_model_msg,
    parse_schedule_msg,
    parse_state_msg,
    tensor_to_b64,
    recv_msg,
)


def test_protocol_message_and_tensor_round_trips() -> None:
    array = np.array([[1.25, -2.5], [3.75, 4.0]], dtype=np.float64)
    encoded = tensor_to_b64(array)
    decoded = b64_to_tensor(encoded, [2, 2])

    assert decoded.dtype == np.float32
    assert np.allclose(decoded, array.astype(np.float32))

    target = np.arange(8, dtype=np.float32).reshape(2, 2, 2, 1)
    config = {"name": "demo", "steps": 4}
    init_msg = build_init_msg(config, target)
    parsed_config, parsed_target = parse_init_msg(init_msg)

    assert parsed_config == config
    assert np.array_equal(parsed_target, target)

    state = np.ones((1, 4, 2, 2, 2), dtype=np.float32)
    state_msg = build_state_msg(state, epoch=7, loss=1.5)
    parsed_state, epoch, loss = parse_state_msg(state_msg)

    assert epoch == 7
    assert loss == 1.5
    assert np.array_equal(parsed_state, state)

    model_msg = build_run_model_msg("Y2hlY2twb2ludA==", phase_steps=12, broadcast_every=3)
    model_bytes, phase_steps, broadcast_every = parse_run_model_msg(model_msg)

    assert model_bytes == b"checkpoint"
    assert phase_steps == 12
    assert broadcast_every == 3

    message = {"type": "ping", "payload": {"nested": [1, 2, 3]}}
    wire = encode_message(message)
    assert decode_message(wire[4:]) == message


def test_protocol_json_handles_utf8_payload() -> None:
    message = {"type": "ack", "message": "Ň3D model připraven"}
    wire = encode_message(message)
    decoded = decode_message(wire[4:])
    assert decoded == message


def test_recv_msg_rejects_payload_larger_than_max_size() -> None:
    oversized_header = struct.pack(">I", MAX_MSG_SIZE + 1)

    class FakeSocket:
        def __init__(self) -> None:
            self.calls = 0

        def recv(self, n: int) -> bytes:
            self.calls += 1
            if self.calls == 1:
                return oversized_header
            return b""

    with pytest.raises(ValueError, match="Message too large"):
        recv_msg(FakeSocket())


def test_recv_msg_returns_none_on_disconnect_before_header_complete() -> None:
    class FakeSocket:
        def recv(self, n: int) -> bytes:
            return b""

    assert recv_msg(FakeSocket()) is None


def test_schedule_serialization_and_execution() -> None:
    schedule = Schedule()
    target = np.ones((2, 2, 2, 1), dtype=np.float32)

    schedule.add_event(Event(epoch=1, event_type=EventType.LEARNING_RATE, value=0.05))
    schedule.add_event(Event(epoch=NOW, event_type=EventType.BATCH_SIZE, value=2.0))
    schedule.add_event(
        Event(epoch=1, event_type=EventType.TARGET_CHANGE, value=1.0, target=target)
    )

    snapshot = schedule.to_dict_list()
    restored = Schedule.from_dict_list(snapshot)

    assert [event.event_type for event in restored.events] == [
        EventType.LEARNING_RATE,
        EventType.BATCH_SIZE,
        EventType.TARGET_CHANGE,
    ]
    assert parse_schedule_msg({"events": snapshot}) == snapshot

    runner = SimpleNamespace(
        optimizer=SimpleNamespace(param_groups=[{"lr": 0.1}]),
        _batch_size=4,
        _alpha_weight=4.0,
        _color_weight=1.0,
        _overflow_weight=2.0,
        target=None,
    )

    def set_target(tensor: torch.Tensor) -> None:
        runner.target = tensor

    runner.set_target = set_target

    schedule.check_and_execute(1, runner)

    assert runner.optimizer.param_groups[0]["lr"] == 0.05
    assert runner._batch_size == 2
    assert isinstance(runner.target, torch.Tensor)
    assert runner.target.shape == (1, 1, 2, 2, 2)
    assert torch.allclose(runner.target, torch.ones_like(runner.target))
    assert schedule.events == []


def test_schedule_now_event_executes_on_any_epoch() -> None:
    schedule = Schedule()
    schedule.add_event(Event(epoch=NOW, event_type=EventType.BATCH_SIZE, value=9.0))

    runner = SimpleNamespace(_batch_size=1)
    schedule.check_and_execute(123, runner)

    assert runner._batch_size == 9
    assert schedule.events == []


def test_schedule_executes_multiple_same_epoch_events() -> None:
    schedule = Schedule()
    schedule.add_event(Event(epoch=4, event_type=EventType.ALPHA_WEIGHT, value=1.5))
    schedule.add_event(Event(epoch=4, event_type=EventType.COLOR_WEIGHT, value=0.7))
    schedule.add_event(Event(epoch=4, event_type=EventType.OVERFLOW_WEIGHT, value=2.2))

    runner = SimpleNamespace(
        _alpha_weight=0.0,
        _color_weight=0.0,
        _overflow_weight=0.0,
        optimizer=SimpleNamespace(param_groups=[]),
        _batch_size=1,
    )
    runner.set_target = lambda tensor: None

    schedule.check_and_execute(4, runner)

    assert runner._alpha_weight == 1.5
    assert runner._color_weight == 0.7
    assert runner._overflow_weight == 2.2
    assert schedule.events == []