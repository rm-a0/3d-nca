from __future__ import annotations

from nca3d.core.schedule import Event, EventType, Schedule


class _DummyRunner:
    def __init__(self, handled: bool = True) -> None:
        self.handled = handled
        self.received: list[Event] = []

    def on_event(self, event: Event) -> bool:
        self.received.append(event)
        return self.handled


def test_event_type_str_uses_value() -> None:
    assert str(EventType.LEARNING_RATE) == "LEARNING_RATE"


def test_event_normalizes_builtin_string_type() -> None:
    ev = Event(epoch=1, event_type="LEARNING_RATE", value=1e-3)
    assert ev.event_type == EventType.LEARNING_RATE


def test_event_from_dict_keeps_unknown_custom_type_as_string() -> None:
    ev = Event.from_dict({"epoch": 7, "event_type": "CUSTOM_PHASE", "value": 2.0})
    assert ev.event_type == "CUSTOM_PHASE"


def test_event_from_dict_recovers_pool_expand_enum() -> None:
    ev = Event.from_dict({"epoch": 3, "event_type": "POOL_EXPAND", "value": 1.0})
    assert ev.event_type == EventType.POOL_EXPAND


def test_schedule_executes_string_backed_event_without_crash() -> None:
    schedule = Schedule()
    runner = _DummyRunner(handled=True)

    schedule.add_event(Event(epoch=3, event_type="POOL_EXPAND", value=1.0))
    schedule.check_and_execute(epoch=3, runner=runner)

    assert len(runner.received) == 1
    assert runner.received[0].event_type == EventType.POOL_EXPAND
    assert schedule.events == []
