"""
Schedule - Event manager for dynamic training parameter changes.

Supports one-shot training events triggered at specific epochs:
  - Learning rate updates
  - Batch size adjustments
  - Loss weight modifications (alpha, color, overflow)
  - Target voxel grid changes

Thread-safe: Uses locking to allow main thread updates while the training
loop runs in a background thread without blocking or race conditions.
"""

from __future__ import annotations

import base64
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from .runners import NCARunner

NOW = -1


def _tensor_to_b64(tensor: np.ndarray) -> str:
    return base64.b64encode(tensor.astype(np.float32).tobytes()).decode("ascii")


def _b64_to_tensor(b64: str, shape: list) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.float32).reshape(shape)


class EventType(str, Enum):
    LEARNING_RATE = "LEARNING_RATE"
    BATCH_SIZE = "BATCH_SIZE"
    ALPHA_WEIGHT = "ALPHA_WEIGHT"
    COLOR_WEIGHT = "COLOR_WEIGHT"
    OVERFLOW_WEIGHT = "OVERFLOW_WEIGHT"
    TARGET_CHANGE = "TARGET_CHANGE"
    POOL_EXPAND = "POOL_EXPAND"

    def __str__(self) -> str:
        return self.value


def _event_type_name(event_type: EventType | str) -> str:
    if isinstance(event_type, EventType):
        return event_type.value
    return str(event_type)


@dataclass
class Event:
    """A single scheduled parameter change.

    For TARGET_CHANGE events, ``target`` is in external (D, H, W, C) format.
    The runner is responsible for converting it to the internal format.
    Use ``NOW`` (-1) as ``epoch`` to fire the event on the next training step.
    """

    epoch: int
    event_type: EventType | str
    value: float
    target: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Normalize built-in string event types to EventType enums.

        Unknown values are kept as raw strings so custom runner events remain supported.
        """
        if isinstance(self.event_type, str) and not isinstance(
            self.event_type, EventType
        ):
            name = self.event_type.strip()
            try:
                self.event_type = EventType(name)
            except ValueError:
                self.event_type = name

    def to_dict(self) -> dict:
        """Serialise event to a JSON-compatible dict.

        Returns:
            Dict with ``epoch``, ``event_type``, and ``value``.
            TARGET_CHANGE events also include base64-encoded ``target`` and ``target_shape``.
        """
        event_type_name = _event_type_name(self.event_type)
        d: dict = {
            "epoch": self.epoch,
            "event_type": event_type_name,
            "value": self.value,
        }
        if event_type_name == EventType.TARGET_CHANGE.value and self.target is not None:
            d["target"] = _tensor_to_b64(self.target)
            d["target_shape"] = list(self.target.shape)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Event":
        """Deserialise an event from a JSON-compatible dict.

        Args:
            data: Dict produced by ``to_dict()``.

        Returns:
            Reconstructed Event instance.
        """
        event_type_raw = str(data["event_type"])
        target = None
        if event_type_raw == EventType.TARGET_CHANGE.value and "target" in data:
            target = _b64_to_tensor(data["target"], data["target_shape"])

        try:
            event_type: EventType | str = EventType(event_type_raw)
        except ValueError:
            event_type = event_type_raw

        return cls(
            epoch=int(data["epoch"]),
            event_type=event_type,
            value=float(data["value"]),
            target=target,
        )


class Schedule:
    """Ordered collection of one-shot training events.

    Thread-safe: A lock protects the event list so the main thread can
    update the schedule while the training loop reads from a background thread.
    """

    def __init__(self) -> None:
        self.events: List[Event] = []
        self._lock = threading.Lock()

    def add_event(self, event: Event) -> None:
        """Add a single event.

        Args:
            event: Event to append.
        """
        with self._lock:
            self.events.append(event)

    def remove_event(self, index: int) -> None:
        """Remove event at index.

        Args:
            index: Position in the event list to remove.
        """
        with self._lock:
            if 0 <= index < len(self.events):
                self.events.pop(index)

    def replace(self, events: List[Event]) -> None:
        """Replace all events atomically.

        Used when updating the schedule from the server or UI main thread.

        Args:
            events: New event list to install.
        """
        with self._lock:
            self.events = list(events)

    def clear(self) -> None:
        """Remove all events."""
        with self._lock:
            self.events.clear()

    def check_and_execute(self, epoch: int, runner: "NCARunner") -> None:
        """Fire all events whose epoch matches and remove them from the list.

        Delivers each event to ``runner.on_event(event)``.

        Args:
            epoch: Current epoch number.
            runner: NCARunner instance that handles the events.
        """
        with self._lock:
            remaining: List[Event] = []
            for ev in self.events:
                if ev.epoch == epoch or ev.epoch == NOW:
                    _apply_event(ev, runner)
                else:
                    remaining.append(ev)
            self.events = remaining

    def to_dict_list(self) -> List[dict]:
        """Serialise the event list for JSON persistence.

        Returns:
            List of dicts produced by ``Event.to_dict()``.
        """
        with self._lock:
            return [ev.to_dict() for ev in self.events]

    @classmethod
    def from_dict_list(cls, data: List[dict]) -> "Schedule":
        """Deserialise a schedule from a list of event dicts.

        Args:
            data: List produced by ``to_dict_list()``.

        Returns:
            Schedule instance with the deserialised events.
        """
        sched = cls()
        with sched._lock:
            sched.events = [Event.from_dict(d) for d in data]
        return sched


def _apply_event(event: Event, runner: "NCARunner") -> None:
    """Deliver one event to the runner and log the outcome.

    Args:
        event: Event to deliver.
        runner: NCARunner that handles the event via on_event().
    """
    handled = runner.on_event(event)
    label = _event_type_name(event.event_type).lower()
    if handled:
        print(f"[Schedule] Epoch {event.epoch}: {label}")
    else:
        print(f"[Schedule] Epoch {event.epoch}: {label} - not handled by runner")
