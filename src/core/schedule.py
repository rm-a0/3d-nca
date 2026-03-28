"""
Schedule - Event manager for dynamic training parameter changes.

Supports one-shot training events triggered at specific epochs:
  - Learning rate updates
  - Batch size adjustments
  - Loss weight modifications (alpha, color, overflow)
  - Target voxel grid changes

Thread-safe: Uses locking to allow main thread updates while training loop
runs in background without blocking or race conditions.
"""

from __future__ import annotations

import base64
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from .runner import NCARunner

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


@dataclass
class Event:
    """A single scheduled parameter change.

    For TARGET_CHANGE events, target is in external (D,H,W,C) format.
    Converted to internal (B,C,D,H,W) format when applied to runner.
    """

    epoch: int
    event_type: EventType
    value: float
    target: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = {
            "epoch": self.epoch,
            "event_type": self.event_type.value,
            "value": self.value,
        }
        if self.event_type == EventType.TARGET_CHANGE and self.target is not None:
            d["target"] = _tensor_to_b64(self.target)
            d["target_shape"] = list(self.target.shape)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Event":
        target = None
        if data.get("event_type") == EventType.TARGET_CHANGE and "target" in data:
            target = _b64_to_tensor(data["target"], data["target_shape"])
        return cls(
            epoch=int(data["epoch"]),
            event_type=EventType(data["event_type"]),
            value=float(data["value"]),
            target=target,
        )


class Schedule:
    """Ordered collection of one-shot training events.

    Thread-safe: Uses locking to protect event list from concurrent modifications.
    Supports safe updates from main thread while training loop reads from background thread.
    """

    def __init__(self) -> None:
        self.events: List[Event] = []
        self._lock = threading.Lock()

    def add_event(self, event: Event) -> None:
        """Add a single event. Thread-safe."""
        with self._lock:
            self.events.append(event)

    def remove_event(self, index: int) -> None:
        """Remove event at index. Thread-safe."""
        with self._lock:
            if 0 <= index < len(self.events):
                self.events.pop(index)

    def replace(self, events: List[Event]) -> None:
        """Replace all events atomically. Thread-safe.

        Used when updating schedule from server/UI (main thread).
        """
        with self._lock:
            self.events = list(events)

    def clear(self) -> None:
        """Clear all events. Thread-safe."""
        with self._lock:
            self.events.clear()

    def check_and_execute(self, epoch: int, runner: "NCARunner") -> None:
        """Check for events at this epoch and execute them.

        Args:
            epoch: Current epoch number
            runner: NCARunner instance to apply events to

        Thread-safe: acquires internal lock while reading/modifying event list.
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
        """Serialize events to list of dicts for JSON persistence.

        Returns:
            List of serialized event dictionaries. Thread-safe.
        """
        with self._lock:
            return [ev.to_dict() for ev in self.events]

    @classmethod
    def from_dict_list(cls, data: List[dict]) -> "Schedule":
        """Deserialize schedule from JSON-serialized event dictionaries.

        Args:
            data: List of event dictionaries from to_dict_list() output.

        Returns:
            Schedule instance with deserialized events.
        """
        sched = cls()
        with sched._lock:
            sched.events = [Event.from_dict(d) for d in data]
        return sched


def _apply_event(event: Event, runner: "NCARunner") -> None:
    t = event.event_type
    v = event.value

    if t == EventType.LEARNING_RATE:
        for pg in runner.optimizer.param_groups:
            pg["lr"] = v
        print(f"[Schedule] Epoch {event.epoch}: learning_rate -> {v}")

    elif t == EventType.BATCH_SIZE:
        runner._batch_size = max(1, int(v))
        print(f"[Schedule] Epoch {event.epoch}: batch_size -> {int(v)}")

    elif t == EventType.ALPHA_WEIGHT:
        runner._alpha_weight = v
        print(f"[Schedule] Epoch {event.epoch}: alpha_weight -> {v}")

    elif t == EventType.COLOR_WEIGHT:
        runner._color_weight = v
        print(f"[Schedule] Epoch {event.epoch}: color_weight -> {v}")

    elif t == EventType.OVERFLOW_WEIGHT:
        runner._overflow_weight = v
        print(f"[Schedule] Epoch {event.epoch}: overflow_weight -> {v}")

    elif t == EventType.TARGET_CHANGE:
        # Convert from external (D,H,W,C) to internal (B,C,D,H,W) format
        import torch

        target_chw = np.transpose(event.target, (3, 0, 1, 2)).astype(
            np.float32
        )  # (D,H,W,C) -> (C,D,H,W)
        t_tensor = torch.from_numpy(target_chw).unsqueeze(0)  # -> (1,C,D,H,W)
        runner.set_target(t_tensor)
        print(f"[Schedule] Epoch {event.epoch}: target_change")

    else:
        print(f"[Schedule] Unknown event type: {t}")
