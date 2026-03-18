from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, List, Optional
import numpy as np

if TYPE_CHECKING:
    from .trainer import NCATrainer

NOW = -1


class EventType(str, Enum):
    """Concrete set of parameters that can be changed mid-training."""
    LEARNING_RATE   = "LEARNING_RATE"
    BATCH_SIZE      = "BATCH_SIZE"
    ALPHA_WEIGHT    = "ALPHA_WEIGHT"
    COLOR_WEIGHT    = "COLOR_WEIGHT"
    OVERFLOW_WEIGHT = "OVERFLOW_WEIGHT"
    TARGET_CHANGE   = "TARGET_CHANGE"


@dataclass
class Event:
    """A single scheduled parameter change."""
    epoch: int
    event_type: EventType
    value: float
    target: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        from .protocol import tensor_to_b64
        d = {
            "epoch": self.epoch,
            "event_type": self.event_type.value,
            "value": self.value,
        }
        if self.event_type == EventType.TARGET_CHANGE and self.target is not None:
            d["target"] = tensor_to_b64(self.target)
            d["target_shape"] = list(self.target.shape)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Event":
        from .protocol import b64_to_tensor
        target = None
        if data.get("event_type") == EventType.TARGET_CHANGE and "target" in data:
            target = b64_to_tensor(data["target"], data["target_shape"])
        return cls(
            epoch=int(data["epoch"]),
            event_type=EventType(data["event_type"]),
            value=float(data["value"]),
            target=target,
        )

class Schedule:
    """Ordered collection of one-shot training events."""
    def __init__(self) -> None:
        self.events: List[Event] = []

    def add_event(self, event: Event) -> None:
        self.events.append(event)

    def remove_event(self, index: int) -> None:
        if 0 <= index < len(self.events):
            self.events.pop(index)

    def replace(self, events: List[Event]) -> None:
        """Bulk-replace all events."""
        self.events = list(events)

    def clear(self) -> None:
        self.events.clear()

    def check_and_execute(self, epoch: int, trainer: "NCATrainer") -> None:
        """Execute and remove every event whose epoch matches *epoch*."""
        remaining: List[Event] = []
        for ev in self.events:
            if ev.epoch == epoch or ev.epoch == NOW:
                _apply_event(ev, trainer)
            else:
                remaining.append(ev)
        self.events = remaining

    def to_dict_list(self) -> List[dict]:
        return [ev.to_dict() for ev in self.events]

    @classmethod
    def from_dict_list(cls, data: List[dict]) -> "Schedule":
        sched = cls()
        sched.events = [Event.from_dict(d) for d in data]
        return sched

def _apply_event(event: Event, trainer: "NCATrainer") -> None:
    """Mutate *trainer* according to *event*."""
    t = event.event_type
    v = event.value

    if t == EventType.LEARNING_RATE:
        for pg in trainer.optimizer.param_groups:
            pg["lr"] = v
        print(f"[Schedule] Epoch {event.epoch}: learning_rate -> {v}")

    elif t == EventType.BATCH_SIZE:
        trainer._batch_size = max(1, int(v))
        print(f"[Schedule] Epoch {event.epoch}: batch_size -> {int(v)}")

    elif t == EventType.ALPHA_WEIGHT:
        trainer._alpha_weight = v
        print(f"[Schedule] Epoch {event.epoch}: alpha_weight -> {v}")

    elif t == EventType.COLOR_WEIGHT:
        trainer._color_weight = v
        print(f"[Schedule] Epoch {event.epoch}: color_weight -> {v}")

    elif t == EventType.OVERFLOW_WEIGHT:
        trainer._overflow_weight = v
        print(f"[Schedule] Epoch {event.epoch}: overflow_weight -> {v}")

    elif t == EventType.TARGET_CHANGE:
        import torch
        target_chw = np.transpose(event.target, (3, 0, 1, 2)).astype(np.float32)
        t_tensor = torch.from_numpy(target_chw).unsqueeze(0)
        trainer.set_target(t_tensor)
        print(f"[Schedule] Epoch {event.epoch}: target_change")

    else:
        print(f"[Schedule] Unknown event type: {t}")