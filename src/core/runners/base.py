"""Training runner interfaces and shared DTOs.

This module defines the abstraction consumed by server-side orchestration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Generator

import numpy as np


@dataclass(frozen=True)
class TrainingSnapshot:
    """Immutable point-in-time view of training state for logging and broadcast.

    Attributes:
        state: Current NCA state in internal (B, C, D, H, W) format.
        epoch: Current training epoch.
        total_epochs: Total number of epochs for this session.
        loss: Latest total loss value.
        visible_channels: Number of visible channels to broadcast.
        metrics: Optional per-epoch metric breakdown.
    """

    state: np.ndarray
    epoch: int
    total_epochs: int
    loss: float
    visible_channels: int
    metrics: dict[str, float] = field(default_factory=dict)


class NCARunner(ABC):
    """Interface for NCA training backends."""

    @abstractmethod
    def init(self, config: dict, target: np.ndarray | list[np.ndarray]) -> None:
        """Prepare the runner for a new training session."""

    @abstractmethod
    def train(
        self, schedule: "Schedule | None" = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Run the training loop and yield per-epoch metrics."""

    @abstractmethod
    def snapshot(self) -> TrainingSnapshot:
        """Return a point-in-time snapshot of the current training state."""

    @abstractmethod
    def set_target(self, target: Any) -> None:
        """Swap the active training target."""

    def on_event(self, event: "Event") -> bool:
        """Handle a schedule event.

        Returns True when handled, False when ignored.
        """
        return False


if TYPE_CHECKING:
    from ..schedule import Event, Schedule
