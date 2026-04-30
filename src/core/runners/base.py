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
    """Immutable point-in-time view of training state for logging and broadcast."""

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
        """Prepare the runner for a new training session.

        Args:
            config: Training hyperparameters (learning rate, batch size, etc.).
            target: Target voxel grid(s) in ``(D, H, W, C)`` format.
        """

    @abstractmethod
    def train(
        self, schedule: "Schedule | None" = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Run the training loop and yield per-epoch metrics.

        Args:
            schedule: Optional event schedule applied each epoch.

        Yields:
            Dict of metric names to values for each completed epoch.
        """

    @abstractmethod
    def snapshot(self) -> TrainingSnapshot:
        """Return a point-in-time snapshot of the current training state.

        Returns:
            Immutable :class:`TrainingSnapshot` for logging and broadcast.
        """

    @abstractmethod
    def set_target(self, target: Any) -> None:
        """Swap the active training target without restarting the session.

        Args:
            target: New target voxel grid(s).
        """

    def on_event(self, event: "Event") -> bool:
        """Handle a schedule event.

        Args:
            event: The event to handle.

        Returns:
            ``True`` when the event was handled, ``False`` when ignored.
        """
        return False


if TYPE_CHECKING:
    from ..schedule import Event, Schedule
