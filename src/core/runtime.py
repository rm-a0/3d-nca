"""Runtime contracts for training backends.

This module defines the stable boundary between the core training execution
engine and higher-level orchestration layers such as the server and Blender UI.
Custom trainers can implement this interface without inheriting the default
runner implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Any, Generator, Optional, TYPE_CHECKING

import numpy as np


@dataclass(frozen=True)
class TrainingSnapshot:
    """Immutable view of the current training state.

    The state tensor is kept in the internal NCA layout so downstream code can
    decide how to broadcast or visualize it without depending on a specific
    trainer implementation.
    """

    state: np.ndarray
    epoch: int
    total_epochs: int
    loss: float
    visible_channels: int
    metrics: dict[str, float] = field(default_factory=dict)


class TrainingRuntime(ABC):
    """Abstract contract for a trainable NCA runtime."""

    @abstractmethod
    def init(self, config: dict, target: np.ndarray | list[np.ndarray]) -> None:
        """Initialize the runtime for a new training session."""

    @abstractmethod
    def train(self, schedule: Optional["Schedule"] = None) -> Generator[dict[str, Any], None, None]:
        """Run the training loop and yield per-epoch metrics."""

    def pause(self) -> None:
        """Pause the active session."""
        return None

    def resume(self) -> None:
        """Resume a paused session."""
        return None

    def stop(self) -> None:
        """Stop the active session and release resources."""
        return None

    @abstractmethod
    def set_target(self, target: Any) -> None:
        """Swap the current training target."""

    @abstractmethod
    def snapshot(self) -> TrainingSnapshot:
        """Return a point-in-time snapshot for broadcasting/logging."""

    def apply_schedule_event(self, event: Any) -> bool:
        """Apply a scheduled event and return True when handled.

        Returning False means the runtime intentionally ignores the event.
        """
        return False

    @property
    def supports_schedule_events(self) -> bool:
        """Return True when this runtime wants schedule events delivered."""
        return False

    @property
    def is_running(self) -> bool:
        """Return True while a background session is active."""
        return False

    @property
    def is_paused(self) -> bool:
        """Return True while the session is intentionally paused."""
        return False


class BaseTrainingRuntime(TrainingRuntime):
    """Base runtime with opt-in defaults for lifecycle control.

    Subclasses get pause/resume/stop state handling for free and can opt into
    schedule events by overriding supports_schedule_events/apply_schedule_event.
    """

    def __init__(
        self,
        *,
        verbose: bool = True,
        pause_poll_interval_s: float = 0.01,
    ) -> None:
        self.verbose = verbose
        self._is_running = False
        self._is_paused = False
        self._stop_requested = False
        self._pause_poll_interval_s = max(0.0, float(pause_poll_interval_s))

    def pause(self) -> None:
        self._is_paused = True

    def resume(self) -> None:
        self._is_paused = False

    def stop(self) -> None:
        self._stop_requested = True
        self._is_paused = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    def _begin_training_loop(self) -> None:
        self._is_running = True
        self._stop_requested = False

    def _end_training_loop(self) -> None:
        self._is_running = False

    def _wait_if_paused(self) -> None:
        while self._is_paused and not self._stop_requested:
            time.sleep(self._pause_poll_interval_s)


if TYPE_CHECKING:
    from .schedule import Schedule