"""Regeneration runner with damage-and-recovery training.

This strategy periodically damages pooled states so the automaton learns
to recover target structure after local perturbations.
"""

from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
from torch import Tensor

from .default import MorphRunner

_DEFAULT_REGEN_DAMAGE_PROB = 0.5
_DEFAULT_REGEN_DAMAGE_SIZE = 6


class RegenRunner(MorphRunner):
    """Morphogenesis runner augmented with random regeneration damage."""

    def __init__(self, verbose: bool = True) -> None:
        super().__init__(verbose=verbose)
        self._regen_damage_prob: float = _DEFAULT_REGEN_DAMAGE_PROB
        self._regen_damage_size: int = _DEFAULT_REGEN_DAMAGE_SIZE

    def init(self, config: dict, target: np.ndarray | list[np.ndarray]) -> None:
        super().init(config, target)

        training_cfg = config.get("training", {})
        self._regen_damage_prob = float(
            training_cfg.get("regen_damage_prob", _DEFAULT_REGEN_DAMAGE_PROB)
        )
        self._regen_damage_size = int(
            training_cfg.get("regen_damage_size", _DEFAULT_REGEN_DAMAGE_SIZE)
        )

        if not 0.0 <= self._regen_damage_prob <= 1.0:
            raise ValueError("training.regen_damage_prob must be in [0, 1]")
        if self._regen_damage_size < 1:
            raise ValueError("training.regen_damage_size must be >= 1")

    def _step(self) -> Dict[str, Any]:
        damaged = self._apply_pool_damage()
        metrics = super()._step()
        metrics["damaged_pool_states"] = float(damaged)
        return metrics

    def _apply_pool_damage(self) -> int:
        if not self._pool or self._regen_damage_prob <= 0.0:
            return 0

        indices = [
            i for i in range(len(self._pool))
            if random.random() < self._regen_damage_prob
        ]

        if not indices:
            indices = [random.randrange(len(self._pool))]

        for idx in indices:
            damaged = self._pool[idx].clone()
            self._apply_box_damage_(damaged)
            self._pool[idx] = damaged

        return len(indices)

    def _apply_box_damage_(self, state: Tensor) -> None:
        _, _, depth, height, width = state.shape
        side = max(1, min(self._regen_damage_size, depth, height, width))

        z0 = random.randint(0, depth - side)
        y0 = random.randint(0, height - side)
        x0 = random.randint(0, width - side)

        state[:, :, z0:z0 + side, y0:y0 + side, x0:x0 + side] = 0.0
