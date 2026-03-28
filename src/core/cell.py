"""
CellState - defines what a single voxel is.

A cell is a vector of length C = hidden + visible:
  - hidden channels (default 16) - internal memory, not rendered
  - visible channels (default 4) - RGBA, A = alpha (alive signal)

The alive mask uses a 3x3x3 max-pool on alpha:
max(alpha of self + 26 neighbors) > threshold

Original implementation (update_alive_mask) inspired by:
https://github.com/SkyLionx/3d-cellular-automaton
"""

from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class CellConfig:
    """Configuration of a single cell."""

    hidden_channels: int = 16
    visible_channels: int = 4
    alive_threshold: float = 0.1
    task_channels: int = 0

    @property
    def total_channels(self) -> int:
        """Total number of channels (hidden + visible)."""
        return self.hidden_channels + self.visible_channels + self.task_channels


class CellState(torch.nn.Module):
    """
    Holds cell configuration and the alive mask.

    The alive mask [B,1,X,Y,Z] tells Grid3D which cells may update.
    Recomputed every step via max-pool on alpha.
    """

    def __init__(self, cfg: CellConfig):
        super().__init__()
        self.cfg = cfg
        self.alive_mask: Tensor | None = None

    @property
    def total_channels(self) -> int:
        """Total number of channels (hidden + visible)."""
        return self.cfg.total_channels

    @torch.no_grad()
    def update_alive_mask(self, state: Tensor) -> Tensor:
        """Compute alive mask using 3x3x3 max-pool on alpha channel.

        Identifies live cells (alpha > threshold) plus all neighbors of live cells.
        Enables growth: dead neighbors of alive cells can receive nonzero updates
        and transition to alive state.

        Algorithm:
        1. Extract alpha channel from state [B,1,X,Y,Z]
        2. Apply 3x3x3 max-pool (max of cell + 26 neighbors)
        3. Compare pooled values to alive_threshold

        Args:
            state: Current state [B,C,X,Y,Z].

        Returns:
            Boolean mask [B,1,X,Y,Z]. True where cell or neighbor has alpha > threshold.
            Also stored in self.alive_mask for inspection/logging.
        """
        alpha = state[:, -1:, ...]  # [B,1,X,Y,Z]
        pooled = F.max_pool3d(alpha, kernel_size=3, stride=1, padding=1)
        alive = pooled > self.cfg.alive_threshold
        self.alive_mask = alive
        return alive
