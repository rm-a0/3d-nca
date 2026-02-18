"""
Grid3D - 3D Neural Cellular Automata (NCA) simulator.

State is a tensor of shape [B, C, X, Y, Z]:
  - B = batch size (run multiple worlds in parallel)
  - C = total channels per cell (hidden + visible)
  - X, Y, Z = 3D lattice size

Each voxel is a cell. The update is local (3x3x3 neighborhood),
learned via a small MLP, and damage-robust using alive masking.

Original implementation inspired by:
https://github.com/SkyLionx/3d-cellular-automaton
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor 
from .cell import CellState, CellConfig
from .perception import Perception3D, PerceptionConfig
from .update import UpdateConfig, UpdateRule

@dataclass(frozen=True)
class GridConfig:
    """Lattice dimensions. Must be odd for clean center seeding."""
    size: Tuple[int, int, int] = (32, 32, 32)
    padding_mode: str = "zeros"

class Grid3D(torch.nn.Module):
    """
    Composes the three core NCA components:
      - CellState: defines the per-voxel state vector
      - Perception3D: fixed 3x3x3 depthwise filters (identity, neighbor sum, gradient)
      - UpdateRule: 1x1x1 MLP that predicts state delta
    Damage robustness via alive-mask intersection (pre & post).
    """
    def __init__(
        self, 
        cell_cfg: CellConfig,
        perc_cfg: PerceptionConfig,
        upd_cfg: UpdateConfig,
        grid_cfg: GridConfig, 
    ): 
        super().__init__()
        self.cfg = grid_cfg
        self.cell = CellState(cell_cfg)
        self.perception = Perception3D(perc_cfg, cell_cfg.total_channels)
        self.update = UpdateRule(perc_cfg, cell_cfg, upd_cfg)

    def init_empty(self, batch_size: int, device: torch.device | str) -> Tensor:
        return torch.zeros(
            batch_size,
            self.cell.total_channels,
            *self.cfg.size,
            dtype=torch.float32,
            device=device
        )
    
    def seed_center(self, batch_size: int, device: torch.device | str) -> Tensor:
        """
        Seed a single living cell at the lattice center.

        Visible channels: random RGB + alpha = 1.0 (alive signal).
        Hidden channels remain zero.
        """
        state = self.init_empty(batch_size, device)
        center = tuple(s // 2 for s in self.cfg.size)
        seed_vis = torch.rand(batch_size, self.cell.cfg.visible_channels, 1, 1, 1, device=device)
        seed_vis[:, -1:, ...] = 1.0
        state[:, -self.cell.cfg.visible_channels:, center[0]:center[0]+1,
                    center[1]:center[1]+1, center[2]:center[2]+1] = seed_vis
        return state


    def step(self, state: Tensor) -> Tensor:
        """
        Advance the NCA one iteration.

        1. Compute pre_life: which cells are alive now (max(alpha of cell + 26 neighbors) > threshold)
        2. Apply fixed 3x3x3 perception filters
        3. MLP predicts `dx` (change to apply to each cell's state)
        4. (Optional) Stochastic fire (randomly zero out some updates)
        5. Add `dx` to current `state` (temporary next state)
        6. Compute `post_life` (which cells would be alive after the update)
        7. Final update: keep changes only where `pre_life & post_life` is True

        The `pre & post` intersection ensures:
          - Dead cells can't come back to life
          - Living cells don't die from a single bad update
          - Growth is stable and damage-resistant

        Source: https://github.com/SkyLionx/3d-cellular-automaton
        """
        device = state.device
        batch = state.shape[0]

        pre_life = self.cell.update_alive_mask(state)
        perceived = self.perception(state)

        dx = self.update(perceived, pre_life, state)

        post_life = self.cell.update_alive_mask(state + dx)
        life_mask = pre_life & post_life
        result = (state + dx) * life_mask.float()

        return result

    def forward(self, state: Tensor, steps: int = 1) -> Tensor:
        """
        Run `steps` iterations with clamping to [-1, 1] after each step.
        Prevents value explosion during training or long rollouts.
        """
        for _ in range(steps):
            state = self.step(state)
            state = torch.clamp(state, -1.0, 1.0)
        return state
