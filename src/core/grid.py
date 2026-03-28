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
from torch.utils.checkpoint import checkpoint
from .cell import CellState, CellConfig
from .perception import Perception3D, PerceptionConfig
from .update import UpdateConfig, UpdateRule


@dataclass(frozen=True)
class GridConfig:
    """Lattice dimensions."""

    size: Tuple[int, int, int] = (32, 32, 32)


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
            device=device,
        )

    def seed_center(
        self,
        batch_size: int,
        device: torch.device | str,
        task_ids: Tensor | None = None,
    ) -> Tensor:
        """
        Seed a single living cell at the lattice center.

        Visible channels: random RGB + alpha = 1.0 (alive signal).
        Hidden channels remain zero.
        """
        state = self.init_empty(batch_size, device)
        center = tuple(s // 2 for s in self.cfg.size)
        seed_vis = torch.rand(
            batch_size, self.cell.cfg.visible_channels, 1, 1, 1, device=device
        )
        seed_vis[:, -1:, ...] = 1.0
        state[
            :,
            -self.cell.cfg.visible_channels :,
            center[0] : center[0] + 1,
            center[1] : center[1] + 1,
            center[2] : center[2] + 1,
        ] = seed_vis

        if self.cell.cfg.task_channels > 0 and task_ids is not None:
            tc = self.cell.cfg.task_channels
            one_hot = (
                torch.nn.functional.one_hot(task_ids, num_classes=tc).to(device).float()
            )
            one_hot_grid = (
                one_hot.view(batch_size, tc, 1, 1, 1)
                .expand(-1, -1, *self.cfg.size)
                .contiguous()
            )
            vis = self.cell.cfg.visible_channels
            state[:, -(vis + tc) : -vis, ...] = one_hot_grid

        return state

    def step(self, state: Tensor) -> Tensor:
        """
        Advance the NCA one iteration.

        1. Compute pre_life: which cells are alive now
        2. Apply fixed 3x3x3 perception filters (all cells perceive)
        3. MLP predicts `dx` for every cell
        4. (Optional) Stochastic fire masks some updates
        5. new_state = state + dx
        6. Compute post_life on new_state
        7. Zero out cells where post_life is False

        Growth mechanism: dead cells adjacent to alive cells receive
        nonzero perception (neighbor info leaks in), so the MLP can
        produce a positive alpha delta that brings them to life.  The
        post_life mask then validates only cells with sufficient alpha.
        """
        pre_life = self.cell.update_alive_mask(state)
        perceived = self.perception(state)

        dx = self.update(perceived, pre_life, state)

        new_state = state + dx
        post_life = self.cell.update_alive_mask(new_state)
        return new_state * post_life.float()

    def forward(
        self,
        state: Tensor,
        steps: int = 1,
        use_checkpointing: bool = True,
    ) -> Tensor:
        """Run NCA for specified iterations.

        1. Apply alive mask (pre_life)
        2. Compute perception (3x3x3 fixed filters)
        3. MLP predicts state delta with optional stochastic fire
        4. new_state = state + delta
        5. Apply post_life mask (growth/death)
        Repeat for each step. Uses gradient checkpointing to reduce memory.

        Args:
            state: Current state [B, C, X, Y, Z].
            steps: Number of update iterations.
            use_checkpointing: If True, use gradient checkpointing (reduces memory, costs compute).

        Returns:
            Updated state [B, C, X, Y, Z] after `steps` iterations.

        When `use_checkpointing` is True (default) each step is wrapped in
        `torch.utils.checkpoint.checkpoint` so that intermediate activations
        are recomputed during the backward pass instead of being stored.
        This trades ~2x compute for O(1) memory w.r.t. step count - critical
        for long unrolls on low-VRAM GPUs.
        """
        for _ in range(steps):
            if use_checkpointing and state.requires_grad:
                state = checkpoint(self.step, state, use_reentrant=False)
            else:
                state = self.step(state)
            state = torch.clamp(state, -1.0, 1.0)
        return state
