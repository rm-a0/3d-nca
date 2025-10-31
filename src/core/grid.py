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
    size: Tuple[int, int, int] = (32, 32, 32)
    padding_mode: str = "zeros"

class Grid3D(torch.nn.Module):
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
        self.perception = Perception3D(perc_cfg, self.cell.total_channels())
        self.update = UpdateRule(perc_cfg, cell_cfg, upd_cfg)

    def init_empty(self, batch_size: int, device: torch.device | str) -> Tensor:
        return torch.zeros(
            batch_size,
            self.cell.total_channels(),
            *self.cfg.size,
            dtype=torch.float32,
            device=device
        )
    
   # source: https://github.com/SkyLionx/3d-cellular-automaton 
    def seed_center(self, batch_size: int, device: torch.device | str) -> Tensor:
        state = self.init_empty(batch_size, device)
        center = tuple(s // 2 for s in self.cfg.size)
        seed_vis = torch.rand(batch_size, self.cell.cfg.visible_channels, 1, 1, 1, device=device)
        seed_vis[:, -1:, ...] = 1.0
        state[:, -self.cell.cfg.visible_channels:, center[0]:center[0]+1,
                    center[1]:center[1]+1, center[2]:center[2]+1] = seed_vis
        return state


    # source: https://github.com/SkyLionx/3d-cellular-automaton 
    def step(self, state: Tensor) -> Tensor:
        device = state.device
        batch = state.shape[0]

        pre_life = self.cell.update_alive_mask(state)
        perceived = self.perception(state)

        dx = self.update(perceived, pre_life, state)

        if self.update.upd_cfg.stochastic_update:
            fire = (torch.rand(batch, 1, *self.cfg.size, device=device) <= self.update.upd_cfg.fire_rate)
            dx = dx * fire.float()

        post_life = self.cell.update_alive_mask(state + dx)
        life_mask = pre_life & post_life
        result = (state + dx) * life_mask.float()

        return result

    def forward(self, state: Tensor, steps: int = 1) -> Tensor:
        for _ in range(steps):
            state = self.step(state)
        return state
