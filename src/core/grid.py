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
        self.perception = Perception3D(perc_cfg, self.cell.total_channels)
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
        state = self.init_empty(batch_size, device)
        center = tuple(s // 2 for s in self.cfg.size)
        seed_vis = torch.rand(batch_size, 4, 1, 1, 1, device=device)
        seed_vis[:, 3:] = 1.0
        state[:, -4:, center[0]:center[0]+1,
                    center[1]:center[1]+1,
                    center[2]:center[2]+1] = seed_vis
        return state
    
    def step(self, state: Tensor) -> Tensor:
        alive = self.cell.update_alive_mask(state)
        perceived = self.perception(state)
        state = self.update(perceived, alive, state)
        return state

    def forward(self, state: Tensor, steps: int = 1) -> Tensor:
        for _ in range(steps):
            state = self.step(state)
        return state
