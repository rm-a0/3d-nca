from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

@dataclass(frozen=True)
class CellConfig:
    hidden_channels: int = 48
    visible_channels: int = 4
    alive_threshold: int = 0.1

    @property
    def total_channels(self) -> int:
        return self.hidden_channels + self.visible_channels

class CellState(torch.nn.Module):
    def __init__(self, cfg: CellConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "alive_mask", 
            torch.zeros(1, 1, 1, 1, dtype=torch.bool),
            persistent=False
        )

    def total_channels(self) -> int:
        return self.cfg.total_channels

    # source: https://github.com/SkyLionx/3d-cellular-automaton
    @torch.no_grad()
    def update_alive_mask(self, state: Tensor) -> Tensor:
        alpha = state[:, -1,:, ...]
        pooled = F.max_pool3d(alpha, kernel_size=3, stride=1, padding=1)
        alive = pooled > self.cfg.alive_threshold
        self.alive_mask = alive
        return alive
