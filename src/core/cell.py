from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import Tensor

@dataclass(frozen=True)
class CellConfig:
    hidden_channels: int = 48
    visible_channels: int = 4
    alive_threshold: int = 0.1

class CellState(torch.nn.Module):
    def __init__(self, cfg: CellConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "alive_mask", 
            torch.zeros(1, 1, 1, 1, dtype=torch.bool),
            persistent=False
        )

    @property
    def total_channels(self) -> int:
        return self.cfg.hidden_channels + self.cfg.visible_channels

    @torch.no_grad()
    def update_alive_mask(self, state: Tensor) -> Tensor:
        alpha = state[:, -1,:, ...]
        self.alive_mask = alpha > self.cfg.alive_threshold
        return self.alive_mask
