from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .perception import Perception3D, PerceptionConfig
from .cell import CellConfig, CellState

@dataclass(frozen=True)
class UpdateConfig:
    hidden_dim: int = 128
    stochastic_update: bool = False
    fire_rate: float = 0.5

# source: https://github.com/SkyLionx/3d-cellular-automaton
class UpdateRule(nn.Module):
    def __init__(
        self,
        perc_cfg: PerceptionConfig,
        cell_cfg: CellConfig,
        upd_cfg: UpdateConfig,
    ):
        super().__init__()
        self.upd_cfg = upd_cfg

        in_channels = cell_cfg.total_channels * perc_cfg.channel_groups

        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, upd_cfg.hidden_dim, kernel_size=1),
            nn.GroupNorm(4, upd_cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(upd_cfg.hidden_dim, upd_cfg.hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(upd_cfg.hidden_dim, cell_cfg.total_channels, kernel_size=1),
        )

        final_conv = self.mlp[-1]
        if isinstance(final_conv, nn.Conv3d):
            nn.init.zeros_(final_conv.weight)
            if final_conv.bias is not None:
                final_conv.bias.data.fill_(0.0)

    def forward(self, perceived: Tensor, alive_mask: Tensor, state: Tensor) -> Tensor:
        delta = self.mlp(perceived)

        if self.upd_cfg.stochastic_update:
            fire = torch.rand_like(alive_mask.float()) < self.upd_cfg.fire_rate
            delta = delta * fire

        delta = torch.tanh(delta) * 0.1
        return state + delta * alive_mask.float()
