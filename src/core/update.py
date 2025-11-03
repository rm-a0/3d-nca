"""
UpdateRule - MLP that learns how a cell should change.

Input
  - perceived  [B, 3*C, X, Y, Z]   (output of Perception3D)
  - alive_mask [B, 1, X, Y, Z]    (bool, from CellState)
  - state      [B, C, X, Y, Z]    (current world)

Output
  - dx         [B, C, X, Y, Z]    (change to add)

Architecture
  1. 1x1x1 Conv -> hidden_dim
  2. GroupNorm + ReLU
  3. 1x1x1 Conv -> hidden_dim
  4. ReLU
  5. 1x1x1 Conv -> C channels
  6. tanh(·) x 0.1  (bounded update)
  7. (optional) stochastic fire
  8. multiply by alive_mask

Original implementation inspired by:
https://github.com/SkyLionx/3d-cellular-automaton
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .perception import Perception3D, PerceptionConfig
from .cell import CellConfig, CellState

@dataclass(frozen=True)
class UpdateConfig:
    """Configuration for the update MLP."""
    hidden_dim: int = 128
    stochastic_update: bool = False
    fire_rate: float = 0.5

class UpdateRule(nn.Module):
    """
    Two-hidden-layer 1x1x1 MLP with GroupNorm and bounded output.
    """
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
        """
        Predict bounded change `dx` and apply alive-mask.

        Returns updated state:  state + dx * alive_mask
        """
        delta = self.mlp(perceived)

        if self.upd_cfg.stochastic_update:
            fire = torch.rand_like(alive_mask.float()) < self.upd_cfg.fire_rate
            delta = delta * fire

        delta = torch.tanh(delta) * 0.1
        return state + delta * alive_mask.float()
