"""
Perception3D - fixed 3x3x3 filters that let each cell percept its neighbors.

Input:  state [B, C, X, Y, Z]
Output: perceived [B, 3*C, X, Y, Z]

Three filter groups (depthwise conv):
  - Identity: the cell itself
  - Neighbor sum: sum of 6 direct neighbors
  - Gradient: center - neighbors (edges)

Each group has C channels → output is 3xC.

Original implementation heavily inspired by:
https://github.com/SkyLionx/3d-cellular-automaton
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import Tensor

@dataclass(frozen=True)
class PerceptionConfig:
    """Configuration for perception filters."""
    kernel_radius: int = 1
    channel_groups: int = 3

class Perception3D(torch.nn.Module):
    """
    Fixed, non-learnable 3x3x3 depthwise convolution.

    For every input channel we produce three output channels:
      - identity
      - sum of the 6 direct neighbors
      - center-minus-neighbors (gradient)
    """
    def __init__(self, cfg: PerceptionConfig, in_channels: int):
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels

        kernel_size = 2 * cfg.kernel_radius + 1
        out_channels = in_channels * cfg.channel_groups

        self.depthwise = torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=cfg.kernel_radius,
            groups=in_channels,
            bias=False,
        )

        self._init_perception_kernels()

    def _init_perception_kernels(self) -> None:
        """Initialize the fixed perception weights (no gradients).

            1. Group 0 : identity (center voxel)
            2. Group 1 : sum of 6 direct neighbours (positions relative to centre)
            3. Group 2 : gradient (center - neighbours)
            4. Normalise so each group sums to 1
        """
        with torch.no_grad():
            w = self.depthwise.weight
            w.zero_()
            k = self.cfg.kernel_radius
            g = self.cfg.channel_groups

            w[:, 0:g, k, k, k] = 1.0

            w[:, g:2*g, k, k, k] = 1.0
            for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                w[:, g:2*g, k+dx, k+dy, k+dz] = 1.0

            w[:, 2*g:, k, k, k] = -1.0
            for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                w[:, 2*g:, k+dx, k+dy, k+dz] = 1.0

            norm = w.abs().sum(dim=(2,3,4), keepdim=True).clamp(min=1e-6)
            w.div_(norm)

    def forward(self, state: Tensor) -> Tensor:
        """
        Apply the fixed perception filters.

        Input:  [B, C, X, Y, Z]  
        Output: [B, 3*C, X, Y, Z]
        """
        return self.depthwise(state)