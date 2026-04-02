"""
Perception3D - fixed 3x3x3 filters that let each cell percept its neighbors.

Input:  state [B, C, X, Y, Z]
Output: perceived [B, 3*C, X, Y, Z]

Three filter groups (depthwise conv):
  - Identity: the cell itself
  - Neighbor sum: sum of 6 direct neighbors
  - Gradient: center - neighbors (edges)

Each group has C channels -> output is 3xC.

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

        # Keep perception kernels fixed (matches the intended NCA formulation).
        self.depthwise.weight.requires_grad_(False)

    def _init_perception_kernels(self) -> None:
        """Initialize the fixed perception weights (no gradients).

        Weight shape: [out_channels, 1, kD, kH, kW]
        out_channels = in_channels * channel_groups.
        With groups=in_channels, for input channel i the 3 output
        filters sit at indices 3*i+0, 3*i+1, 3*i+2 along dim 0.

            Group 0 (offset 0): identity - center voxel only
            Group 1 (offset 1): 6-neighbor sum
            Group 2 (offset 2): Laplacian (center - neighbors)

        No normalization - matches Mordvintsev et al. convention.
        """
        with torch.no_grad():
            w = self.depthwise.weight  # [3*C, 1, 3, 3, 3]
            w.zero_()
            k = self.cfg.kernel_radius  # center index in kernel
            g = self.cfg.channel_groups  # 3

            for c in range(self.in_channels):
                base = c * g

                # Group 0 - identity
                w[base + 0, 0, k, k, k] = 1.0

                # Group 1 - sum of 6 face-adjacent neighbors
                for dx, dy, dz in [
                    (1, 0, 0),
                    (-1, 0, 0),
                    (0, 1, 0),
                    (0, -1, 0),
                    (0, 0, 1),
                    (0, 0, -1),
                ]:
                    w[base + 1, 0, k + dx, k + dy, k + dz] = 1.0

                # Group 2 - Laplacian: 6*center - neighbors
                w[base + 2, 0, k, k, k] = 6.0
                for dx, dy, dz in [
                    (1, 0, 0),
                    (-1, 0, 0),
                    (0, 1, 0),
                    (0, -1, 0),
                    (0, 0, 1),
                    (0, 0, -1),
                ]:
                    w[base + 2, 0, k + dx, k + dy, k + dz] = -1.0

    def forward(self, state: Tensor) -> Tensor:
        """
        Apply the fixed perception filters.

        Input:  [B, C, X, Y, Z]
        Output: [B, 3*C, X, Y, Z]
        """
        return self.depthwise(state)
