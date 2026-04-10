"""
Perception3D - fixed 3x3x3 filters that let each cell perceive local context.

Input:  state [B, C_in, D, H, W]
Output: perceived [B, G*C_in, D, H, W]

Default (G=5) filter groups per input channel:
    - Group 0: identity
    - Group 1: 6-neighbor sum
    - Group 2: signed x-gradient (+x minus -x)
    - Group 3: signed y-gradient (+y minus -y)
    - Group 4: signed z-gradient (+z minus -z)

Legacy mode (G=3) remains supported for older configs/checkpoints:
    - identity, 6-neighbor sum, Laplacian

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
    channel_groups: int = 5


class Perception3D(torch.nn.Module):
    """
    Fixed, non-learnable 3x3x3 depthwise convolution.

        For every input channel, produce G output channels where G is configured.
        The default G=5 uses directional signed gradients to break reflection ambiguity.
    """

    def __init__(self, cfg: PerceptionConfig, in_channels: int):
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels

        if cfg.channel_groups not in (3, 5):
            raise ValueError(
                "PerceptionConfig.channel_groups must be 3 (legacy) or 5 (directional), "
                f"got {cfg.channel_groups}"
            )

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

                        channel_groups == 5:
                            Group 0: identity
                            Group 1: 6-neighbor sum
                            Group 2: x-gradient (+x - -x)
                            Group 3: y-gradient (+y - -y)
                            Group 4: z-gradient (+z - -z)

                        channel_groups == 3 (legacy):
                            Group 0: identity
                            Group 1: 6-neighbor sum
                            Group 2: Laplacian

        No normalization - matches Mordvintsev et al. convention.
        """
        with torch.no_grad():
            w = self.depthwise.weight
            w.zero_()
            k = self.cfg.kernel_radius
            g = self.cfg.channel_groups

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

                if g == 5:
                    # Group 2 - signed x-gradient
                    w[base + 2, 0, k + 1, k, k] = 1.0
                    w[base + 2, 0, k - 1, k, k] = -1.0

                    # Group 3 - signed y-gradient
                    w[base + 3, 0, k, k + 1, k] = 1.0
                    w[base + 3, 0, k, k - 1, k] = -1.0

                    # Group 4 - signed z-gradient
                    w[base + 4, 0, k, k, k + 1] = 1.0
                    w[base + 4, 0, k, k, k - 1] = -1.0
                else:
                    # Legacy Group 2 - Laplacian: 6*center - neighbors
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

        Input:  [B, C_in, D, H, W]
        Output: [B, G*C_in, D, H, W]
        """
        return self.depthwise(state)
