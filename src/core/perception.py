from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import Tensor

@dataclass(frozen=True)
class PerceptionConfig:
    kernel_radius: int = 1
    channel_groups: int = 3

class Perception3D(torch.nn.Module):
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
        with torch.no_grad():
            weight = self.depthwise.weight
            weight.zero_()
            k = self.cfg.kernel_radius
            g = self.cfg.channel_groups

            def init_identity() -> None:
                weight[:, 0:g, k, k, k] = 1.0

            def init_local_average() -> None:
                weight[:, g:2*g, k, k, k] = 1.0
                for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                    weight[:, g:2*g, k+dx, k+dy, k+dz] = 1.0

            def init_gradient() -> None:
                weight[:, 2*g:, k, k, k] = -1.0
                for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                    weight[:, 2*g:, k+dx, k+dy, k+dz] = 1.0

            def normalize() -> None:
                norm = weight.abs().sum(dim=(2, 3, 4), keepdim=True).clamp(min=1e-6)
                weight.div_(norm)

            init_identity()
            init_local_average()
            init_gradient()
            normalize()

    def forward(self, state: Tensor) -> Tensor:
        return self.depthwise(state)