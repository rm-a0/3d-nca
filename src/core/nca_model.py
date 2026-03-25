"""
NCA - High-level wrapper for the 3D Neural Cellular Automata model.

This is the primary interface for users. Construct once with desired configuration,
and it manages all internal components (Cell, Perception, Update, Grid).

Example usage:
    nca = NCAModel(
        grid_size=(32, 32, 32),
        hidden_channels=16,
        visible_channels=4,
        hidden_dim=128,
    )
    output = nca(state, steps=64)

This follows the standard ML/AI pattern where a single orchestrating class
manages all the internal configurations and components.
"""
from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch import Tensor

from .cell import CellConfig, CellState
from .perception import PerceptionConfig, Perception3D
from .update import UpdateConfig, UpdateRule
from .grid import GridConfig, Grid3D

_CHECKPOINT_VERSION = 1


@dataclass
class NCAConfig:
    """Complete NCA configuration bundled into a single dataclass."""

    # Grid dimensions: (depth, height, width)
    grid_size: Tuple[int, int, int] = (32, 32, 32)

    # Cell channels
    hidden_channels: int = 16
    visible_channels: int = 4
    alive_threshold: float = 0.1

    # Perception (fixed 3x3x3 filters)
    perception_kernel_radius: int = 1
    perception_channel_groups: int = 3

    # Update MLP
    update_hidden_dim: int = 128
    update_stochastic: bool = False
    update_fire_rate: float = 0.5

    def to_configs(self) -> tuple[CellConfig, PerceptionConfig, UpdateConfig, GridConfig]:
        """Convert to individual config objects needed by Grid3D."""
        cell_cfg = CellConfig(
            hidden_channels=self.hidden_channels,
            visible_channels=self.visible_channels,
            alive_threshold=self.alive_threshold,
            task_channels=self.task_channels,
        )
        perc_cfg = PerceptionConfig(
            kernel_radius=self.perception_kernel_radius,
            channel_groups=self.perception_channel_groups,
        )
        upd_cfg = UpdateConfig(
            hidden_dim=self.update_hidden_dim,
            stochastic_update=self.update_stochastic,
            fire_rate=self.update_fire_rate,
        )
        grid_cfg = GridConfig(size=self.grid_size)
        return cell_cfg, perc_cfg, upd_cfg, grid_cfg

    @classmethod
    def from_dict(cls, d: dict) -> "NCAConfig":
        d = dict(d)
        if "grid_size" in d and not isinstance(d["grid_size"], tuple):
            d["grid_size"] = tuple(d["grid_size"])
        known = {f.name for f in dataclasses.fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        return cls(**d)


class NCAModel(torch.nn.Module):
    """
    High-level 3D Neural Cellular Automata model.

    Single entry point for users. Automatically manages:
      - Cell configurations
      - Perception (fixed 3x3x3 filters)
      - Update rule (learned MLP)
      - Grid operations

    Attributes:
        config: The NCAConfig used to initialize this model
        grid: The underlying Grid3D module
    """

    def __init__(self, config: Optional[NCAConfig] = None, **kwargs):
        """
        Initialize NCAModel.

        Args:
            config: NCAConfig object. If None, created from kwargs.
            **kwargs: Used to construct NCAConfig if config is None.
                     Supports all NCAConfig fields (e.g., grid_size, hidden_channels, etc.)

        Example:
            # Using NCAConfig object
            cfg = NCAConfig(grid_size=(64, 64, 64), hidden_channels=32)
            nca = NCAModel(cfg)

            # Using keyword arguments
            nca = NCAModel(grid_size=(64, 64, 64), hidden_channels=32)

            # Using defaults
            nca = NCAModel()
        """
        super().__init__()

        if config is None:
            self.config = NCAConfig(**kwargs)
        else:
            self.config = config

        cell_cfg, perc_cfg, upd_cfg, grid_cfg = self.config.to_configs()
        self.grid = Grid3D(cell_cfg, perc_cfg, upd_cfg, grid_cfg)

    def forward(
        self,
        state: Tensor,
        steps: int = 1,
        use_checkpointing: bool = True,
    ) -> Tensor:
        """
        Run NCAModel for specified number of steps.

        Args:
            state: Current state [B, C, X, Y, Z]
            steps: Number of update steps to perform
            use_checkpointing: Whether to use gradient checkpointing (trades compute for memory)

        Returns:
            Updated state after `steps` iterations [B, C, X, Y, Z]
        """
        return self.grid(state, steps=steps, use_checkpointing=use_checkpointing)

    def seed_center(self, batch_size: int, device: torch.device | str) -> Tensor:
        """
        Create seed state(s) with a living cell at the center.

        Args:
            batch_size: Number of seeds to generate
            device: Device to place tensors on (cuda/cpu)

        Returns:
            Seed state(s) [batch_size, C, X, Y, Z] with alpha=1 at center
        """
        return self.grid.seed_center(batch_size, device)

    def init_empty(self, batch_size: int, device: torch.device | str) -> Tensor:
        """
        Create empty state(s) with all zeros.

        Args:
            batch_size: Number of states to generate
            device: Device to place tensors on (cuda/cpu)

        Returns:
            Empty state(s) [batch_size, C, X, Y, Z]
        """
        return self.grid.init_empty(batch_size, device)

    def save(self, path: str) -> None:
        torch.save({
            "version": _CHECKPOINT_VERSION,
            "config": dataclasses.asdict(self.config),
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None, strict: bool = True) -> "NCAModel":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if ckpt.get("version", 0) > _CHECKPOINT_VERSION:
            raise ValueError(
                f"Checkpoint version {ckpt['version']} is newer than supported version {_CHECKPOINT_VERSION}."
            )
        model = cls(NCAConfig.from_dict(ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"], strict=strict)
        if device is not None:
            model = model.to(device)
        return model

    @property
    def total_channels(self) -> int:
        """Total number of channels (hidden + visible)."""
        return self.config.hidden_channels + self.config.visible_channels

    @property
    def hidden_channels(self) -> int:
        """Number of hidden channels."""
        return self.config.hidden_channels

    @property
    def visible_channels(self) -> int:
        """Number of visible channels (RGBA)."""
        return self.config.visible_channels

    @property
    def grid_size(self) -> Tuple[int, int, int]:
        """Grid dimensions (depth, height, width)."""
        return self.config.grid_size
