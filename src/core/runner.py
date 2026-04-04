"""
NCARunner - Training harness for 3D Neural Cellular Automata.

Manages the complete training loop:
  - Model initialization and device management
  - Pool-based state management (diverse training samples)
  - Loss computation (alpha, color, overflow weighted components)
  - Curriculum learning (step count varies with epoch)
  - Integration with Schedule for dynamic training events

All internal states maintain (B, C, D, H, W) batch-first tensor format
required by PyTorch. External I/O (targets, exports) uses (D, H, W, C)
channels-last format.
"""

from __future__ import annotations

import random
from numbers import Integral
from typing import Any, Dict, Generator, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .cell import CellConfig
from .grid import Grid3D, GridConfig
from .perception import PerceptionConfig
from .runtime import BaseTrainingRuntime, TrainingSnapshot
from .update import UpdateConfig
from .schedule import Schedule

POOL_SIZE = 32
DEFAULT_BATCH_SIZE = 4

STEP_MIN_START = 8
STEP_MIN_END = 32
STEP_MAX_START = 16
STEP_MAX_END = 64
CURRICULUM_EPOCHS = 2000

DEFAULT_ALPHA_WEIGHT = 4.0
DEFAULT_COLOR_WEIGHT = 1.0
DEFAULT_OVERFLOW_WEIGHT = 2.0


class NCARunner(BaseTrainingRuntime):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose=verbose)
        self.model: Optional[Grid3D] = None
        self.optimizer = None
        self.target: Optional[Tensor] = None
        self.state: Optional[Tensor] = None

        self.current_epoch: int = 0
        self.total_epochs: int = 0
        self.latest_loss: float = 0.0

        self._alpha_weight: float = DEFAULT_ALPHA_WEIGHT
        self._color_weight: float = DEFAULT_COLOR_WEIGHT
        self._overflow_weight: float = DEFAULT_OVERFLOW_WEIGHT
        self._batch_size: int = DEFAULT_BATCH_SIZE

        self._cell_cfg: Optional[CellConfig] = None
        self._device: Optional[str] = None
        self._pool: List[Tensor] = []
        self._pool_task_ids: List[int] = []
        self._lr_scheduler = None
        self._latest_metrics: Dict[str, float] = {}

    def init(self, config: dict, target: np.ndarray | list[np.ndarray]) -> None:
        """Initialize runner with config and target.

        Args:
            config: Configuration dict with keys for cell, perception, update, grid, training
            target: Target voxel grid with shape (D, H, W, C) in channels-last external format.
                Internally converted to (B, C, D, H, W) batch-first for all computations.
                Transpose: (D,H,W,C) -> (C,D,H,W) -> unsqueeze -> (1,C,D,H,W)
        """
        self._validate_config(config)

        # Reset pool state so repeated init() calls do not accumulate old samples.
        self._pool = []
        self._pool_task_ids = []

        self._cell_cfg = CellConfig(**config["cell"])
        perc_cfg = PerceptionConfig(**config["perception"])
        upd_cfg = UpdateConfig(**config["update"])
        grid_cfg = GridConfig(**config["grid"])

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = Grid3D(self._cell_cfg, perc_cfg, upd_cfg, grid_cfg).to(
            self._device
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=1e-5,
        )
        self._lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config["training"]["num_epochs"],
            eta_min=1e-5,
        )

        self.targets = self._prepare_targets(target, self._cell_cfg.visible_channels)

        self.target = self.targets[0]  # Fallback to single target

        num_tasks = len(self.targets)
        # Note: Target now has shape (B=1, C, D, H, W) for internal NCA computations

        for _ in range(POOL_SIZE):
            tid = random.randint(0, max(0, num_tasks - 1))
            t_tensor = (
                torch.tensor([tid], device=self._device)
                if self._cell_cfg.task_channels > 0
                else None
            )
            self._pool.append(self.model.seed_center(1, self._device, t_tensor))
            self._pool_task_ids.append(tid)

        self.state = self._pool[0]

        self.current_epoch = 0
        self.total_epochs = config["training"]["num_epochs"]
        self._batch_size = config["training"].get("batch_size", DEFAULT_BATCH_SIZE)
        self.latest_loss = 0.0
        self._latest_metrics = {}

    def _validate_config(self, config: dict) -> None:
        if not isinstance(config, dict):
            raise TypeError(f"Config must be a dict, got {type(config).__name__}")

        required_sections = ("cell", "perception", "update", "grid", "training")
        missing_sections = [name for name in required_sections if name not in config]
        if missing_sections:
            raise ValueError(
                f"Missing required config section(s): {', '.join(missing_sections)}"
            )

        for section_name in required_sections:
            section_value = config[section_name]
            if not isinstance(section_value, dict):
                raise TypeError(
                    f"Config section '{section_name}' must be a dict, got {type(section_value).__name__}"
                )

        grid_size = config["grid"].get("size")
        if not isinstance(grid_size, (list, tuple)) or len(grid_size) != 3:
            raise ValueError("Config section 'grid' must define a 3D 'size' tuple")
        if any(not isinstance(dim, Integral) or dim <= 0 for dim in grid_size):
            raise ValueError("Grid dimensions must be positive integers")

        training_cfg = config["training"]
        learning_rate = training_cfg.get("learning_rate")
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError("Config section 'training.learning_rate' must be positive")

        num_epochs = training_cfg.get("num_epochs")
        if not isinstance(num_epochs, Integral) or num_epochs <= 0:
            raise ValueError(
                "Config section 'training.num_epochs' must be a positive integer"
            )

        batch_size = training_cfg.get("batch_size", DEFAULT_BATCH_SIZE)
        if not isinstance(batch_size, Integral) or batch_size <= 0:
            raise ValueError(
                "Config section 'training.batch_size' must be a positive integer"
            )

    def _prepare_targets(
        self,
        target: np.ndarray | list[np.ndarray],
        visible_channels: int,
    ) -> list[Tensor]:
        target_list = target if isinstance(target, list) else [target]
        if not target_list:
            raise ValueError("Target list cannot be empty")

        prepared_targets: list[Tensor] = []
        for index, t in enumerate(target_list):
            if not isinstance(t, np.ndarray):
                raise TypeError(
                    f"Target {index} must be a numpy.ndarray, got {type(t).__name__}"
                )
            if t.ndim != 4:
                raise ValueError(
                    f"Target {index} must have shape (D, H, W, C), got {t.ndim}D"
                )
            if t.shape[-1] != visible_channels:
                raise ValueError(
                    f"Target {index} channel count {t.shape[-1]} does not match visible_channels={visible_channels}"
                )

            t_chw = np.transpose(t, (3, 0, 1, 2)).astype(
                np.float32
            )  # (D,H,W,C) -> (C,D,H,W)
            prepared_targets.append(
                torch.from_numpy(t_chw).unsqueeze(0).to(self._device)
            )  # (1,C,D,H,W)

        return prepared_targets

    def set_target(self, target: Tensor) -> None:
        """Update training target.

        Args:
            target: Target tensor with shape (B, C, D, H, W) in internal NCA format.
                Should be on the same device as the model.
        """
        self.target = target.to(self._device)
        if self.verbose:
            print(f"[Runner] Target swapped at epoch {self.current_epoch}")

    def snapshot(self) -> TrainingSnapshot:
        """Return a stable snapshot for broadcasting and logging."""
        if self.state is None or self._cell_cfg is None:
            raise RuntimeError("Runner is not initialized")

        state_np = self.state.detach().cpu().numpy().astype(np.float32)
        return TrainingSnapshot(
            state=state_np,
            epoch=self.current_epoch,
            total_epochs=self.total_epochs,
            loss=self.latest_loss,
            visible_channels=self._cell_cfg.visible_channels,
            metrics=dict(self._latest_metrics),
        )

    def apply_schedule_event(self, event: Any) -> bool:
        """Apply one schedule event through the runtime's public boundary."""
        from .schedule import EventType

        if self.optimizer is None or self._cell_cfg is None:
            return False

        event_type = event.event_type
        value = float(event.value)

        if event_type == EventType.LEARNING_RATE:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = value
            return True
        if event_type == EventType.BATCH_SIZE:
            self._batch_size = max(1, int(value))
            return True
        if event_type == EventType.ALPHA_WEIGHT:
            self._alpha_weight = value
            return True
        if event_type == EventType.COLOR_WEIGHT:
            self._color_weight = value
            return True
        if event_type == EventType.OVERFLOW_WEIGHT:
            self._overflow_weight = value
            return True
        if event_type == EventType.TARGET_CHANGE:
            if event.target is None:
                raise ValueError("TARGET_CHANGE event requires a target array")
            import torch

            target_chw = np.transpose(event.target, (3, 0, 1, 2)).astype(np.float32)
            self.set_target(torch.from_numpy(target_chw).unsqueeze(0))
            return True

        return False

    @property
    def supports_schedule_events(self) -> bool:
        return True

    def train(
        self, schedule: Optional[Schedule] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Training loop generator.

        All internal states maintain shape (B, C, D, H, W) - batch-first format required by PyTorch.

        Args:
            schedule: Optional Schedule object for training events (target changes, LR updates, etc.).
                Thread-safe: main thread updates don't block background training loop.

        Yields:
            Dict[epoch, loss_alpha, loss_color, loss_overflow, loss_total, best_np]
        """
        self._begin_training_loop()
        try:
            for epoch in range(1, self.total_epochs + 1):
                if self.stop_requested:
                    break

                self._wait_if_paused()
                if self.stop_requested:
                    break

                metrics = self._step()
                self.current_epoch = epoch
                self.latest_loss = metrics["loss_total"]
                self._latest_metrics = {
                    key: float(value)
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                }

                if schedule is not None and self.supports_schedule_events:
                    schedule.check_and_execute(epoch, self)

                if self.verbose:
                    print(
                        f"Epoch {epoch}/{self.total_epochs} - Loss: {metrics['loss_total']:.4f}"
                    )
                yield metrics
        finally:
            self._end_training_loop()

    def _get_step_range(self) -> tuple[int, int]:
        """Compute curriculum learning step range for current epoch.

        Uses linear interpolation from STEP_MIN_START to STEP_MIN_END over
        CURRICULUM_EPOCHS to gradually increase training difficulty.

        Returns:
            Tuple (min_steps, max_steps) for random.randint selection this epoch.
        """
        t = min(self.current_epoch / max(CURRICULUM_EPOCHS, 1), 1.0)
        lo = int(STEP_MIN_START + t * (STEP_MIN_END - STEP_MIN_START))
        hi = int(STEP_MAX_START + t * (STEP_MAX_END - STEP_MAX_START))
        return max(lo, 4), max(hi, lo + 1)

    def _step(self) -> Dict[str, Any]:
        """Execute one training epoch: sample batch, forward pass, loss, backward.

        1. Select random batch from state pool
        2. Replace worst loss sample with fresh seed (curriculum)
        3. Forward NCA for random step count (curriculum)
        4. Compute weighted component losses: alpha + color + overflow
        5. Backward pass, gradient clipping, optimizer step
        6. Update pool with new states

        Returns:
            Dict with keys: loss_total, loss_alpha, loss_color, loss_overflow, best_np.
            best_np is the lowest-loss sample output in (D,H,W,C) external format.
        """
        device = self._device
        cell_cfg = self._cell_cfg
        batch_size = min(len(self._pool), self._batch_size)
        lo, hi = self._get_step_range()
        n_steps = random.randint(lo, hi)

        indices = random.sample(range(len(self._pool)), batch_size)
        batch = torch.cat([self._pool[i] for i in indices], dim=0)
        batch_task_ids = [self._pool_task_ids[i] for i in indices]

        vis = cell_cfg.visible_channels

        tgt_batch = torch.cat([self.targets[tid] for tid in batch_task_ids], dim=0)

        with torch.no_grad():
            per_sample_loss = [
                F.mse_loss(
                    batch[i : i + 1, -vis:],
                    self.targets[batch_task_ids[i]][:, -vis:],
                ).item()
                for i in range(batch_size)
            ]
            worst = int(np.argmax(per_sample_loss))
            best_idx = int(np.argmin(per_sample_loss))

            new_tid = random.randint(0, len(self.targets) - 1)
            t_tensor = (
                torch.tensor([new_tid], device=device)
                if cell_cfg.task_channels > 0
                else None
            )
            batch[worst : worst + 1] = self.model.seed_center(1, device, t_tensor)
            batch_task_ids[worst] = new_tid

        self.optimizer.zero_grad()

        state = batch
        state = self.model(state, steps=n_steps)

        pred_vis = state[:, -vis:]
        tgt_vis = tgt_batch[:, -vis:]

        pred_alpha = pred_vis[:, -1:]
        tgt_alpha = tgt_vis[:, -1:]

        loss_alpha = F.mse_loss(pred_alpha, tgt_alpha)

        tgt_mask = (tgt_alpha > cell_cfg.alive_threshold).float()
        if vis > 1:
            pred_color = pred_vis[:, :-1]
            tgt_color = tgt_vis[:, :-1]
            color_diff = (pred_color - tgt_color) ** 2 * tgt_mask
            loss_color = color_diff.sum() / tgt_mask.sum().clamp(min=1.0)
        else:
            loss_color = torch.tensor(0.0, device=device)

        overflow_mask = 1.0 - tgt_mask
        overflow = (pred_alpha * overflow_mask) ** 2
        loss_overflow = overflow.mean()

        loss = (
            self._alpha_weight * loss_alpha
            + self._color_weight * loss_color
            + self._overflow_weight * loss_overflow
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self._lr_scheduler.step()

        with torch.no_grad():
            for j, idx in enumerate(indices):
                self._pool[idx] = state[j : j + 1].detach()
                self._pool_task_ids[idx] = batch_task_ids[j]

        best_np = state[best_idx].detach().cpu().numpy().transpose(1, 2, 3, 0)

        self.state = state[0:1].detach()

        return {
            "loss_total": loss.item(),
            "loss_alpha": loss_alpha.item(),
            "loss_color": loss_color.item(),
            "loss_overflow": loss_overflow.item(),
            "best_np": best_np,
        }
