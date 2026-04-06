"""Baseline default runner.

This strategy is intentionally task-agnostic and used as the standard
single-target 3D NCA training path.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Generator, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..cell import CellConfig
from ..grid import Grid3D, GridConfig
from ..perception import PerceptionConfig
from ..schedule import Event, EventType, Schedule
from ..update import UpdateConfig
from .base import NCARunner, TrainingSnapshot

POOL_SIZE = 32
DEFAULT_BATCH_SIZE = 4

_STEP_MIN_START, _STEP_MIN_END = 8, 32
_STEP_MAX_START, _STEP_MAX_END = 16, 64
_CURRICULUM_EPOCHS = 2000

_DEFAULT_ALPHA_WEIGHT = 4.0
_DEFAULT_COLOR_WEIGHT = 1.0
_DEFAULT_OVERFLOW_WEIGHT = 2.0


class MorphRunner(NCARunner):
    """Baseline pool runner with task-channel experiments disabled."""

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

        self.model: Optional[Grid3D] = None
        self.optimizer = None
        self.target: Optional[Tensor] = None
        self.state: Optional[Tensor] = None
        self.current_epoch: int = 0
        self.total_epochs: int = 0
        self.latest_loss: float = 0.0

        self._alpha_weight: float = _DEFAULT_ALPHA_WEIGHT
        self._color_weight: float = _DEFAULT_COLOR_WEIGHT
        self._overflow_weight: float = _DEFAULT_OVERFLOW_WEIGHT
        self._batch_size: int = DEFAULT_BATCH_SIZE

        self._cell_cfg: Optional[CellConfig] = None
        self._device: Optional[str] = None
        self._pool: List[Tensor] = []
        self._lr_scheduler = None
        self._latest_metrics: Dict[str, float] = {}

    def init(self, config: dict, target: np.ndarray | list[np.ndarray]) -> None:
        task_channels = int(config.get("cell", {}).get("task_channels", 0))
        if task_channels != 0:
            raise ValueError(
                "MorphRunner requires cell.task_channels == 0. "
            )

        if isinstance(target, list):
            if not target:
                raise ValueError("Target list cannot be empty")
            if len(target) != 1:
                raise ValueError(
                    "MorphRunner expects a single target. "
                )
            target = target[0]

        self._pool = []

        self._cell_cfg = CellConfig(**config["cell"])
        perc_cfg = PerceptionConfig(**config["perception"])
        upd_cfg = UpdateConfig(**config["update"])
        grid_cfg = GridConfig(**config["grid"])

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Grid3D(self._cell_cfg, perc_cfg, upd_cfg, grid_cfg).to(self._device)

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

        self.target = self._prepare_target(target, self._cell_cfg.visible_channels)

        for _ in range(POOL_SIZE):
            self._pool.append(self.model.seed_center(1, self._device, None))

        self.state = self._pool[0]
        self.current_epoch = 0
        self.total_epochs = config["training"]["num_epochs"]
        self._batch_size = config["training"].get("batch_size", DEFAULT_BATCH_SIZE)
        self.latest_loss = 0.0
        self._latest_metrics = {}

    def train(
        self, schedule: Schedule | None = None
    ) -> Generator[Dict[str, Any], None, None]:
        for epoch in range(1, self.total_epochs + 1):
            metrics = self._step()
            self.current_epoch = epoch
            self.latest_loss = metrics["loss_total"]
            self._latest_metrics = {
                k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))
            }

            if schedule is not None:
                schedule.check_and_execute(epoch, self)

            if self.verbose:
                print(f"Epoch {epoch}/{self.total_epochs} - Loss: {metrics['loss_total']:.4f}")

            yield metrics

    def snapshot(self) -> TrainingSnapshot:
        if self.state is None or self._cell_cfg is None:
            raise RuntimeError("Runner not initialised - call init() first")
        return TrainingSnapshot(
            state=self.state.detach().cpu().numpy().astype(np.float32),
            epoch=self.current_epoch,
            total_epochs=self.total_epochs,
            loss=self.latest_loss,
            visible_channels=self._cell_cfg.visible_channels,
            metrics=dict(self._latest_metrics),
        )

    def set_target(self, target: Tensor) -> None:
        self.target = target.to(self._device)
        if self.verbose:
            print(f"[Runner] Target swapped at epoch {self.current_epoch}")

    def on_event(self, event: Event) -> bool:
        if self.optimizer is None:
            return False

        t, v = event.event_type, float(event.value)

        if t == EventType.LEARNING_RATE:
            for pg in self.optimizer.param_groups:
                pg["lr"] = v

        elif t == EventType.BATCH_SIZE:
            self._batch_size = max(1, int(v))

        elif t == EventType.ALPHA_WEIGHT:
            self._alpha_weight = v

        elif t == EventType.COLOR_WEIGHT:
            self._color_weight = v

        elif t == EventType.OVERFLOW_WEIGHT:
            self._overflow_weight = v

        elif t == EventType.TARGET_CHANGE:
            if event.target is None:
                raise ValueError("TARGET_CHANGE event requires a target array")
            t_chw = np.transpose(event.target, (3, 0, 1, 2)).astype(np.float32)
            self.set_target(torch.from_numpy(t_chw).unsqueeze(0))

        else:
            return False

        return True

    def _get_step_range(self) -> tuple[int, int]:
        t = min(self.current_epoch / max(_CURRICULUM_EPOCHS, 1), 1.0)
        lo = int(_STEP_MIN_START + t * (_STEP_MIN_END - _STEP_MIN_START))
        hi = int(_STEP_MAX_START + t * (_STEP_MAX_END - _STEP_MAX_START))
        return max(lo, 4), max(hi, lo + 1)

    def _step(self) -> Dict[str, Any]:
        device = self._device
        cell_cfg = self._cell_cfg
        batch_size = min(len(self._pool), self._batch_size)
        lo, hi = self._get_step_range()
        n_steps = random.randint(lo, hi)

        indices = random.sample(range(len(self._pool)), batch_size)
        batch = torch.cat([self._pool[i] for i in indices], dim=0)

        vis = cell_cfg.visible_channels
        tgt_batch = self.target.expand(batch_size, -1, -1, -1, -1)

        with torch.no_grad():
            per_sample = [
                F.mse_loss(
                    batch[i:i+1, -vis:],
                    self.target[:, -vis:],
                ).item()
                for i in range(batch_size)
            ]
            worst = int(np.argmax(per_sample))
            best_idx = int(np.argmin(per_sample))
            batch[worst:worst+1] = self.model.seed_center(1, device, None)

        self.optimizer.zero_grad()
        state = self.model(batch, steps=n_steps)

        pred_vis = state[:, -vis:]
        tgt_vis = tgt_batch[:, -vis:]
        pred_alpha = pred_vis[:, -1:]
        tgt_alpha = tgt_vis[:, -1:]

        loss_alpha = F.mse_loss(pred_alpha, tgt_alpha)
        tgt_mask = (tgt_alpha > cell_cfg.alive_threshold).float()

        if vis > 1:
            color_diff = (pred_vis[:, :-1] - tgt_vis[:, :-1]) ** 2 * tgt_mask
            loss_color = color_diff.sum() / tgt_mask.sum().clamp(min=1.0)
        else:
            loss_color = torch.tensor(0.0, device=device)

        loss_overflow = ((pred_alpha * (1.0 - tgt_mask)) ** 2).mean()

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
                self._pool[idx] = state[j:j+1].detach()

        self.state = state[0:1].detach()

        return {
            "loss_total": loss.item(),
            "loss_alpha": loss_alpha.item(),
            "loss_color": loss_color.item(),
            "loss_overflow": loss_overflow.item(),
            "best_np": state[best_idx].detach().cpu().numpy().transpose(1, 2, 3, 0),
        }

    def _prepare_target(
        self,
        target: np.ndarray,
        visible_channels: int,
    ) -> Tensor:
        if not isinstance(target, np.ndarray):
            raise TypeError(f"Target must be numpy.ndarray, got {type(target).__name__}")
        if target.ndim != 4:
            raise ValueError(f"Target must have shape (D, H, W, C), got {target.ndim}D")
        if target.shape[-1] != visible_channels:
            raise ValueError(
                f"Target has {target.shape[-1]} channels; expected visible_channels={visible_channels}"
            )
        t_chw = np.transpose(target, (3, 0, 1, 2)).astype(np.float32)
        return torch.from_numpy(t_chw).unsqueeze(0).to(self._device)
