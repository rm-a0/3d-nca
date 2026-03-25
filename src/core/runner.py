from __future__ import annotations

import random
from typing import Any, Dict, Generator, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .cell import CellConfig
from .grid import Grid3D, GridConfig
from .perception import PerceptionConfig
from .update import UpdateConfig
from .schedule import Schedule

POOL_SIZE = 32
DEFAULT_BATCH_SIZE = 4

STEP_MIN_START = 8
STEP_MIN_END   = 32
STEP_MAX_START = 16
STEP_MAX_END   = 64
CURRICULUM_EPOCHS = 2000

DEFAULT_ALPHA_WEIGHT    = 4.0
DEFAULT_COLOR_WEIGHT    = 1.0
DEFAULT_OVERFLOW_WEIGHT = 2.0


class NCARunner:
    def __init__(self, verbose: bool = True):
        self.model: Optional[Grid3D] = None
        self.optimizer = None
        self.target: Optional[Tensor] = None
        self.state: Optional[Tensor] = None

        self.current_epoch: int = 0
        self.total_epochs: int = 0
        self.latest_loss: float = 0.0
        self.verbose = verbose

        self._alpha_weight: float    = DEFAULT_ALPHA_WEIGHT
        self._color_weight: float    = DEFAULT_COLOR_WEIGHT
        self._overflow_weight: float = DEFAULT_OVERFLOW_WEIGHT
        self._batch_size: int        = DEFAULT_BATCH_SIZE

        self._cell_cfg: Optional[CellConfig] = None
        self._device: Optional[str] = None
        self._pool: List[Tensor] = []
        self._lr_scheduler = None

    def init(self, config: dict, target: np.ndarray) -> None:
        """Initialize runner with config and target.
        
        Args:
            config: Configuration dict with keys for cell, perception, update, grid, training
            target: Target voxel grid with shape (D, H, W, C) — channels-last external format.
                   Internally converted to (B, C, D, H, W) batch-first for all computations.
                   Transpose: (D,H,W,C) → (C,D,H,W) → unsqueeze → (1,C,D,H,W)
        """
        self._cell_cfg = CellConfig(**config["cell"])
        perc_cfg = PerceptionConfig(**config["perception"])
        upd_cfg  = UpdateConfig(**config["update"])
        grid_cfg = GridConfig(**config["grid"])

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = Grid3D(
            self._cell_cfg, perc_cfg, upd_cfg, grid_cfg
        ).to(self._device)

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

        target_chw = np.transpose(target, (3, 0, 1, 2)).astype(np.float32)  # (D,H,W,C) → (C,D,H,W)
        self.target = torch.from_numpy(target_chw).unsqueeze(0).to(self._device)  # → (1,C,D,H,W)
        # Note: Target now has shape (B=1, C, D, H, W) for internal NCA computations

        self._pool = [
            self.model.seed_center(1, self._device)  # Returns (1, C, D, H, W) — batch-first internal format
            for _ in range(POOL_SIZE)
        ]
        self.state = self._pool[0]

        self.current_epoch = 0
        self.total_epochs  = config["training"]["num_epochs"]
        self._batch_size   = config["training"].get("batch_size", DEFAULT_BATCH_SIZE)
        self.latest_loss   = 0.0

    def set_target(self, target: Tensor) -> None:
        """Update training target.
        
        Args:
            target: Target tensor with shape (B, C, D, H, W) in internal NCA format.
                   Should be on the same device as the model.
        """
        self.target = target.to(self._device)
        if self.verbose:
            print(f"[Runner] Target swapped at epoch {self.current_epoch}")

    def train(self, schedule: Optional[Schedule] = None) -> Generator[Dict[str, Any], None, None]:
        """Training loop generator.
        
        All internal states maintain shape (B, C, D, H, W) — batch-first format required by PyTorch.
        
        Args:
            schedule: Optional Schedule object for training events (target changes, LR updates, etc.).
                     Thread-safe: main thread updates don't block background training loop.
        
        Yields:
            Dict[epoch, loss_alpha, loss_color, loss_overflow, loss_total, best_np]
        """
        for epoch in range(1, self.total_epochs + 1):
            metrics = self._step()
            self.current_epoch = epoch
            self.latest_loss = metrics["loss_total"]
            if schedule is not None:
                schedule.check_and_execute(epoch, self)
            if self.verbose:
                print(f"Epoch {epoch}/{self.total_epochs} - Loss: {metrics['loss_total']:.4f}")
            yield metrics

    def _get_step_range(self) -> tuple[int, int]:
        t = min(self.current_epoch / max(CURRICULUM_EPOCHS, 1), 1.0)
        lo = int(STEP_MIN_START + t * (STEP_MIN_END - STEP_MIN_START))
        hi = int(STEP_MAX_START + t * (STEP_MAX_END - STEP_MAX_START))
        return max(lo, 4), max(hi, lo + 1)

    def _step(self) -> Dict[str, Any]:
        device    = self._device
        cell_cfg  = self._cell_cfg
        batch_size = min(len(self._pool), self._batch_size)
        lo, hi    = self._get_step_range()
        n_steps   = random.randint(lo, hi)

        indices = random.sample(range(len(self._pool)), batch_size)
        batch   = torch.cat([self._pool[i] for i in indices], dim=0)

        vis = cell_cfg.visible_channels
        with torch.no_grad():
            per_sample_loss = [
                F.mse_loss(batch[i:i+1, -vis:], self.target[:, -vis:]).item()
                for i in range(batch_size)
            ]
            worst    = int(np.argmax(per_sample_loss))
            best_idx = int(np.argmin(per_sample_loss))
            batch[worst:worst+1] = self.model.seed_center(1, device)

        self.optimizer.zero_grad()

        state = batch
        state = self.model(state, steps=n_steps)

        pred_vis  = state[:, -vis:]
        tgt_vis   = self.target[:, -vis:].expand_as(pred_vis)

        pred_alpha = pred_vis[:, -1:]
        tgt_alpha  = tgt_vis[:, -1:]

        loss_alpha = F.mse_loss(pred_alpha, tgt_alpha)

        tgt_mask = (tgt_alpha > cell_cfg.alive_threshold).float()
        if vis > 1:
            pred_color = pred_vis[:, :-1]
            tgt_color  = tgt_vis[:, :-1]
            color_diff = (pred_color - tgt_color) ** 2 * tgt_mask
            loss_color = color_diff.sum() / tgt_mask.sum().clamp(min=1.0)
        else:
            loss_color = torch.tensor(0.0, device=device)

        overflow_mask = (1.0 - tgt_mask)
        overflow      = (pred_alpha * overflow_mask) ** 2
        loss_overflow = overflow.mean()

        loss = (
            self._alpha_weight    * loss_alpha
          + self._color_weight    * loss_color
          + self._overflow_weight * loss_overflow
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self._lr_scheduler.step()

        with torch.no_grad():
            for j, idx in enumerate(indices):
                self._pool[idx] = state[j:j+1].detach()

        best_np = (
            state[best_idx]
            .detach().cpu().numpy()
            .transpose(1, 2, 3, 0)
        )

        self.state = state[0:1].detach()

        return {
            "loss_total":    loss.item(),
            "loss_alpha":    loss_alpha.item(),
            "loss_color":    loss_color.item(),
            "loss_overflow": loss_overflow.item(),
            "best_np":       best_np,
        }