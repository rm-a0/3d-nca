"""
NCARunner - Training strategy interface and default NCA implementation.

TrainingRunner defines the interface used by NCATrainer.  Implement it to
plug a custom training loop into the server and get live Blender
visualisation without changing any other code.

NCARunner is the default implementation: pool-based curriculum training
with a three-part weighted loss (alpha, colour, overflow).
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Dict, Generator, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .cell import CellConfig
from .grid import Grid3D, GridConfig
from .perception import PerceptionConfig
from .update import UpdateConfig
from .schedule import Schedule, Event, EventType


# --- Data transfer object ---

@dataclass(frozen=True)
class TrainingSnapshot:
    """Immutable point-in-time view of training state for broadcasting and logging.

    Attributes:
        state: Current NCA state in internal (B, C, D, H, W) format.
        epoch: Current training epoch.
        total_epochs: Total number of epochs for this session.
        loss: Latest total loss value.
        visible_channels: Number of visible (RGBA) channels.
        metrics: Optional per-epoch loss breakdown.
    """

    state: np.ndarray
    epoch: int
    total_epochs: int
    loss: float
    visible_channels: int
    metrics: dict[str, float] = field(default_factory=dict)


# --- Strategy interface ---

class TrainingRunner(ABC):
    """Interface for NCA training backends.

    Implement this class to use a custom training loop with NCATrainer.
    The trainer calls these methods from a background thread and handles
    logging and state broadcasting automatically.
    """

    @abstractmethod
    def init(self, config: dict, target: np.ndarray | list[np.ndarray]) -> None:
        """Prepare the runner for a new training session.

        Args:
            config: Nested configuration dict with keys ``cell``, ``perception``,
                ``update``, ``grid``, and ``training``.
            target: Target voxel grid(s) in (D, H, W, C) channels-last format.
        """

    @abstractmethod
    def train(
        self, schedule: Schedule | None = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Run the training loop, yielding a metrics dict each epoch.

        Args:
            schedule: Optional schedule whose events are applied after each epoch.

        Yields:
            Dict containing at least ``loss_total``.
        """

    @abstractmethod
    def snapshot(self) -> TrainingSnapshot:
        """Return a point-in-time snapshot of the current training state.

        Returns:
            TrainingSnapshot with current state tensor, epoch, and metrics.
        """

    @abstractmethod
    def set_target(self, target: Any) -> None:
        """Swap the training target mid-session.

        Args:
            target: New target in the format expected by the implementation.
        """

    def on_event(self, event: Event) -> bool:
        """Handle a schedule event.

        Override to react to learning rate changes, target swaps, loss weight
        adjustments, or custom event types.

        Args:
            event: Scheduled event to handle.

        Returns:
            True if handled, False if ignored.
        """
        return False


# --- Default implementation ---

POOL_SIZE = 32
DEFAULT_BATCH_SIZE = 4

_STEP_MIN_START, _STEP_MIN_END = 8, 32
_STEP_MAX_START, _STEP_MAX_END = 16, 64
_CURRICULUM_EPOCHS = 2000

_DEFAULT_ALPHA_WEIGHT = 4.0
_DEFAULT_COLOR_WEIGHT = 1.0
_DEFAULT_OVERFLOW_WEIGHT = 2.0


class NCARunner(TrainingRunner):
    """Default NCA training implementation.

    Pool-based curriculum training with a three-part weighted loss:
    alpha emergence, colour fidelity, and overflow suppression.

    All internal tensors use (B, C, D, H, W) batch-first format.
    External I/O (targets, snapshot state) uses (D, H, W, C) channels-last.
    """

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
        self._pool_task_ids: List[int] = []
        self._lr_scheduler = None
        self._latest_metrics: Dict[str, float] = {}

    # --- TrainingRunner interface ---

    def init(self, config: dict, target: np.ndarray | list[np.ndarray]) -> None:
        """Initialise runner with config dict and (D, H, W, C) target(s).

        Args:
            config: Nested dict with keys ``cell``, ``perception``, ``update``,
                ``grid``, and ``training``.
            target: Single (D, H, W, C) array or list of arrays for multi-task.
        """
        self._pool = []
        self._pool_task_ids = []

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

        self.targets = self._prepare_targets(target, self._cell_cfg.visible_channels)
        self.target = self.targets[0]

        for _ in range(POOL_SIZE):
            tid = random.randint(0, max(0, len(self.targets) - 1))
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

    def train(
        self, schedule: Schedule | None = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Training loop generator.

        All internal tensors stay in (B, C, D, H, W) batch-first format.

        Args:
            schedule: Optional schedule whose events are applied after each epoch.

        Yields:
            Dict with keys ``loss_total``, ``loss_alpha``, ``loss_color``,
            ``loss_overflow``, and ``best_np``.
        """
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
        """Return a point-in-time snapshot of the current training state.

        Returns:
            TrainingSnapshot with state in (B, C, D, H, W) internal format.

        Raises:
            RuntimeError: If called before init().
        """
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
        """Swap the training target.

        Args:
            target: New target in internal (B, C, D, H, W) format on any device.
        """
        self.target = target.to(self._device)
        if self.verbose:
            print(f"[Runner] Target swapped at epoch {self.current_epoch}")

    def on_event(self, event: Event) -> bool:
        """Apply a schedule event - LR, batch size, loss weights, or target swap.

        Args:
            event: Scheduled event to apply.

        Returns:
            True if handled, False for unrecognised event types.
        """
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

    # --- Internal helpers ---

    def _get_step_range(self) -> tuple[int, int]:
        """Compute curriculum step range for the current epoch.

        Returns:
            Tuple (min_steps, max_steps) for random step count selection.
        """
        t = min(self.current_epoch / max(_CURRICULUM_EPOCHS, 1), 1.0)
        lo = int(_STEP_MIN_START + t * (_STEP_MIN_END - _STEP_MIN_START))
        hi = int(_STEP_MAX_START + t * (_STEP_MAX_END - _STEP_MAX_START))
        return max(lo, 4), max(hi, lo + 1)

    def _step(self) -> Dict[str, Any]:
        """Execute one training epoch: sample pool, forward pass, loss, backward.

        Returns:
            Dict with keys ``loss_total``, ``loss_alpha``, ``loss_color``,
            ``loss_overflow``, and ``best_np`` (lowest-loss sample in (D, H, W, C) format).
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
            per_sample = [
                F.mse_loss(
                    batch[i:i+1, -vis:],
                    self.targets[batch_task_ids[i]][:, -vis:],
                ).item()
                for i in range(batch_size)
            ]
            worst = int(np.argmax(per_sample))
            best_idx = int(np.argmin(per_sample))

            new_tid = random.randint(0, len(self.targets) - 1)
            t_tensor = (
                torch.tensor([new_tid], device=device)
                if cell_cfg.task_channels > 0
                else None
            )
            batch[worst:worst+1] = self.model.seed_center(1, device, t_tensor)
            batch_task_ids[worst] = new_tid

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
                self._pool_task_ids[idx] = batch_task_ids[j]

        self.state = state[0:1].detach()

        return {
            "loss_total": loss.item(),
            "loss_alpha": loss_alpha.item(),
            "loss_color": loss_color.item(),
            "loss_overflow": loss_overflow.item(),
            "best_np": state[best_idx].detach().cpu().numpy().transpose(1, 2, 3, 0),
        }

    def _prepare_targets(
        self,
        target: np.ndarray | list[np.ndarray],
        visible_channels: int,
    ) -> list[Tensor]:
        """Convert external (D, H, W, C) target(s) to internal (1, C, D, H, W) tensors.

        Args:
            target: Single array or list of arrays in (D, H, W, C) format.
            visible_channels: Expected number of channels in each target.

        Returns:
            List of (1, C, D, H, W) tensors on the training device.

        Raises:
            TypeError: If any target is not a numpy array.
            ValueError: If any target has wrong rank or channel count.
        """
        targets = target if isinstance(target, list) else [target]
        if not targets:
            raise ValueError("Target list cannot be empty")
        prepared = []
        for i, t in enumerate(targets):
            if not isinstance(t, np.ndarray):
                raise TypeError(f"Target {i} must be numpy.ndarray, got {type(t).__name__}")
            if t.ndim != 4:
                raise ValueError(f"Target {i} must have shape (D, H, W, C), got {t.ndim}D")
            if t.shape[-1] != visible_channels:
                raise ValueError(
                    f"Target {i} has {t.shape[-1]} channels; "
                    f"expected visible_channels={visible_channels}"
                )
            t_chw = np.transpose(t, (3, 0, 1, 2)).astype(np.float32)
            prepared.append(torch.from_numpy(t_chw).unsqueeze(0).to(self._device))
        return prepared