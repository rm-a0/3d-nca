import random
import threading
import time
from typing import Any, Callable, Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from src.core import Grid3D, CellConfig, PerceptionConfig, UpdateConfig, GridConfig
from .protocol import build_state_msg
from .schedule import Event, Schedule

SendFn = Callable[[Dict[str, Any]], None] # type alias for callback

# Hardcoded training hyper-parameters for simplicity. These could be made configurable later.
# Training hyper-parameters (pool-based, per Mordvintsev et al.)
POOL_SIZE = 32            # number of persistent states
DEFAULT_BATCH_SIZE = 4    # samples per training iteration (overridden by config)

# Steps ramp linearly from STEP_MIN_START to STEP_MIN_END over CURRICULUM_EPOCHS
STEP_MIN_START = 8        # early training: few steps (learn core shape)
STEP_MIN_END   = 32       # late training: more steps  (refine details)
STEP_MAX_START = 16
STEP_MAX_END   = 64
CURRICULUM_EPOCHS = 2000

# Default loss weights (used to initialise trainer instance attributes)
DEFAULT_ALPHA_WEIGHT    = 4.0   # shape/occupancy matters most
DEFAULT_COLOR_WEIGHT    = 1.0   # color loss only where target is alive
DEFAULT_OVERFLOW_WEIGHT = 2.0   # penalise alive cells outside target

class NCATrainer:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.target = None
        self.state = None

        self.current_epoch = 0
        self.total_epochs = 0
        self.latest_loss = 0.0

        # Mutable loss weights (can be changed by the schedule mid-training)
        self._alpha_weight: float    = DEFAULT_ALPHA_WEIGHT
        self._color_weight: float    = DEFAULT_COLOR_WEIGHT
        self._overflow_weight: float = DEFAULT_OVERFLOW_WEIGHT

        self._schedule = Schedule()

        self._pool: List[torch.Tensor] = []   # state pool

        self._train_thread = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()

    def init(self, config: dict, target: np.ndarray, send_fn: SendFn):
        self._cell_cfg = CellConfig(**config["cell"])
        self._perc_cfg = PerceptionConfig(**config["perception"])
        self._upd_cfg = UpdateConfig(**config["update"])
        self._grid_cfg = GridConfig(**config["grid"])

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = Grid3D(
            self._cell_cfg, self._perc_cfg, self._upd_cfg, self._grid_cfg
        ).to(self._device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config["training"]["learning_rate"],
            weight_decay=1e-5,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config["training"]["num_epochs"],
            eta_min=1e-5,
        )

        # (D,H,W,C) channels-last
        target_chw = np.transpose(target, (3, 0, 1, 2)).astype(np.float32)
        self.target = torch.from_numpy(target_chw).unsqueeze(0).to(self._device)

        # Initialize pool with seed states
        self._pool = [
            self.model.seed_center(1, self._device)
            for _ in range(POOL_SIZE)
        ]
        self.state = self._pool[0]

        self.current_epoch = 0
        self.total_epochs = config["training"]["num_epochs"]
        self._batch_size = config["training"].get("batch_size", DEFAULT_BATCH_SIZE)
        self.latest_loss = 0.0
        self._send_fn = send_fn

        self._stop_event.clear()
        self._pause_event.set()
        self._train_thread = threading.Thread(
            target=self._training_loop
        )
        self._train_thread.start()


    def pause(self):
        self._pause_event.clear()
        print("Training paused")

    def resume(self):
        self._pause_event.set()
        print("Training resumed")

    def update_schedule(self, events_data: List[dict]) -> None:
        """Replace the current schedule."""
        events = [Event.from_dict(d) for d in events_data]
        self._schedule.replace(events)
        print(f"Schedule updated: {len(self._schedule.events)} pending event(s)")

    def stop(self):
        self._stop_event.set()
        self._pause_event.set()
        if self._train_thread is not None:
            self._train_thread.join()
        print("Training stopped")

    @property
    def is_running(self):
        return self._train_thread is not None and self._train_thread.is_alive()
    
    @property
    def is_paused(self):
        return not self._pause_event.is_set()
    
    def get_current_state(self):
        return self.state
    
    def _training_loop(self):
        for epoch in range(1, self.total_epochs + 1):
            self._pause_event.wait()
            if self._stop_event.is_set():
                break

            loss = self._step()
            self.current_epoch = epoch
            self.latest_loss = loss
            self._schedule.check_and_execute(epoch, self)
            self._send_state()
            print(f"Epoch {epoch}/{self.total_epochs} - Loss: {loss:.4f}")
            time.sleep(0.1)  # throttle to avoid flooding the socket

        print("Training completed")

    def _get_step_range(self) -> tuple[int, int]:
        """Return (min_steps, max_steps) for the current epoch via curriculum."""
        t = min(self.current_epoch / max(CURRICULUM_EPOCHS, 1), 1.0)
        lo = int(STEP_MIN_START + t * (STEP_MIN_END - STEP_MIN_START))
        hi = int(STEP_MAX_START + t * (STEP_MAX_END - STEP_MAX_START))
        return max(lo, 4), max(hi, lo + 1)

    def _step(self):
        """One pool-based training iteration with structured loss & curriculum.

        1. Compute step count from curriculum
        2. Sample a batch from the pool
        3. Replace the highest-loss sample with a fresh seed
        4. Forward pass
        5. Structured loss: alpha + color + overflow
        6. Write updated states back to the pool
        """
        device = self._device
        cell_cfg = self._cell_cfg
        batch_size = min(len(self._pool), self._batch_size)
        lo, hi = self._get_step_range()
        n_steps = random.randint(lo, hi)

        # sample batch from pool
        indices = random.sample(range(len(self._pool)), batch_size)
        batch = torch.cat([self._pool[i] for i in indices], dim=0)

        # replace the highest-loss element with a fresh seed
        vis = cell_cfg.visible_channels
        with torch.no_grad():
            per_sample_loss = [
                F.mse_loss(batch[i:i+1, -vis:], self.target[:, -vis:]).item()
                for i in range(batch_size)
            ]
            worst = int(np.argmax(per_sample_loss))
            batch[worst:worst+1] = self.model.seed_center(1, device)

        self.optimizer.zero_grad()

        # forward pass
        state = batch
        state = self.model(state, steps=n_steps)

        # structured loss
        pred_vis = state[:, -vis:]                         # [B, V, X, Y, Z]
        tgt_vis  = self.target[:, -vis:].expand_as(pred_vis)

        pred_alpha = pred_vis[:, -1:]                      # [B, 1, X, Y, Z]
        tgt_alpha  = tgt_vis[:, -1:]

        # 1) Alpha / occupancy loss — shape fidelity
        loss_alpha = F.mse_loss(pred_alpha, tgt_alpha)

        # 2) Color loss — only where the target IS occupied
        tgt_mask = (tgt_alpha > cell_cfg.alive_threshold).float()
        if vis > 1:
            pred_color = pred_vis[:, :-1]                  # [B, V-1, X, Y, Z]
            tgt_color  = tgt_vis[:, :-1]
            color_diff = (pred_color - tgt_color) ** 2 * tgt_mask
            loss_color = color_diff.sum() / tgt_mask.sum().clamp(min=1.0)
        else:
            loss_color = torch.tensor(0.0, device=device)

        # 3) Overflow loss — penalise alive voxels outside the target
        overflow_mask = (1.0 - tgt_mask)                   # where target is dead
        overflow = (pred_alpha * overflow_mask) ** 2
        loss_overflow = overflow.mean()

        loss = (
            self._alpha_weight   * loss_alpha
          + self._color_weight   * loss_color
          + self._overflow_weight * loss_overflow
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        # write states back to pool
        with torch.no_grad():
            for j, idx in enumerate(indices):
                self._pool[idx] = state[j:j+1].detach()

        # Store the first sample for visualization
        self.state = state[0:1].detach()

        return loss.item()
    
    def _send_state(self):
        """Send full NCA state + training progress to the client."""
        if self._send_fn is None or self.state is None:
            return
        try:
            if isinstance(self.state, torch.Tensor):
                arr = self.state.detach().cpu().numpy()
            else:
                arr = self.state
            arr = arr.astype(np.float32)
            self._send_fn(build_state_msg(arr, self.current_epoch, self.latest_loss))
        except Exception as e:
            self._stop_event.set()
            print(f"Error sending state: {e}")
