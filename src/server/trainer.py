"""
NCATrainer - orchestrates training and inference in a background thread.

Two modes:
  - Training:  runner.train() generator, streams state every broadcast cycle.
  - Inference: load saved model from bytes, run forward phases, stream state.
"""

from __future__ import annotations

import io
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.core.runner import NCARunner
from src.core.schedule import Event, Schedule
from .protocol import build_state_msg
from .logger import NCALogger

SendFn = Callable[[Dict[str, Any]], None]

# Minimum seconds between state broadcasts to avoid flooding the socket.
_BROADCAST_INTERVAL = 0.05  # 20 fps max


def _switch_task_channel(state, task_id: int, cell_cfg) -> Any:
    """Replace task-channel slice with a fresh one-hot via torch.cat.

    Uses cat instead of in-place assignment so autograd graphs in the
    hidden and visible channels are preserved (needed for sequential morph
    training where gradients must flow through task switches).
    """
    import torch

    tc = cell_cfg.task_channels
    vc = cell_cfg.visible_channels
    B, C, D, H, W = state.shape
    hidden_end = C - vc - tc

    new_task = torch.zeros(B, tc, D, H, W, device=state.device, dtype=state.dtype)
    new_task[:, task_id, ...] = 1.0
    return torch.cat([state[:, :hidden_end], new_task, state[:, -vc:]], dim=1)


class NCATrainer:
    """Manages NCA training and inference in a background thread.

    Supports two modes via separate thread loops:
    - Training: runner.train() generator yields metrics each epoch
    - Inference: load pretrained model, run forward phases

    Provides public control API: init, run_inference, pause, resume, stop, update_schedule.
    All state access and thread coordination uses lock-free event flags or
    thread-safe Schedule class. Broadcasts state snapshots to client at configurable intervals.
    """

    def __init__(
        self,
        base_dir: str = "runs",
        checkpoint_interval: int = 500,
        verbose: bool = True,
    ) -> None:
        self._base_dir = base_dir
        self._checkpoint_interval = checkpoint_interval
        self.verbose = verbose

        self._runner: Optional[NCARunner] = None
        self._schedule = Schedule()
        self.logger: Optional[NCALogger] = None

        self._train_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()

        self._send_fn: Optional[SendFn] = None
        self._last_broadcast: float = 0.0

    # --- Public API ---

    def init(self, config: dict, target: np.ndarray, send_fn: SendFn) -> None:
        """Stop any active session, then start a new training run."""
        self.stop()
        self._send_fn = send_fn

        self.logger = NCALogger(
            base_dir=self._base_dir,
            checkpoint_interval=self._checkpoint_interval,
        )
        self.logger._send_fn = send_fn
        self.logger.log_meta(config)

        self._runner = NCARunner(verbose=self.verbose)
        self._runner.init(config, target)
        self._schedule = Schedule()

        self._start_thread(self._training_loop)

    def run_inference(
        self,
        model_bytes: bytes,
        phase_steps: int,
        broadcast_every: int,
        send_fn: SendFn,
    ) -> None:
        """Stop any active session, load model from bytes, run forward inference."""
        self.stop()
        self._send_fn = send_fn
        self._start_thread(
            self._inference_loop, model_bytes, phase_steps, broadcast_every
        )

    def pause(self) -> None:
        self._pause_event.clear()
        if self.verbose:
            print("[Trainer] Paused")

    def resume(self) -> None:
        self._pause_event.set()
        if self.verbose:
            print("[Trainer] Resumed")

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.set()  # unblock any waiting thread
        if self._train_thread is not None and self._train_thread.is_alive():
            self._train_thread.join(timeout=10.0)
        self._train_thread = None
        self._stop_event.clear()

    def update_schedule(self, events_data: List[dict]) -> None:
        events = [Event.from_dict(d) for d in events_data]
        self._schedule.replace(events)
        if self.verbose:
            print(f"[Trainer] Schedule updated: {len(events)} event(s)")

    @property
    def is_running(self) -> bool:
        return self._train_thread is not None and self._train_thread.is_alive()

    @property
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    # --- Private Loops ---

    def _start_thread(self, target, *args) -> None:
        self._stop_event.clear()
        self._pause_event.set()
        self._last_broadcast = 0.0
        self._train_thread = threading.Thread(target=target, args=args, daemon=True)
        self._train_thread.start()

    def _training_loop(self) -> None:
        """Run training loop in background thread.

        Pulls metrics from runner.train() generator, logs to disk, broadcasts state,
        and checks schedule for parameter updates each epoch. Respects pause/stop events.
        """
        runner = self._runner
        gen = runner.train(schedule=self._schedule)

        while not self._stop_event.is_set():
            self._pause_event.wait()
            if self._stop_event.is_set():
                break
            try:
                metrics = next(gen)
            except StopIteration:
                break

            if self.logger is not None:
                is_final = runner.current_epoch == runner.total_epochs
                self.logger.log_epoch(
                    epoch=runner.current_epoch,
                    loss_alpha=metrics["loss_alpha"],
                    loss_color=metrics["loss_color"],
                    loss_overflow=metrics["loss_overflow"],
                    loss_total=metrics["loss_total"],
                    phase=str(metrics.get("phase", "")),
                    model=runner.model,
                    is_final=is_final,
                )

            self._broadcast(
                runner.state,
                runner.current_epoch,
                runner.latest_loss,
                runner._cell_cfg.visible_channels,
            )

        if self.verbose:
            run_id = self.logger.run_id if self.logger else "?"
            print(f"[Trainer] Run {run_id} finished")

    def _inference_loop(
        self, model_bytes: bytes, phase_steps: int, broadcast_every: int
    ) -> None:
        """Run inference loop in background thread.

        Load model checkpoint from bytes, initialize state, and run forward passes
        (one phase per task channel). Broadcasts state every N steps.

        Args:
            model_bytes: Serialized PyTorch checkpoint (.pt file binary).
            phase_steps: How many forward steps per task phase.
            broadcast_every: Broadcast state every N steps (rate limiting).
        """
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Load Checkpoint ---
        try:
            ckpt = torch.load(
                io.BytesIO(model_bytes), map_location=device, weights_only=False
            )
        except Exception as exc:
            self._send_error(f"Failed to load model: {exc}")
            return

        config = ckpt.get("config", {})

        from src.core.cell import CellConfig
        from src.core.grid import Grid3D, GridConfig
        from src.core.perception import PerceptionConfig
        from src.core.update import UpdateConfig

        try:
            cell_cfg = CellConfig(**config["cell"])
            perc_cfg = PerceptionConfig(**config.get("perception", {}))
            upd_cfg = UpdateConfig(**config.get("update", {}))
            grid_cfg = GridConfig(size=tuple(config["grid"]["size"]))
        except Exception as exc:
            self._send_error(f"Invalid model config: {exc}")
            return

        model = Grid3D(cell_cfg, perc_cfg, upd_cfg, grid_cfg).to(device)
        try:
            model.load_state_dict(ckpt["state_dict"])
        except Exception as exc:
            self._send_error(f"State dict mismatch: {exc}")
            return

        model.eval()
        n_phases = max(cell_cfg.task_channels, 1)
        step = 0

        # --- Forward Pass ---
        with torch.no_grad():
            task0 = (
                torch.tensor([0], device=device) if cell_cfg.task_channels > 0 else None
            )
            state = model.seed_center(1, device, task0)

            for phase_id in range(n_phases):
                if phase_id > 0 and cell_cfg.task_channels > 0:
                    state = _switch_task_channel(state, phase_id, cell_cfg)

                for _ in range(phase_steps):
                    if self._stop_event.is_set():
                        return
                    self._pause_event.wait()
                    if self._stop_event.is_set():
                        return

                    state = model(state, steps=1, use_checkpointing=False)
                    state = torch.clamp(state, -1.0, 1.0)
                    step += 1

                    if step % broadcast_every == 0:
                        self._broadcast(state, step, 0.0, cell_cfg.visible_channels)

            # always send final state
            self._broadcast(state, step, 0.0, cell_cfg.visible_channels)

        if self.verbose:
            print(f"[Trainer] Inference complete ({step} steps)")

    # --- Helpers ---

    def _broadcast(
        self,
        state,
        epoch: int,
        loss: float,
        visible_channels: int,
    ) -> None:
        """Send state snapshot to client at rate-limited intervals.

        Drops redundant frames if called faster than _BROADCAST_INTERVAL to avoid
        flooding the socket. Extracts only visible channels from state [B,C,X,Y,Z].

        Args:
            state: Current state tensor [B, C, X, Y, Z] or None.
            epoch: Current training epoch or inference step count.
            loss: Scalar loss value for display.
            visible_channels: Number of channels to extract and send.
        """
        if self._send_fn is None or state is None:
            return
        now = time.monotonic()
        if now - self._last_broadcast < _BROADCAST_INTERVAL:
            return
        self._last_broadcast = now
        try:
            arr = state.detach().cpu().numpy().astype(np.float32)
            # extract only visible channels
            arr = (
                arr[:, -visible_channels:] if arr.ndim == 5 else arr[-visible_channels:]
            )
            self._send_fn(build_state_msg(arr, epoch, loss))
        except Exception as exc:
            if self.verbose:
                print(f"[Trainer] Broadcast error: {exc}")

    def _send_error(self, message: str) -> None:
        """Send error message to client and log to console if verbose.

        Args:
            message: Error description.
        """
        if self._send_fn is not None:
            try:
                self._send_fn({"type": "error", "message": message})
            except Exception:
                pass
        if self.verbose:
            print(f"[Trainer] Error: {message}")
