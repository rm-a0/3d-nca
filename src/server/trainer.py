"""
NCATrainer - Orchestrates an NCARunner backend in a background thread.

Wraps any NCARunner implementation in a background thread and handles
logging and state broadcasting.  Swap the backend by passing runner_factory::

    trainer = NCATrainer(runner_factory=MyRunner)
    server  = NCAServer(trainer=trainer)
"""

from __future__ import annotations

import io
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.core.runners import NCARunner, MorphRunner
from src.core.schedule import Event, Schedule
from .protocol import build_state_msg
from .logger import NCALogger

SendFn = Callable[[Dict[str, Any]], None]

_BROADCAST_INTERVAL = 0.02  # seconds - cap broadcasts at 50 fps


class NCATrainer:
    """Orchestrates an NCARunner backend in a background thread.

    Handles threading, logging, and state broadcasting for any NCARunner
    implementation.  The runner_factory parameter is the extension point for
    custom training backends.

    Args:
        base_dir: Root directory for run logs and checkpoints.
        checkpoint_interval: Save a model checkpoint every N epochs.
        verbose: Print progress and event messages.
        runner_factory: Callable that returns a new NCARunner instance.
            Defaults to MorphRunner.
    """

    def __init__(
        self,
        base_dir: str = "runs",
        checkpoint_interval: int = 500,
        verbose: bool = True,
        runner_factory: Optional[Callable[[], NCARunner]] = None,
    ) -> None:
        self._base_dir = base_dir
        self._checkpoint_interval = checkpoint_interval
        self.verbose = verbose
        self._runner_factory: Callable[[], NCARunner] = (
            runner_factory or (lambda: MorphRunner(verbose=verbose))
        )

        self._runner: Optional[NCARunner] = None
        self._schedule = Schedule()
        self.logger: Optional[NCALogger] = None

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._last_broadcast: float = 0.0
        self._send_fn: Optional[SendFn] = None

    # --- Public API ---

    def init(self, config: dict, target: np.ndarray, send_fn: SendFn) -> None:
        """Stop any active session and start a new training run.

        Args:
            config: Training configuration dict.
            target: Target voxel grid in (D, H, W, C) external format.
            send_fn: Callback used to send protocol messages to the client.
        """
        self.stop()
        self._send_fn = send_fn

        self.logger = NCALogger(
            base_dir=self._base_dir,
            checkpoint_interval=self._checkpoint_interval,
        )
        self.logger._send_fn = send_fn
        self.logger.log_meta(config)

        self._runner = self._runner_factory()
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
        """Stop any active session and start an inference run.

        Args:
            model_bytes: Serialised PyTorch checkpoint (.pt file bytes).
            phase_steps: Forward steps per task phase.
            broadcast_every: Broadcast state every N steps.
            send_fn: Callback used to send protocol messages to the client.
        """
        self.stop()
        self._send_fn = send_fn
        self._start_thread(self._inference_loop, model_bytes, phase_steps, broadcast_every)

    def pause(self) -> None:
        """Pause the active training session."""
        self._pause_event.clear()
        if self.verbose:
            print("[Trainer] Paused")

    def resume(self) -> None:
        """Resume a paused training session."""
        self._pause_event.set()
        if self.verbose:
            print("[Trainer] Resumed")

    def stop(self) -> None:
        """Stop the active session and wait for the background thread to exit."""
        self._stop_event.set()
        self._pause_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10.0)
        self._thread = None
        self._runner = None
        self._stop_event.clear()

    def update_schedule(self, events_data: List[dict]) -> None:
        """Replace the active schedule with a new event list.

        Args:
            events_data: List of event dicts from the wire protocol.
        """
        events = [Event.from_dict(d) for d in events_data]
        self._schedule.replace(events)
        if self.verbose:
            print(f"[Trainer] Schedule updated: {len(events)} event(s)")

    @property
    def is_running(self) -> bool:
        """True while the background thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_paused(self) -> bool:
        """True while the session is intentionally paused."""
        return not self._pause_event.is_set()

    # --- Private: thread management ---

    def _start_thread(self, target: Callable, *args: Any) -> None:
        self._stop_event.clear()
        self._pause_event.set()
        self._last_broadcast = 0.0
        self._thread = threading.Thread(target=target, args=args, daemon=True)
        self._thread.start()

    # --- Private: training loop ---

    def _training_loop(self) -> None:
        runner = self._runner
        if runner is None:
            return

        gen = runner.train(schedule=self._schedule)
        try:
            while not self._stop_event.is_set():
                self._pause_event.wait()
                if self._stop_event.is_set():
                    break

                try:
                    next(gen)
                except StopIteration:
                    break

                snap = runner.snapshot()

                if self.logger is not None:
                    m = snap.metrics
                    self.logger.log_epoch(
                        epoch=snap.epoch,
                        loss_alpha=float(m.get("loss_alpha", snap.loss)),
                        loss_color=float(m.get("loss_color", 0.0)),
                        loss_overflow=float(m.get("loss_overflow", 0.0)),
                        loss_total=float(m.get("loss_total", snap.loss)),
                        model=getattr(runner, "model", None),
                        is_final=snap.epoch == snap.total_epochs,
                    )

                self._broadcast(snap.state, snap.epoch, snap.loss, snap.visible_channels)
        finally:
            gen.close()

        if self.verbose:
            run_id = self.logger.run_id if self.logger else "?"
            print(f"[Trainer] Run {run_id} finished")

    # --- Private: inference loop ---

    def _inference_loop(
        self, model_bytes: bytes, phase_steps: int, broadcast_every: int
    ) -> None:
        """Load a checkpoint from bytes and run forward inference steps.

        Args:
            model_bytes: Serialised PyTorch checkpoint (.pt file bytes).
            phase_steps: Total number of forward steps to run.
            broadcast_every: Broadcast state every N steps.
        """
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            ckpt = torch.load(io.BytesIO(model_bytes), map_location=device, weights_only=False)
        except Exception as exc:
            self._send_error(f"Failed to load model: {exc}")
            return

        from src.core.cell import CellConfig
        from src.core.grid import Grid3D, GridConfig
        from src.core.perception import PerceptionConfig
        from src.core.update import UpdateConfig

        config = ckpt.get("config", {})
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
        total_steps = max(int(phase_steps), 1)
        broadcast_every = max(int(broadcast_every), 1)
        step = 0

        with torch.no_grad():
            state = model.seed_center(1, device, None)

            for _ in range(total_steps):
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

            self._broadcast(state, step, 0.0, cell_cfg.visible_channels)

        if self.verbose:
            print(f"[Trainer] Inference complete ({step} steps)")

    # --- Private: helpers ---

    def _broadcast(
        self,
        state: Any,
        epoch: int,
        loss: float,
        visible_channels: int,
    ) -> None:
        """Send a state snapshot to the client at a rate-limited frequency.

        Drops frames if called faster than _BROADCAST_INTERVAL.

        Args:
            state: State tensor [B, C, D, H, W] or numpy array.
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
            if hasattr(state, "detach"):
                arr = state.detach().cpu().numpy().astype(np.float32)
            else:
                arr = np.asarray(state, dtype=np.float32)
            arr = arr[:, -visible_channels:] if arr.ndim == 5 else arr[-visible_channels:]
            self._send_fn(build_state_msg(arr, epoch, loss))
        except Exception as exc:
            if self.verbose:
                print(f"[Trainer] Broadcast error: {exc}")

    def _send_error(self, message: str) -> None:
        """Send an error message to the client.

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