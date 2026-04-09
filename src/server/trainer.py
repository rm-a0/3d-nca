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
        send_delay_ms: int,
        send_fn: SendFn,
    ) -> None:
        """Stop any active session and start an inference run.

        Args:
            model_bytes: Serialised PyTorch checkpoint (.pt file bytes).
            phase_steps: Forward steps per task phase.
            broadcast_every: Broadcast state every N steps.
            send_delay_ms: Delay before each state send in milliseconds.
            send_fn: Callback used to send protocol messages to the client.
        """
        self.stop()
        self._send_fn = send_fn
        self._start_thread(
            self._inference_loop,
            model_bytes,
            phase_steps,
            broadcast_every,
            send_delay_ms,
        )

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
        self,
        model_bytes: bytes,
        phase_steps: int,
        broadcast_every: int,
        send_delay_ms: int,
    ) -> None:
        """Load a checkpoint from bytes and run forward inference steps.

        Args:
            model_bytes: Serialised PyTorch checkpoint (.pt file bytes).
            phase_steps: Number of forward steps to run per phase.
            broadcast_every: Broadcast state every N steps.
            send_delay_ms: Delay before each broadcast in milliseconds.
        """
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            ckpt = torch.load(io.BytesIO(model_bytes), map_location=device, weights_only=False)
        except Exception as exc:
            self._send_error(f"Failed to load model: {exc}")
            return

        from src.core.grid import Grid3D, GridConfig

        try:
            cell_cfg, perc_cfg, upd_cfg, grid_cfg, state_dict = (
                self._parse_inference_checkpoint(ckpt)
            )
        except Exception as exc:
            self._send_error(f"Invalid model config: {exc}")
            return

        model = Grid3D(cell_cfg, perc_cfg, upd_cfg, grid_cfg).to(device)
        try:
            model.load_state_dict(state_dict)
        except Exception as exc:
            self._send_error(f"State dict mismatch: {exc}")
            return

        model.eval()
        phase_steps = max(int(phase_steps), 1)
        broadcast_every = max(int(broadcast_every), 1)
        send_delay_s = max(float(send_delay_ms), 0.0) / 1000.0
        step = 0

        with torch.no_grad():
            if cell_cfg.task_channels > 0:
                n_tasks = int(cell_cfg.task_channels)
                task_ids = torch.tensor([0], device=device)
                state = model.seed_center(1, device, task_ids)

                for task_id in range(n_tasks):
                    self._set_task_channels(state, task_id, cell_cfg)

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
                            if send_delay_s > 0.0:
                                time.sleep(send_delay_s)
                            self._broadcast(state, step, 0.0, cell_cfg.visible_channels)

                    if send_delay_s > 0.0:
                        time.sleep(send_delay_s)
                    self._broadcast(state, step, 0.0, cell_cfg.visible_channels)
            else:
                state = model.seed_center(1, device, None)
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
                        if send_delay_s > 0.0:
                            time.sleep(send_delay_s)
                        self._broadcast(state, step, 0.0, cell_cfg.visible_channels)

            if send_delay_s > 0.0:
                time.sleep(send_delay_s)
            self._broadcast(state, step, 0.0, cell_cfg.visible_channels)

        if self.verbose:
            print(f"[Trainer] Inference complete ({step} steps)")

    # --- Private: helpers ---

    def _parse_inference_checkpoint(self, ckpt: Dict[str, Any]) -> tuple:
        """Parse both legacy and NCAModel checkpoint formats for inference."""
        from src.core.cell import CellConfig
        from src.core.perception import PerceptionConfig
        from src.core.update import UpdateConfig
        from src.core.grid import GridConfig

        config = ckpt.get("config")
        if not isinstance(config, dict):
            raise ValueError("checkpoint missing dict config")

        if "cell" in config and "grid" in config:
            # Legacy run checkpoint format emitted by NCALogger.
            cell_cfg = CellConfig(**config["cell"])
            perc_cfg = PerceptionConfig(**config.get("perception", {}))
            upd_cfg = UpdateConfig(**config.get("update", {}))
            grid_cfg = GridConfig(size=tuple(int(v) for v in config["grid"]["size"]))
        elif "grid_size" in config:
            # NCAModel.save() format (flat NCAConfig fields).
            cell_cfg = CellConfig(
                hidden_channels=int(config["hidden_channels"]),
                visible_channels=int(config["visible_channels"]),
                alive_threshold=float(config.get("alive_threshold", 0.1)),
                task_channels=int(config.get("task_channels", 0)),
            )
            perc_cfg = PerceptionConfig(
                kernel_radius=int(config.get("perception_kernel_radius", 1)),
                channel_groups=int(config.get("perception_channel_groups", 3)),
            )
            upd_cfg = UpdateConfig(
                hidden_dim=int(config.get("update_hidden_dim", 128)),
                stochastic_update=bool(config.get("update_stochastic", False)),
                fire_rate=float(config.get("update_fire_rate", 0.5)),
            )
            grid_cfg = GridConfig(size=tuple(int(v) for v in config["grid_size"]))
        else:
            raise ValueError(
                "unsupported config schema (expected nested keys 'cell'/'grid' or flat 'grid_size')"
            )

        state_dict = ckpt.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("checkpoint missing state_dict")

        state_dict = self._normalize_grid_state_dict(state_dict)
        return cell_cfg, perc_cfg, upd_cfg, grid_cfg, state_dict

    @staticmethod
    def _normalize_grid_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert NCAModel-style keys (grid.*) into Grid3D keys when needed."""
        has_grid_prefix = any(k.startswith("grid.") for k in state_dict)
        if not has_grid_prefix:
            return state_dict
        return {
            (k[5:] if k.startswith("grid.") else k): v for k, v in state_dict.items()
        }

    @staticmethod
    def _set_task_channels(state: Any, task_id: int, cell_cfg: Any) -> None:
        """Set one-hot task conditioning channels on an existing state tensor."""
        tc = int(cell_cfg.task_channels)
        if tc <= 0:
            return
        vis = int(cell_cfg.visible_channels)
        task_slice = state[:, -(vis + tc) : -vis, ...]
        task_slice.zero_()
        task_slice[:, int(task_id), ...] = 1.0

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