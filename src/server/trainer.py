import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.core.runner import NCARunner
from src.core.schedule import Event, Schedule
from .protocol import build_state_msg
from .logger import NCALogger

SendFn = Callable[[Dict[str, Any]], None]


class NCATrainer:
    def __init__(self, run_id: str = "000", checkpoint_interval: int = 500, verbose: bool = True):
        self._runner: Optional[NCARunner] = None
        self._schedule = Schedule()

        self._train_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()

        self._send_fn: Optional[SendFn] = None
        self.verbose = verbose

        self.logger = NCALogger(run_id=run_id, checkpoint_interval=checkpoint_interval)

    def init(self, config: dict, target: np.ndarray, send_fn: SendFn) -> None:
        self._runner = NCARunner(verbose=self.verbose)
        self._runner.init(config, target)
        self._schedule = Schedule()
        self._send_fn = send_fn
        self.logger._send_fn = send_fn
        self.logger.log_meta(config)

        self._stop_event.clear()
        self._pause_event.set()
        self._train_thread = threading.Thread(target=self._training_loop)
        self._train_thread.start()

    def pause(self) -> None:
        self._pause_event.clear()
        if self.verbose:
            print("Training paused")

    def resume(self) -> None:
        self._pause_event.set()
        if self.verbose:
            print("Training resumed")

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.set()
        if self._train_thread is not None:
            self._train_thread.join()
        if self.verbose:
            print("Training stopped")

    def update_schedule(self, events_data: List[dict]) -> None:
        events = [Event.from_dict(d) for d in events_data]
        self._schedule.replace(events)
        if self.verbose:
            print(f"Schedule updated: {len(self._schedule.events)} pending event(s)")

    @property
    def is_running(self) -> bool:
        return self._train_thread is not None and self._train_thread.is_alive()

    @property
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    def get_current_state(self):
        return self._runner.state if self._runner else None

    def _training_loop(self) -> None:
        runner = self._runner
        gen = runner.train(schedule=self._schedule)
        while True:
            self._pause_event.wait()
            if self._stop_event.is_set():
                break
            try:
                metrics = next(gen)
            except StopIteration:
                break
            is_final = runner.current_epoch == runner.total_epochs
            self.logger.log_epoch(
                epoch=runner.current_epoch,
                loss_alpha=metrics["loss_alpha"],
                loss_color=metrics["loss_color"],
                loss_overflow=metrics["loss_overflow"],
                loss_total=metrics["loss_total"],
                best_pool_state=metrics["best_np"],
                is_final=is_final,
            )
            self._send_state()
            time.sleep(0.1)
        if self.verbose:
            print("Training completed")

    def _send_state(self) -> None:
        if self._send_fn is None or self._runner is None or self._runner.state is None:
            return
        try:
            arr = self._runner.state.detach().cpu().numpy().astype(np.float32)
            vis = int(self._runner._cell_cfg.visible_channels)
            arr = arr[:, -vis:, ...] if arr.ndim == 5 else arr[-vis:, ...]
            self._send_fn(build_state_msg(arr, self._runner.current_epoch, self._runner.latest_loss))
        except Exception as e:
            self._stop_event.set()
            if self.verbose:
                print(f"Error sending state: {e}")