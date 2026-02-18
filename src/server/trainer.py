import base64
import threading
import time
from typing import Any, Callable, Dict
import numpy as np
import torch
import torch.nn.functional as F
from src.core import Grid3D, CellConfig, PerceptionConfig, UpdateConfig, GridConfig
from .protocol import build_state_msg

SendFn = Callable[[Dict[str, Any]], None] # type alias for callback

class NCATrainer:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.target = None
        self.state = None

        self.current_epoch = 0
        self.total_epochs = 0
        self.latest_loss = 0.0

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

        # (D,H,W,C) channels-last
        target_chw = np.transpose(target, (3, 0, 1, 2)).astype(np.float32)
        self.target = torch.from_numpy(target_chw).unsqueeze(0).to(self._device)

        # Start from a single seed cell
        self.state = self.model.seed_center(1, self._device)

        self.current_epoch = 0
        self.total_epochs = config["training"]["num_epochs"]
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
            self._send_state()
            print(f"Epoch {epoch}/{self.total_epochs} - Loss: {loss:.4f}")
            time.sleep(0.1)  # throttle to avoid flooding the socket

        print("Training completed")

    def _step(self):
        """One training iteration"""
        device = self._device
        cell_cfg = self._cell_cfg
        n_steps = 8

        self.optimizer.zero_grad()

        state = self.model.seed_center(1, device)

        state = state + 0.02 * torch.randn_like(state)

        loss = torch.tensor(0.0, device=device)
        vis = cell_cfg.visible_channels
        for _ in range(n_steps):
            state = self.model(state, steps=1)
            pred = state[:, -vis:]
            tgt = self.target[:, :vis]
            loss = loss + F.mse_loss(pred, tgt)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Store for visualization / sending to client
        self.state = state.detach()

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


