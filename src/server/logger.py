from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

class NCALogger:
    """Owns all file I/O for a single training run."""

    def __init__(
        self,
        run_id: str = "001",
        base_dir: str = "runs",
        checkpoint_interval: int = 500,
        send_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.run_id = run_id
        self.checkpoint_interval = checkpoint_interval
        self._send_fn = send_fn

        self.phase: str = "1"

        self.run_dir = Path(base_dir) / f"run_{run_id}"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self._loss_path = self.run_dir / "loss.csv"
        self._events_path = self.run_dir / "events.jsonl"
        self._ensure_loss_header()

    def log_meta(self, config: dict) -> None:
        """Persist the full training config and wall-clock start time."""
        meta = {
            "run_id": self.run_id,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": config,
        }
        (self.run_dir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    def log_epoch(
        self,
        epoch: int,
        loss_alpha: float,
        loss_color: float,
        loss_overflow: float,
        loss_total: float,
        best_pool_state: Optional[np.ndarray] = None,
        is_final: bool = False,
    ) -> None:
        """Append one CSV row and conditionally save a checkpoint."""
        with self._loss_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                self.phase,
                round(loss_alpha,    6),
                round(loss_color,    6),
                round(loss_overflow, 6),
                round(loss_total,    6),
            ])

        should_ckpt = (
            best_pool_state is not None
            and self.checkpoint_interval > 0
            and (epoch == 0 or is_final or epoch % self.checkpoint_interval == 0)
        )
        if should_ckpt:
            self._save_checkpoint(epoch, best_pool_state)

        if self._send_fn is not None:
            try:
                self._send_fn({
                    "type": "log",
                    "epoch": epoch,
                    "phase": self.phase,
                    "loss": round(loss_total, 6),
                })
            except Exception:
                pass  # never let logging kill the training loop

    def log_event(
        self,
        epoch: int,
        event_type: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one JSON line to events.jsonl."""
        record: Dict[str, Any] = {
            "epoch": epoch,
            "event_type": event_type,
            "phase": self.phase,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if details:
            record.update(details)
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _ensure_loss_header(self) -> None:
        if not self._loss_path.exists():
            with self._loss_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "phase",
                    "loss_alpha", "loss_color", "loss_overflow", "loss_total",
                ])

    def _save_checkpoint(self, epoch: int, state: np.ndarray) -> None:
        """Save a (D, H, W, channels) array."""
        path = self.checkpoint_dir / f"ep{epoch:04d}.npy"
        np.save(path, state)
