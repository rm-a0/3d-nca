"""
NCALogger - owns all file I/O for a single training run.

Each instantiation claims the next available run_NNN directory automatically,
so concurrent or sequential runs never collide.

Run directory layout:
  runs/
    run_003/
            meta.json          - config, timestamps, git hash
            loss.csv           - per-epoch losses
            events.jsonl       - schedule events (one JSON per line)
      checkpoints/
                model_ep01000.pt - full PyTorch checkpoint (config + state_dict)
        model_ep02000.pt
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

_BROADCAST_MIN_INTERVAL_S = 0.05  # cap log broadcasts at 20 Hz


class NCALogger:
    """Persists training metadata, losses, events, and checkpoints for one run."""

    def __init__(
        self,
        base_dir: str = "runs",
        checkpoint_interval: int = 500,
        send_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Create logger and allocate a new run directory.

        Args:
            base_dir: Root directory containing run_NNN folders.
            checkpoint_interval: Save checkpoint every N epochs.
            send_fn: Optional callback for lightweight log broadcasts.
        """
        self.run_id = self._next_run_id(base_dir)
        self.checkpoint_interval = checkpoint_interval
        self._send_fn = send_fn
        self._config: Optional[dict] = None

        self.run_dir = Path(base_dir) / f"run_{self.run_id}"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self._loss_path = self.run_dir / "loss.csv"
        self._events_path = self.run_dir / "events.jsonl"
        self._ensure_loss_header()

    # --- Public API ---

    def log_meta(self, config: dict) -> None:
        """Write meta.json with config, wall-clock start time, and git hash."""
        self._config = config
        meta: Dict[str, Any] = {
            "run_id": self.run_id,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "config": config,
        }
        try:
            import subprocess

            meta["git_hash"] = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except Exception:
            pass

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
        model=None,
        is_final: bool = False,
    ) -> None:
        """Append one CSV row.  Optionally save a model checkpoint."""
        with self._loss_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    round(loss_alpha, 6),
                    round(loss_color, 6),
                    round(loss_overflow, 6),
                    round(loss_total, 6),
                ]
            )

        should_ckpt = (
            model is not None
            and self.checkpoint_interval > 0
            and (is_final or epoch % self.checkpoint_interval == 0)
        )
        if should_ckpt:
            self.save_model(model, epoch)

        if self._send_fn is not None:
            try:
                self._send_fn(
                    {
                        "type": "log",
                        "epoch": epoch,
                        "loss": round(loss_total, 6),
                    }
                )
            except Exception:
                pass

    def save_model(self, model, epoch: int) -> Path:
        """Save a full model checkpoint as .pt (config + state_dict)."""
        import torch

        path = self.checkpoint_dir / f"model_ep{epoch:05d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "config": self._config,
                "state_dict": model.state_dict(),
            },
            path,
        )
        return path

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
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        if details:
            record.update(details)
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # --- Private Helpers ---

    @staticmethod
    def _next_run_id(base_dir: str) -> str:
        """Return the next zero-padded run ID by scanning existing run_NNN dirs."""
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)
        existing = [
            int(d.name[4:])
            for d in base.iterdir()
            if d.is_dir() and d.name.startswith("run_") and d.name[4:].isdigit()
        ]
        return f"{max(existing, default=-1) + 1:03d}"

    def _ensure_loss_header(self) -> None:
        """Create loss.csv with header row if the file does not exist."""
        if not self._loss_path.exists():
            with self._loss_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["epoch", "loss_alpha", "loss_color", "loss_overflow", "loss_total"]
                )
