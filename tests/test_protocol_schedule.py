from __future__ import annotations

import numpy as np
import pytest

from nca3d.core.runners import MorphRunner, NCARunner, TrainingSnapshot
from nca3d.core.schedule import Event, EventType
from nca3d.io.object_converter import obj_to_tensor


def _make_config(
    *,
    num_epochs: int = 1,
    learning_rate: float = 1e-3,
    batch_size: int = 1,
    visible_channels: int = 4,
) -> dict:
    return {
        "cell": {
            "hidden_channels": 4,
            "visible_channels": visible_channels,
            "alive_threshold": 0.1,
            "task_channels": 0,
        },
        "perception": {},
        "update": {"hidden_dim": 8},
        "grid": {"size": (4, 4, 4)},
        "training": {
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
        },
    }


def _make_target(visible_channels: int = 4) -> np.ndarray:
    target = np.zeros((4, 4, 4, visible_channels), dtype=np.float32)
    target[..., -1] = 1.0
    return target


# --- Init / basic shape tests ---

def test_runner_init_accepts_valid_config_and_target() -> None:
    runner = MorphRunner(verbose=False)
    runner.init(_make_config(), _make_target())

    assert runner.model is not None
    assert runner.target is not None
    assert runner.total_epochs == 1
    assert runner.target.shape == (1, 4, 4, 4, 4)
    assert len(runner._pool) == 32


def test_runner_train_yields_metrics_for_one_epoch() -> None:
    runner = MorphRunner(verbose=False)
    runner.init(_make_config(num_epochs=1, batch_size=1), _make_target())

    metrics = next(runner.train())

    assert set(metrics) >= {"loss_alpha", "loss_color", "loss_overflow", "loss_total"}
    assert runner.current_epoch == 1
    assert runner.latest_loss == metrics["loss_total"]


# --- Snapshot tests ---

def test_runner_snapshot_returns_training_snapshot() -> None:
    runner = MorphRunner(verbose=False)
    runner.init(_make_config(num_epochs=1, batch_size=1), _make_target())

    snap = runner.snapshot()

    assert isinstance(snap, TrainingSnapshot)
    assert snap.epoch == 0
    assert snap.total_epochs == 1
    assert snap.visible_channels == 4
    assert snap.state.shape == (1, 8, 4, 4, 4)


def test_runner_snapshot_raises_before_init() -> None:
    runner = MorphRunner(verbose=False)
    with pytest.raises(RuntimeError, match="init"):
        runner.snapshot()


# --- on_event tests ---

def test_runner_on_event_learning_rate() -> None:
    runner = MorphRunner(verbose=False)
    runner.init(_make_config(), _make_target())
    before = runner.optimizer.param_groups[0]["lr"]

    handled = runner.on_event(
        Event(epoch=1, event_type=EventType.LEARNING_RATE, value=before * 0.5)
    )

    assert handled is True
    assert runner.optimizer.param_groups[0]["lr"] == before * 0.5


def test_runner_on_event_unknown_type_returns_false() -> None:
    from unittest.mock import MagicMock

    runner = MorphRunner(verbose=False)
    runner.init(_make_config(), _make_target())

    fake_event = MagicMock()
    fake_event.event_type = "NONEXISTENT"
    fake_event.value = 0.0

    assert runner.on_event(fake_event) is False


def test_runner_on_event_before_init_returns_false() -> None:
    runner = MorphRunner(verbose=False)
    assert runner.on_event(
        Event(epoch=1, event_type=EventType.LEARNING_RATE, value=0.001)
    ) is False


# --- Strategy interface compliance ---

def test_training_runner_is_abstract() -> None:
    with pytest.raises(TypeError):
        NCARunner()  # type: ignore[abstract]


def test_morph_runner_is_nca_runner() -> None:
    assert issubclass(MorphRunner, NCARunner)


# --- Target validation ---

def test_runner_init_rejects_invalid_target_shape() -> None:
    runner = MorphRunner(verbose=False)
    with pytest.raises(ValueError, match=r"shape \(D, H, W, C\)"):
        runner.init(_make_config(), np.zeros((4, 4, 4), dtype=np.float32))


def test_obj_to_tensor_rejects_invalid_grid_size() -> None:
    with pytest.raises(
        ValueError, match="grid_size must be a tuple of 3 positive integers"
    ):
        obj_to_tensor("ignored.obj", grid_size=(0, 4, 4))