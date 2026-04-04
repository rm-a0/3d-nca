from __future__ import annotations

import numpy as np
import pytest

from src.core.runner import NCARunner
from src.core.schedule import Event, EventType
from src.io.object_converter import obj_to_tensor


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


def test_runner_init_accepts_valid_config_and_target() -> None:
    runner = NCARunner(verbose=False)

    runner.init(_make_config(), _make_target())

    assert runner.model is not None
    assert runner.target is not None
    assert runner.total_epochs == 1
    assert runner.target.shape == (1, 4, 4, 4, 4)
    assert len(runner._pool) == 32


def test_runner_train_yields_metrics_for_one_epoch() -> None:
    runner = NCARunner(verbose=False)
    runner.init(_make_config(num_epochs=1, batch_size=1), _make_target())

    metrics = next(runner.train())

    assert set(metrics) >= {"loss_alpha", "loss_color", "loss_overflow", "loss_total"}
    assert runner.current_epoch == 1
    assert runner.latest_loss == metrics["loss_total"]


def test_runner_snapshot_and_schedule_event_boundary() -> None:
    runner = NCARunner(verbose=False)
    runner.init(_make_config(num_epochs=1, batch_size=1), _make_target())

    snapshot = runner.snapshot()
    assert snapshot.epoch == 0
    assert snapshot.total_epochs == 1
    assert snapshot.visible_channels == 4
    assert snapshot.state.shape == (1, 8, 4, 4, 4)

    before = runner.optimizer.param_groups[0]["lr"]
    handled = runner.apply_schedule_event(Event(epoch=1, event_type=EventType.LEARNING_RATE, value=before * 0.5))

    assert handled is True
    assert runner.optimizer.param_groups[0]["lr"] == before * 0.5


def test_runner_lifecycle_defaults_exposed_by_base_runtime() -> None:
    runner = NCARunner(verbose=False)

    assert runner.is_running is False
    assert runner.is_paused is False

    runner.pause()
    assert runner.is_paused is True

    runner.resume()
    assert runner.is_paused is False

    runner.stop()
    assert runner.stop_requested is True


def test_runner_init_rejects_missing_config_section() -> None:
    runner = NCARunner(verbose=False)
    config = _make_config()
    config.pop("training")

    with pytest.raises(ValueError, match="training"):
        runner.init(config, _make_target())


def test_runner_init_rejects_invalid_target_shape() -> None:
    runner = NCARunner(verbose=False)

    with pytest.raises(ValueError, match="shape \(D, H, W, C\)"):
        runner.init(_make_config(), np.zeros((4, 4, 4), dtype=np.float32))


def test_obj_to_tensor_rejects_invalid_grid_size() -> None:
    with pytest.raises(
        ValueError, match="grid_size must be a tuple of 3 positive integers"
    ):
        obj_to_tensor("ignored.obj", grid_size=(0, 4, 4))
