from __future__ import annotations

import dataclasses

import pytest
import torch

from src.core.cell import CellConfig, CellState
from src.core.grid import Grid3D, GridConfig
from src.core.nca_model import NCAConfig, NCAModel
from src.core.perception import Perception3D, PerceptionConfig
from src.core.update import UpdateConfig, UpdateRule


def _make_grid(
    hidden_channels: int = 2,
    visible_channels: int = 4,
    task_channels: int = 0,
    grid_size: tuple[int, int, int] = (4, 5, 6),
) -> Grid3D:
    cell_cfg = CellConfig(
        hidden_channels=hidden_channels,
        visible_channels=visible_channels,
        task_channels=task_channels,
    )
    perc_cfg = PerceptionConfig()
    upd_cfg = UpdateConfig(hidden_dim=8)
    grid_cfg = GridConfig(size=grid_size)
    return Grid3D(cell_cfg, perc_cfg, upd_cfg, grid_cfg)


def test_cell_alive_mask_uses_local_neighborhood() -> None:
    cfg = CellConfig(hidden_channels=1, visible_channels=2, task_channels=1)
    assert cfg.total_channels == 4

    cell = CellState(cfg)
    state = torch.zeros(1, cfg.total_channels, 3, 3, 3)
    state[0, -1, 1, 1, 1] = 0.2

    alive = cell.update_alive_mask(state)

    assert alive.shape == (1, 1, 3, 3, 3)
    assert alive[0, 0, 1, 1, 1]
    assert alive[0, 0, 0, 0, 0]


def test_cell_alive_mask_is_strictly_greater_than_threshold() -> None:
    cfg = CellConfig(alive_threshold=0.1)
    cell = CellState(cfg)
    state = torch.zeros(1, cfg.total_channels, 3, 3, 3)
    state[0, -1, 1, 1, 1] = 0.1

    alive = cell.update_alive_mask(state)

    assert not alive.any()


def test_perception_kernels_and_forward_shape() -> None:
    perception = Perception3D(PerceptionConfig(), in_channels=1)
    weights = perception.depthwise.weight.detach()

    assert weights.shape == (3, 1, 3, 3, 3)
    assert weights[0, 0, 1, 1, 1].item() == 1.0
    assert weights[1].sum().item() == 6.0
    assert weights[2, 0, 1, 1, 1].item() == 6.0
    assert weights[2].sum().item() == 0.0

    state = torch.ones(1, 1, 3, 3, 3)
    out = perception(state)

    assert out.shape == (1, 3, 3, 3, 3)
    assert torch.isclose(out[0, 0, 1, 1, 1], torch.tensor(1.0))
    assert torch.isclose(out[0, 1, 1, 1, 1], torch.tensor(6.0))
    assert torch.isclose(out[0, 2, 1, 1, 1], torch.tensor(0.0))


def test_perception_weights_are_frozen_and_channel_groups_scale() -> None:
    in_channels = 5
    perception = Perception3D(PerceptionConfig(channel_groups=3), in_channels=in_channels)

    assert not perception.depthwise.weight.requires_grad
    assert perception.depthwise.weight.shape[0] == in_channels * 3


def test_perception_legacy_three_group_mode_still_supported() -> None:
    perception = Perception3D(PerceptionConfig(channel_groups=3), in_channels=1)
    weights = perception.depthwise.weight.detach()

    assert weights.shape == (3, 1, 3, 3, 3)
    assert weights[2, 0, 1, 1, 1].item() == 6.0
    assert weights[2].sum().item() == 0.0


def test_perception_directional_five_group_mode_supported() -> None:
    perception = Perception3D(PerceptionConfig(channel_groups=5), in_channels=1)
    weights = perception.depthwise.weight.detach()

    assert weights.shape == (5, 1, 3, 3, 3)
    assert weights[2, 0, 2, 1, 1].item() == 1.0
    assert weights[2, 0, 0, 1, 1].item() == -1.0
    assert weights[3, 0, 1, 2, 1].item() == 1.0
    assert weights[3, 0, 1, 0, 1].item() == -1.0
    assert weights[4, 0, 1, 1, 2].item() == 1.0
    assert weights[4, 0, 1, 1, 0].item() == -1.0


def test_cell_config_is_frozen() -> None:
    cfg = CellConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.hidden_channels = 99  # type: ignore[misc]


def test_update_rule_initializes_final_layer_to_zero() -> None:
    cell_cfg = CellConfig(hidden_channels=2, visible_channels=1, task_channels=1)
    upd_cfg = UpdateConfig(hidden_dim=8)
    rule = UpdateRule(PerceptionConfig(), cell_cfg, upd_cfg)

    final_conv = rule.mlp[-1]
    assert torch.count_nonzero(final_conv.weight).item() == 0
    assert torch.count_nonzero(final_conv.bias).item() == 0


def test_update_rule_masks_task_channels() -> None:
    cell_cfg = CellConfig(hidden_channels=2, visible_channels=1, task_channels=1)
    upd_cfg = UpdateConfig(hidden_dim=8)
    rule = UpdateRule(PerceptionConfig(), cell_cfg, upd_cfg)

    with torch.no_grad():
        for layer in rule.mlp:
            if isinstance(layer, torch.nn.Conv3d):
                layer.weight.zero_()
                if layer.bias is not None:
                    layer.bias.zero_()
        final_conv = rule.mlp[-1]
        final_conv.bias.fill_(2.0)

    g = PerceptionConfig().channel_groups
    perceived = torch.zeros(1, cell_cfg.total_channels * g, 2, 2, 2)
    alive_mask = torch.ones(1, 1, 2, 2, 2, dtype=torch.bool)
    state = torch.zeros(1, cell_cfg.total_channels, 2, 2, 2)

    delta = rule(perceived, alive_mask, state)

    expected_value = torch.tanh(torch.tensor(2.0)) * 0.1
    assert torch.allclose(delta[:, :2], torch.full_like(delta[:, :2], expected_value))
    assert torch.allclose(delta[:, 2:3], torch.zeros_like(delta[:, 2:3]))
    assert torch.allclose(delta[:, 3:], torch.full_like(delta[:, 3:], expected_value))


def test_grid_init_empty_and_seed_center_with_tasks() -> None:
    grid = _make_grid(task_channels=2)

    empty = grid.init_empty(batch_size=3, device="cpu")
    assert empty.shape == (3, grid.cell.total_channels, *grid.cfg.size)
    assert torch.count_nonzero(empty).item() == 0

    task_ids = torch.tensor([1, 0])
    seed = grid.seed_center(batch_size=2, device="cpu", task_ids=task_ids)

    center = tuple(size // 2 for size in grid.cfg.size)
    assert seed.shape == (2, grid.cell.total_channels, *grid.cfg.size)
    assert torch.count_nonzero(seed[:, : grid.cell.cfg.hidden_channels]).item() == 0
    assert torch.allclose(seed[:, -1, center[0], center[1], center[2]], torch.ones(2))
    assert torch.count_nonzero(seed[:, -grid.cell.cfg.visible_channels : -1]).item() > 0

    task_slice = seed[
        :, -(grid.cell.cfg.visible_channels + grid.cell.cfg.task_channels) : -grid.cell.cfg.visible_channels
    ]
    expected = torch.tensor([[0.0, 1.0], [1.0, 0.0]]).view(2, 2, 1, 1, 1)
    assert torch.allclose(task_slice, expected.expand_as(task_slice))


def test_grid_seed_center_works_for_single_voxel_grid() -> None:
    grid = _make_grid(grid_size=(1, 1, 1), task_channels=0)
    state = grid.seed_center(batch_size=1, device="cpu")

    assert state.shape == (1, grid.cell.total_channels, 1, 1, 1)
    assert torch.allclose(state[:, -1], torch.ones_like(state[:, -1]))


def test_nca_config_from_dict_and_checkpoint_round_trip(tmp_path) -> None:
    cfg = NCAConfig.from_dict(
        {
            "grid_size": [3, 4, 5],
            "hidden_channels": 2,
            "visible_channels": 4,
            "alive_threshold": 0.2,
            "task_channels": 1,
            "update_hidden_dim": 8,
            "unknown": "ignored",
        }
    )

    assert cfg.grid_size == (3, 4, 5)
    assert cfg.task_channels == 1

    model = NCAModel(cfg)
    path = tmp_path / "model.pt"
    model.save(path.as_posix())

    loaded = NCAModel.load(path.as_posix(), device="cpu")

    # Check the known behavioral subset; avoid brittle equality against future defaults.
    assert loaded.config.grid_size == model.config.grid_size
    assert loaded.config.hidden_channels == model.config.hidden_channels
    assert loaded.config.visible_channels == model.config.visible_channels
    assert loaded.config.task_channels == model.config.task_channels
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_nca_config_round_trip_all_current_and_future_fields() -> None:
    cfg = NCAConfig()
    as_dict = dataclasses.asdict(cfg)

    restored = NCAConfig.from_dict(as_dict)
    for field in dataclasses.fields(NCAConfig):
        assert getattr(restored, field.name) == getattr(cfg, field.name)


def test_nca_config_from_dict_uses_defaults_for_missing_fields() -> None:
    restored = NCAConfig.from_dict({"grid_size": [7, 8, 9], "unknown_param": 123})

    assert restored.grid_size == (7, 8, 9)
    # These checks intentionally validate defaults without pinning the full schema.
    assert restored.hidden_channels == NCAConfig().hidden_channels
    assert restored.visible_channels == NCAConfig().visible_channels