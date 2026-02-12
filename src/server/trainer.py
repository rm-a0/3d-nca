import torch
from src.core import Grid3D, CellConfig, PerceptionConfig, UpdateConfig, GridConfig

class NCATrainer:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.target = None

    def init(self, config, target):
        cell_cfg = CellConfig(**config["cell"])
        perc_cfg = PerceptionConfig(**config["perception"])
        upd_cfg = UpdateConfig(**config["update"])
        grid_cfg = GridConfig(**config["grid"])

        self.model = Grid3D(cell_cfg, perc_cfg, upd_cfg, grid_cfg)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["training"]["learning_rate"])
        self.target = target

    def step(self):
        # TODO: Implement training step logic
        pass