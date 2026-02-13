import threading
import torch
from src.core import Grid3D, CellConfig, PerceptionConfig, UpdateConfig, GridConfig

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
        self._pause_event.set() # not paused by default

    def init(self, config, target):
        cell_cfg = CellConfig(**config["cell"])
        perc_cfg = PerceptionConfig(**config["perception"])
        upd_cfg = UpdateConfig(**config["update"])
        grid_cfg = GridConfig(**config["grid"])

        self.model = Grid3D(cell_cfg, perc_cfg, upd_cfg, grid_cfg)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config["training"]["learning_rate"]
        )
        self.target = target
        self.state = self.model.seed_center(
            batch_size=config["training"]["batch_size"],
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.current_epoch = 0
        self.total_epochs = config["training"]["num_epochs"]
        self.latest_loss = 0.0

        self._stop_event.clear()
        self._pause_event.set()
        self._train_thread = threading.Thread(
            target=self._train_loop
        )
        self._train_thread.start()


    def pause(self):
        self._pause_event.clear()
        print("Training paused")

    def resume(self):
        self._pause_event.set()
        print("Training resumed")

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
            self._pause_event.wait() # wait if paused
            if self._stop_event.is_set():
                break

            loss = self._step()
            self.current_epoch = epoch
            self.latest_loss = loss
            print(f"Epoch {epoch}/{self.total_epochs} - Loss: {loss:.4f}")

        print("Training completed")

    def _step(self):
        return 0.0 # placeholder for step logic
    


