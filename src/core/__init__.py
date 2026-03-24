from .cell       import CellConfig, CellState
from .perception import PerceptionConfig, Perception3D
from .update     import UpdateConfig, UpdateRule
from .grid       import GridConfig, Grid3D
from .nca_model  import NCAModel, NCAConfig

__all__ = [
    # Low-level configs & modules
    "CellConfig",
    "CellState",
    "PerceptionConfig",
    "Perception3D",
    "UpdateConfig",
    "UpdateRule",
    "GridConfig",
    "Grid3D",
    # High-level wrapper (recommended for most users)
    "NCAModel",
    "NCAConfig",
]