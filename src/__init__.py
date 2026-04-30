__version__ = "0.1.0"

from .core import (
    # High-level wrapper (recommended)
    NCAModel,
    NCAConfig,
    # Low-level components (for advanced use)
    Grid3D,
    GridConfig,
    CellState,
    CellConfig,
    Perception3D,
    PerceptionConfig,
    UpdateRule,
    UpdateConfig,
)

__all__ = [
    "__version__",
    # High-level wrapper
    "NCAModel",
    "NCAConfig",
    # Low-level components
    "Grid3D",
    "GridConfig",
    "CellState",
    "CellConfig",
    "Perception3D",
    "PerceptionConfig",
    "UpdateRule",
    "UpdateConfig",
]