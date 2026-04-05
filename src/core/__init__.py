"""
Core NCA Package.

Exports model components, configurations, scheduling primitives,
and the training runner interface.
"""

from .cell import CellConfig, CellState
from .perception import PerceptionConfig, Perception3D
from .update import UpdateConfig, UpdateRule
from .grid import GridConfig, Grid3D
from .nca_model import NCAModel, NCAConfig
from .schedule import Schedule, Event, EventType
from .runner import TrainingRunner, TrainingSnapshot, NCARunner

__all__ = [
    # High level wrapper
    "NCAModel",
    "NCAConfig",
    # Core components
    "Grid3D",
    "GridConfig",
    "CellState",
    "CellConfig",
    "Perception3D",
    "PerceptionConfig",
    "UpdateRule",
    "UpdateConfig",
    # Scheduling
    "Schedule",
    "Event",
    "EventType",
    # Training interface
    "TrainingRunner",
    "TrainingSnapshot",
    "NCARunner",
]