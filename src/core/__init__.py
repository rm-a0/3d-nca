"""
Core NCA Package.

Exports core model components, configurations, scheduling primitives,
and training runtime utilities for 3D neural cellular automata.
"""

from .cell import CellConfig, CellState
from .perception import PerceptionConfig, Perception3D
from .update import UpdateConfig, UpdateRule
from .grid import GridConfig, Grid3D
from .nca_model import NCAModel, NCAConfig
from .schedule import Schedule, Event, EventType
from .runner import NCARunner

__all__ = [
    "NCAModel",
    "NCAConfig",
    "Grid3D",
    "GridConfig",
    "CellState",
    "CellConfig",
    "Perception3D",
    "PerceptionConfig",
    "UpdateRule",
    "UpdateConfig",
    "Schedule",
    "Event",
    "EventType",
    "NCARunner",
]
