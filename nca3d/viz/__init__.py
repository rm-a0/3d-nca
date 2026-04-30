"""
Visualization Package.

Exports 2D slice and 3D volume visualization helpers for Matplotlib
and PyVista render backends.
"""

from .slice_mpl import (
    show_slice_alpha_mpl,
    show_slice_color_comparison_mpl,
    show_slice_color_mpl,
    show_slice_alpha_comparison_mpl,
)

from .volume_mpl import (
    show_volume_alpha_mpl,
    show_volume_alpha_comparison_mpl,
    show_volume_rgba_mpl,
    show_state_rgba_mpl,
    show_state_target_comparison_mpl,
)

from .volume_pv import (
    show_volume_alpha_pv,
    show_volume_alpha_comparison_pv,
    show_volume_color_pv,
    show_volume_color_comparison_pv,
)

__all__ = [
    "show_slice_alpha_mpl",
    "show_slice_color_comparison_mpl",
    "show_slice_color_mpl",
    "show_slice_alpha_comparison_mpl",
    "show_volume_alpha_mpl",
    "show_volume_alpha_comparison_mpl",
    "show_volume_rgba_mpl",
    "show_state_rgba_mpl",
    "show_state_target_comparison_mpl",
    "show_volume_alpha_pv",
    "show_volume_alpha_comparison_pv",
    "show_volume_color_pv",
    "show_volume_color_comparison_pv",
]
