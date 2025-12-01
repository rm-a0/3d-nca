from .slice_mpl import (
    show_slice_mpl,
    show_slice_comparison_mpl,
    show_slice_rgba_mpl,
    show_slice_comparison_rgba_mpl,
)

from .volume_mpl import (
    show_volume_mpl,
    show_volume_comparison_mpl,
)

from .volume_pv import (
    show_volume_pv,
    show_volume_comparison_pv,
    show_volume_rgba_pv,
    show_volume_comparison_rgba_pv,
)

__all__ = [
    # 2D slices (matplotlib)
    "show_slice_mpl",
    "show_slice_comparison_mpl",
    "show_slice_rgba_mpl",
    "show_slice_comparison_rgba_mpl",
    # 3D volumes (matplotlib)
    "show_volume_mpl",
    "show_volume_comparison_mpl",
    # 3D volumes (pyvista)
    "show_volume_pv",
    "show_volume_comparison_pv",
    "show_volume_rgba_pv",
    "show_volume_comparison_rgba_pv",

]