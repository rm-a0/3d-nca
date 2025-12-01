from .slice_mpl import (
    show_slice_nca,
    show_slice_target,
    show_slice_comparison,
)

from .volume_mpl import (
    show_volume_nca_mpl,
    show_volume_target_mpl,
    show_volume_comparison_mpl,
)

from .volume_pv import (
    show_volume_nca_pv,
    show_volume_target_pv,
    show_volume_comparison_pv,
    show_volume_nca_color,
    show_volume_target_color,
    show_volume_comparison_color,
)

__all__ = [
    # 2D slices (matplotlib)
    "show_slice_nca",
    "show_slice_target",
    "show_slice_comparison",
    # 3D volumes (matplotlib)
    "show_volume_nca_mpl",
    "show_volume_target_mpl",
    "show_volume_comparison_mpl",
    # 3D volumes (pyvista)
    "show_volume_nca_pv",
    "show_volume_target_pv",
    "show_volume_comparison_pv",
    "show_volume_nca_color",
    "show_volume_target_color",
    "show_volume_comparison_color",

]