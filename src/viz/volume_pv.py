import numpy as np
from torch import Tensor
import pyvista as pv

def show_volume_nca_pv():
    return

def show_volume_target_pv():
    return

def show_volume_comparison_pv():
    return

def show_volume_nca_color():
    return

def show_volume_target_color():
    return

def show_volume_comparison_color():
    return

def show_volume_target_color(
    target: Tensor,
    *,
    threshold: float = 0.05,
    point_size: float = 10.0,
    title: str = "Colored Target Volume",
    show_grid: bool = False,
    notebook: bool = False,
):
    """Visualize a colored voxel volume (RGB channels) using PyVista."""
    vol = target.squeeze(0).cpu().numpy()
    r, g, b = vol[0], vol[1], vol[2]

    alpha = (r + g + b) / 3.0
    xs, ys, zs = np.nonzero(alpha > threshold)

    if len(xs) == 0:
        print("No voxels above threshold.")
        return

    colors = np.stack([r[xs, ys, zs], g[xs, ys, zs], b[xs, ys, zs]], axis=1)
    colors = np.clip(colors * 255, 0, 255).astype(np.uint8)

    points = np.vstack([xs, ys, zs]).T.astype(np.float32)

    cloud = pv.PolyData(points)
    cloud.point_data["colors"] = colors
    cloud.active_scalars_name = "colors"

    pl = pv.Plotter(notebook=notebook)
    pl.add_points(
        cloud,
        scalars="colors",
        rgb=True,
        point_size=point_size,
        render_points_as_spheres=True,
    )

    pl.add_title(f"{title} ({len(xs)} voxels)")
    if show_grid:
        pl.show_grid()

    return pl.show()