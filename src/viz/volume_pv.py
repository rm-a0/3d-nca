"""
3D volume visualization using PyVista.

High-performance 3D rendering for NCA state using PyVista (VTK backend).
Supports interactive rotation, zooming, and real-time updates. Two render modes:
- points: point cloud (fast, many voxels)
- blocks: solid cubes (slower, better for small dense volumes)

Includes alpha, color, and comparison visualization functions with optional
linked cameras for side-by-side comparison views.
"""

import numpy as np
import pyvista as pv
from typing import Literal, Optional
from torch import Tensor
from .utils import (
    extract_visible,
    extract_alpha,
    extract_rgb,
    get_voxels_above_threshold,
    normalize_values,
)


def _add_alpha_points(
    pl, xs, ys, zs, alpha_values, point_size, opacity, cmap, clim=None
):
    """Add alpha-colored point cloud to PyVista plotter.

    Args:
        pl: PyVista Plotter instance.
        xs, ys, zs: Voxel coordinates.
        alpha_values: Alpha values for each point.
        point_size: Point size for rendering.
        opacity: Global opacity [0,1].
        cmap: Colormap name for alpha coloring.
        clim: Tuple (min, max) for colormap range.
    """
    points = np.vstack([xs, ys, zs]).T.astype(np.float32)
    cloud = pv.PolyData(points)
    cloud.point_data["alpha"] = alpha_values
    actor = pl.add_points(
        cloud,
        scalars="alpha",
        cmap=cmap,
        point_size=point_size,
        render_points_as_spheres=True,
        opacity=opacity,
        clim=clim,
    )
    return actor


def _add_alpha_blocks(pl, xs, ys, zs, alpha_values, opacity, cmap):
    """Add alpha-colored solid cubes to PyVista plotter.

    Args:
        pl: PyVista Plotter instance.
        xs, ys, zs: Voxel coordinates.
        alpha_values: Alpha values for each cube (normalized).
        opacity: Global opacity [0,1].
        cmap: Matplotlib colormap name for mapping alpha to color.
    """
    norm_alpha = normalize_values(alpha_values)
    import matplotlib.pyplot as plt

    colormap = plt.get_cmap(cmap)
    for i in range(len(xs)):
        cube = pv.Cube(center=(xs[i], ys[i], zs[i]), x_length=1, y_length=1, z_length=1)
        color = colormap(norm_alpha[i])[:3]
        pl.add_mesh(cube, color=color, opacity=opacity)


def _add_rgba_points(pl, xs, ys, zs, colors, point_size):
    """Add RGB-colored point cloud to PyVista plotter.

    Args:
        pl: PyVista Plotter instance.
        xs, ys, zs: Voxel coordinates.
        colors: RGB color values [X,Y,Z,3] or [N,3].
        point_size: Point size for rendering.
    """
    points = np.vstack([xs, ys, zs]).T.astype(np.float32)
    cloud = pv.PolyData(points)
    cloud.point_data["colors"] = colors
    actor = pl.add_points(
        cloud,
        scalars="colors",
        rgb=True,
        point_size=point_size,
        render_points_as_spheres=True,
    )
    return actor


def _add_rgba_blocks(pl, xs, ys, zs, colors):
    """Add RGB-colored solid cubes to PyVista plotter.

    Args:
        pl: PyVista Plotter instance.
        xs, ys, zs: Voxel coordinates.
        colors: RGB color values [N,3] as uint8.
    """
    for i in range(len(xs)):
        cube = pv.Cube(center=(xs[i], ys[i], zs[i]), x_length=1, y_length=1, z_length=1)
        pl.add_mesh(cube, color=colors[i])


def show_volume_alpha_pv(
    tensor: Tensor,
    visible_channels: Optional[int] = None,
    threshold: float = 0.05,
    point_size: float = 10.0,
    title: str = "Alpha Volume",
    show_grid: bool = False,
    notebook: bool = False,
    render_mode: Literal["blocks", "points"] = "points",
    opacity: float = 1.0,
    cmap: str = "viridis_r",
):
    """Display 3D volume visualization colored by alpha using PyVista.

    Interactive 3D render with mouse rotation/zoom. Returns plotter object
    for further customization or integration into notebooks.

    Args:
        tensor: State tensor [B,C,X,Y,Z] with batch size 1.
        visible_channels: Number of visible channels; if None, use all.
        threshold: Alpha threshold for voxel inclusion (default 0.05).
        point_size: Size of points/cubes in render (default 10.0).
        title: Window title string.
        show_grid: If True, display reference grid.
        notebook: If True, embed rendering in Jupyter notebook.
        render_mode: "points" for point cloud (fast) or "blocks" for solid cubes.
        opacity: Global opacity for all voxels [0,1].
        cmap: Colormap for alpha-to-color mapping, reversed by default (viridis_r).

    Returns:
        PyVista Plotter instance (call .show() for interactive display).
    """
    visible = extract_visible(tensor, visible_channels)
    alpha = extract_alpha(visible)
    xs, ys, zs, alpha_values = get_voxels_above_threshold(alpha, threshold)

    if len(xs) == 0:
        print("No voxels above threshold.")
        return

    pl = pv.Plotter(notebook=notebook)

    if render_mode == "blocks":
        _add_alpha_blocks(pl, xs, ys, zs, alpha_values, opacity, cmap)
    else:
        _add_alpha_points(
            pl, xs, ys, zs, alpha_values, point_size, opacity, cmap, clim=(0, 1)
        )

    pl.add_title(f"{title} ({len(xs)} voxels)")
    if show_grid:
        pl.show_grid()

    return pl.show()


def show_volume_color_pv(
    tensor: Tensor,
    visible_channels: Optional[int] = None,
    threshold: float = 0.05,
    point_size: float = 10.0,
    title: str = "Color Volume",
    show_grid: bool = False,
    notebook: bool = False,
    render_mode: Literal["blocks", "points"] = "points",
):
    """Display 3D volume visualization with full RGB color using PyVista.

    Args:
        tensor: State tensor [B,C,X,Y,Z] with batch size 1.
        visible_channels: Number of visible channels.
        threshold: Alpha threshold for voxel inclusion (voxels above this are rendered).
        point_size: Size of points/cubes.
        title: Window title.
        show_grid: If True, display reference grid.
        notebook: If True, embed in Jupyter.
        render_mode: "points" or "blocks".

    Returns:
        PyVista Plotter instance.
    """
    visible = extract_visible(tensor, visible_channels)
    alpha = extract_alpha(visible)
    rgb = extract_rgb(visible)
    xs, ys, zs, _ = get_voxels_above_threshold(alpha, threshold)

    if len(xs) == 0:
        print("No voxels above threshold.")
        return

    colors = rgb[xs, ys, zs] * 255
    colors = np.clip(colors, 0, 255).astype(np.uint8)

    pl = pv.Plotter(notebook=notebook)

    if render_mode == "blocks":
        _add_rgba_blocks(pl, xs, ys, zs, colors)
    else:
        _add_rgba_points(pl, xs, ys, zs, colors, point_size)

    pl.add_title(f"{title} ({len(xs)} voxels)")
    if show_grid:
        pl.show_grid()

    return pl.show()


def show_volume_alpha_comparison_pv(
    state: Tensor,
    target: Tensor,
    visible_channels: Optional[int] = None,
    threshold: float = 0.05,
    point_size: float = 10.0,
    show_grid: bool = False,
    notebook: bool = False,
    render_mode: Literal["blocks", "points"] = "points",
    opacity: float = 1.0,
    cmap: str = "viridis_r",
):
    """Display side-by-side comparison of target and predicted volumes (alpha).

    Cameras are linked so rotation in one view rotates the other.

    Args:
        state: Predicted state tensor [B,C,X,Y,Z].
        target: Target state tensor [B,C,X,Y,Z].
        visible_channels: Number of visible channels.
        threshold: Alpha threshold.
        point_size: Point/cube size.
        show_grid: If True, display reference grids.
        notebook: If True, embed in Jupyter.
        render_mode: "points" or "blocks".
        opacity: Global opacity.
        cmap: Colormap for alpha.

    Returns:
        PyVista Plotter instance with 2-subplot layout and linked cameras.
    """
    visible_t = extract_visible(target, visible_channels)
    alpha_t = extract_alpha(visible_t)
    xs_t, ys_t, zs_t, vals_t = get_voxels_above_threshold(alpha_t, threshold)

    visible_s = extract_visible(state, visible_channels)
    alpha_s = extract_alpha(visible_s)
    xs_s, ys_s, zs_s, vals_s = get_voxels_above_threshold(alpha_s, threshold)

    pl = pv.Plotter(shape=(1, 2), notebook=notebook)

    pl.subplot(0, 0)
    if len(xs_t) > 0:
        if render_mode == "blocks":
            norm_vals_t = normalize_values(vals_t, 0, 1)
            _add_alpha_blocks(pl, xs_t, ys_t, zs_t, norm_vals_t, opacity, cmap)
        else:
            _add_alpha_points(
                pl, xs_t, ys_t, zs_t, vals_t, point_size, opacity, cmap, clim=(0, 1)
            )
    else:
        print("Target: No voxels above threshold.")
    pl.add_title(f"Target ({len(xs_t)} voxels)")
    if show_grid:
        pl.show_grid()

    pl.subplot(0, 1)
    if len(xs_s) > 0:
        if render_mode == "blocks":
            norm_vals_s = normalize_values(vals_s, 0, 1)
            _add_alpha_blocks(pl, xs_s, ys_s, zs_s, norm_vals_s, opacity, cmap)
        else:
            _add_alpha_points(
                pl, xs_s, ys_s, zs_s, vals_s, point_size, opacity, cmap, clim=(0, 1)
            )
    else:
        print("Prediction: No voxels above threshold.")
    pl.add_title(f"Prediction ({len(xs_s)} voxels)")
    if show_grid:
        pl.show_grid()

    pl.link_views()
    return pl.show()


def show_volume_color_comparison_pv(
    state: Tensor,
    target: Tensor,
    visible_channels: Optional[int] = None,
    threshold: float = 0.05,
    point_size: float = 10.0,
    show_grid: bool = False,
    notebook: bool = False,
    render_mode: Literal["blocks", "points"] = "points",
):
    """Display side-by-side comparison of target and predicted volumes (color).

    Full RGB color rendering with linked cameras.

    Args:
        state: Predicted state tensor [B,C,X,Y,Z].
        target: Target state tensor [B,C,X,Y,Z].
        visible_channels: Number of visible channels.
        threshold: Alpha threshold for voxel inclusion.
        point_size: Point/cube size.
        show_grid: If True, display grids.
        notebook: If True, embed in Jupyter.
        render_mode: "points" or "blocks".

    Returns:
        PyVista Plotter instance with linked cameras.
    """
    pl = pv.Plotter(shape=(1, 2), notebook=notebook)

    pl.subplot(0, 0)
    visible_t = extract_visible(target, visible_channels)
    alpha_t = extract_alpha(visible_t)
    xs_t, ys_t, zs_t, _ = get_voxels_above_threshold(alpha_t, threshold)
    if len(xs_t) > 0:
        rgb_t = extract_rgb(visible_t)
        colors_t = rgb_t[xs_t, ys_t, zs_t] * 255
        colors_t = np.clip(colors_t, 0, 255).astype(np.uint8)
        if render_mode == "blocks":
            _add_rgba_blocks(pl, xs_t, ys_t, zs_t, colors_t)
        else:
            _add_rgba_points(pl, xs_t, ys_t, zs_t, colors_t, point_size)
    else:
        print("Target: No voxels above threshold.")
    pl.add_title(f"Target ({len(xs_t)} voxels)")
    if show_grid:
        pl.show_grid()

    pl.subplot(0, 1)
    visible_s = extract_visible(state, visible_channels)
    alpha_s = extract_alpha(visible_s)
    xs_s, ys_s, zs_s, _ = get_voxels_above_threshold(alpha_s, threshold)
    if len(xs_s) > 0:
        rgb_s = extract_rgb(visible_s)
        colors_s = rgb_s[xs_s, ys_s, zs_s] * 255
        colors_s = np.clip(colors_s, 0, 255).astype(np.uint8)
        if render_mode == "blocks":
            _add_rgba_blocks(pl, xs_s, ys_s, zs_s, colors_s)
        else:
            _add_rgba_points(pl, xs_s, ys_s, zs_s, colors_s, point_size)
    else:
        print("Prediction: No voxels above threshold.")
    pl.add_title(f"Prediction ({len(xs_s)} voxels)")
    if show_grid:
        pl.show_grid()

    pl.link_views()
    return pl.show()
