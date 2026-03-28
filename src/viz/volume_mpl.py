"""
3D volume visualization using Matplotlib.

Renders NCA state as 3D scatter plots in interactive matplotlib windows.
Supports alpha-channel and full RGBA visualization, with optional comparison
views side-by-side. Point size and colormap are configurable.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple
from torch import Tensor
from .utils import extract_visible, extract_alpha, get_voxels_above_threshold


def show_volume_alpha_mpl(
    tensor: Tensor,
    visible_channels: Optional[int] = None,
    threshold: float = 0.2,
    figsize: Tuple[int, int] = (8, 8),
    cmap: str = "viridis",
    point_size: int = 6,
    title: str = "Alpha Volume",
    view_angle: Optional[Tuple[float, float]] = None,
    show: bool = True,
) -> int:
    """Display 3D volume visualization colored by alpha values.

    Args:
        tensor: State tensor [B,C,X,Y,Z] with batch size 1.
        visible_channels: Number of visible channels to use; if None, use all.
        threshold: Alpha threshold for voxel inclusion (default 0.2).
        figsize: Figure dimensions (width, height).
        cmap: Matplotlib colormap name for alpha coloring.
        point_size: Scatter plot point size.
        title: Plot title.
        view_angle: Tuple (elevation, azimuth) for initial viewpoint; if None, auto.
        show: If True, display plot immediately.

    Returns:
        Number of voxels displayed (above threshold).
    """
    visible = extract_visible(tensor, visible_channels)
    alpha = extract_alpha(visible)
    xs, ys, zs, values = get_voxels_above_threshold(alpha, threshold)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if len(xs) > 0:
        ax.scatter(xs, ys, zs, s=point_size, c=values, cmap=cmap)

    ax.set_title(f"{title} ({len(xs)} voxels)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if view_angle:
        ax.view_init(elev=view_angle[0], azim=view_angle[1])

    plt.tight_layout()
    if show:
        plt.show()

    return len(xs)


def show_volume_alpha_comparison_mpl(
    state: Tensor,
    target: Tensor,
    visible_channels: Optional[int] = None,
    threshold: float = 0.2,
    figsize: Tuple[int, int] = (14, 6),
    cmap: str = "viridis",
    point_size: int = 6,
    title_prefix: str = "",
    view_angle: Optional[Tuple[float, float]] = None,
) -> Tuple[int, int]:
    """Display side-by-side comparison of target and predicted volumes (alpha).

    Args:
        state: Predicted state tensor [B,C,X,Y,Z].
        target: Target state tensor [B,C,X,Y,Z].
        visible_channels: Number of visible channels to use.
        threshold: Alpha threshold for voxel inclusion.
        figsize: Figure dimensions for both subplots.
        cmap: Colormap for alpha coloring.
        point_size: Scatter point size.
        title_prefix: String prepended to subplot titles.
        view_angle: Viewpoint (elevation, azimuth).

    Returns:
        Tuple (target_voxel_count, predicted_voxel_count).
    """
    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(121, projection="3d")
    visible_t = extract_visible(target, visible_channels)
    alpha_t = extract_alpha(visible_t)
    xs_t, ys_t, zs_t, vals_t = get_voxels_above_threshold(alpha_t, threshold)
    if len(xs_t) > 0:
        ax1.scatter(xs_t, ys_t, zs_t, s=point_size, c=vals_t, cmap=cmap)
    ax1.set_title(f"{title_prefix}Target ({len(xs_t)} voxels)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    if view_angle:
        ax1.view_init(elev=view_angle[0], azim=view_angle[1])

    ax2 = fig.add_subplot(122, projection="3d")
    visible_s = extract_visible(state, visible_channels)
    alpha_s = extract_alpha(visible_s)
    xs_s, ys_s, zs_s, vals_s = get_voxels_above_threshold(alpha_s, threshold)
    if len(xs_s) > 0:
        ax2.scatter(xs_s, ys_s, zs_s, s=point_size, c=vals_s, cmap=cmap)
    ax2.set_title(f"{title_prefix}Predicted ({len(xs_s)} voxels)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    if view_angle:
        ax2.view_init(elev=view_angle[0], azim=view_angle[1])

    plt.tight_layout()
    plt.show()

    return len(xs_t), len(xs_s)
