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
from .utils import (
    extract_visible,
    extract_alpha,
    extract_rgba,
    get_voxels_above_threshold,
)


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


def _surface_mask(alpha: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    """Return a boolean mask of surface voxels above alpha threshold."""
    filled = alpha > threshold
    interior = np.ones_like(filled)
    for axis in range(3):
        interior &= np.roll(filled, 1, axis=axis)
        interior &= np.roll(filled, -1, axis=axis)
    return filled & ~interior


def _to_rgba_array(
    target: np.ndarray | Tensor, visible_channels: int = 4
) -> np.ndarray:
    """Convert target/state tensor or array to clipped RGBA array [D,H,W,4]."""
    if isinstance(target, Tensor):
        visible = extract_visible(target, visible_channels=visible_channels)
        return np.clip(extract_rgba(visible), 0.0, 1.0)

    if not isinstance(target, np.ndarray):
        raise TypeError(
            f"Expected numpy.ndarray or Tensor, got {type(target).__name__}"
        )
    if target.ndim != 4:
        raise ValueError(f"Expected target shape (D,H,W,C), got {target.shape}")

    channels = target.shape[-1]
    if channels == 4:
        rgba = target
    elif channels == 3:
        alpha = np.mean(target, axis=-1, keepdims=True)
        rgba = np.concatenate([target, alpha], axis=-1)
    else:
        raise ValueError(f"Expected 3 or 4 channels in target, got {channels}")

    return np.clip(rgba.astype(np.float32), 0.0, 1.0)


def show_volume_rgba_mpl(
    rgba: np.ndarray,
    threshold: float = 0.15,
    surface_only: bool = True,
    figsize: Tuple[int, int] = (8, 8),
    point_size: int = 12,
    title: str = "RGBA Volume",
    view_angle: Optional[Tuple[float, float]] = (25.0, 45.0),
    ax=None,
    show: bool = True,
) -> int:
    """Display 3D RGBA voxels from array [D,H,W,4] as a scatter cloud.

    Args:
        rgba: RGBA array [D,H,W,4] with values in [0,1].
        threshold: Alpha threshold for voxel inclusion.
        surface_only: If True, render only surface voxels.
        figsize: Figure size when creating a new figure.
        point_size: Scatter point size.
        title: Plot title.
        view_angle: Tuple (elevation, azimuth) in degrees.
        ax: Optional existing 3D axes.
        show: If True, display plot when a new figure is created.

    Returns:
        Number of voxels rendered.
    """
    if rgba.ndim != 4 or rgba.shape[-1] != 4:
        raise ValueError(f"Expected RGBA shape (D,H,W,4), got {rgba.shape}")

    alpha = np.clip(rgba[..., 3], 0.0, 1.0)
    rgb = np.clip(rgba[..., :3], 0.0, 1.0)
    mask = _surface_mask(alpha, threshold) if surface_only else (alpha > threshold)
    xs, ys, zs = np.where(mask)
    colors = rgb[xs, ys, zs] if len(xs) > 0 else None

    created = ax is None
    if created:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

    if len(xs) > 0:
        ax.scatter(xs, ys, zs, c=colors, s=point_size, depthshade=True, linewidths=0)

    ax.set_title(title, pad=4)
    ax.set_xlabel("X", labelpad=1)
    ax.set_ylabel("Y", labelpad=1)
    ax.set_zlabel("Z", labelpad=1)
    ax.tick_params(labelsize=6)

    if view_angle is not None:
        ax.view_init(elev=float(view_angle[0]), azim=float(view_angle[1]))

    g = rgba.shape[0]
    ax.set_xlim(0, g)
    ax.set_ylim(0, g)
    ax.set_zlim(0, g)
    ax.set_box_aspect([1, 1, 1])

    if created and show:
        plt.tight_layout()
        plt.show()

    return len(xs)


def show_state_rgba_mpl(
    state: Tensor,
    visible_channels: int = 4,
    threshold: float = 0.2,
    surface_only: bool = True,
    figsize: Tuple[int, int] = (8, 8),
    point_size: int = 12,
    title: str = "NCA State",
    view_angle: Optional[Tuple[float, float]] = (25.0, 45.0),
    ax=None,
    show: bool = True,
) -> int:
    """Display NCA state tensor [1,C,D,H,W] as RGBA voxel scatter."""
    rgba = _to_rgba_array(state, visible_channels=visible_channels)
    return show_volume_rgba_mpl(
        rgba=rgba,
        threshold=threshold,
        surface_only=surface_only,
        figsize=figsize,
        point_size=point_size,
        title=title,
        view_angle=view_angle,
        ax=ax,
        show=show,
    )


def show_state_target_comparison_mpl(
    state: Tensor,
    target: np.ndarray | Tensor,
    visible_channels: int = 4,
    threshold: float = 0.2,
    surface_only: bool = True,
    figsize: Tuple[int, int] = (12, 5),
    point_size: int = 14,
    title_prefix: str = "",
    view_angle: Optional[Tuple[float, float]] = (28.0, 40.0),
    show: bool = True,
) -> Tuple[int, int]:
    """Display target and predicted state side-by-side in matching RGBA style.

    Args:
        state: Predicted state tensor [1,C,D,H,W].
        target: Target as array [D,H,W,C] or tensor [1,C,D,H,W].
        visible_channels: Number of visible channels in state tensor.
        threshold: Alpha threshold for voxel inclusion.
        surface_only: If True, render only surface voxels.
        figsize: Figure dimensions.
        point_size: Scatter point size.
        title_prefix: Prefix added to both subplot titles.
        view_angle: Shared camera angle (elevation, azimuth).
        show: If True, display figure.

    Returns:
        Tuple (target_voxels, state_voxels).
    """
    target_rgba = _to_rgba_array(target, visible_channels=visible_channels)

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(121, projection="3d")
    target_count = show_volume_rgba_mpl(
        rgba=target_rgba,
        threshold=threshold,
        surface_only=surface_only,
        point_size=point_size,
        title="",
        view_angle=view_angle,
        ax=ax1,
        show=False,
    )
    ax1.set_title(f"{title_prefix}Target ({target_count} voxels)")

    ax2 = fig.add_subplot(122, projection="3d")
    state_count = show_state_rgba_mpl(
        state=state,
        visible_channels=visible_channels,
        threshold=threshold,
        surface_only=surface_only,
        point_size=point_size,
        title="",
        view_angle=view_angle,
        ax=ax2,
        show=False,
    )
    ax2.set_title(f"{title_prefix}Predicted ({state_count} voxels)")

    plt.tight_layout()
    if show:
        plt.show()

    return target_count, state_count
