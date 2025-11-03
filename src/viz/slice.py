"""
2-D slice visualization utilities for NCA alpha (alive) channel.

Usage:
    show_nca(state)                     # Show NCA slice
    show_target(target)                 # Show target slice
    show_comparison(state, target)      # Side-by-side comparison
    show_slice_at(state, idx)           # Show custom slice index
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from typing import Optional

def _alpha_np(tensor: Tensor) -> np.ndarray:
    """Extract alpha channel (last channel) as NumPy array [X, Y, Z]."""
    return tensor[:, -1:, ...].squeeze(0).squeeze(0).detach().cpu().numpy()


def _get_slice(alpha: np.ndarray, axis: int, idx: Optional[int] = None) -> np.ndarray:
    """Return 2D slice from 3D volume at given axis/index (defaults to middle)."""
    if idx is None:
        idx = alpha.shape[axis] // 2
    if axis == 0:
        return alpha[idx]
    elif axis == 1:
        return alpha[:, idx, :]
    elif axis == 2:
        return alpha[:, :, idx]
    else:
        raise ValueError("axis must be 0, 1, or 2")

def _plot_slice(
    cur: np.ndarray,
    *,
    title: Optional[str] = None,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> None:
    """Plot a 2D slice on given axes or standalone."""
    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        own_fig = True

    im = ax.imshow(cur, vmin=vmin, vmax=vmax, cmap=cmap, origin="lower")
    ax.set_title(title or "")
    ax.axis("off")

    if own_fig:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        if show:
            plt.show()
    return im

def show_nca(
    state: Tensor,
    *,
    axis: int = 0,
    idx: Optional[int] = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> None:
    """Show NCA alpha slice (center or specific index)."""
    alpha = _alpha_np(state)
    cur = _get_slice(alpha, axis, idx)
    title = title or f"NCA α (slice {idx or alpha.shape[axis] // 2}, axis={['X','Y','Z'][axis]})"
    return _plot_slice(cur, title=title, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax, show=show)

def show_target(
    target: Tensor,
    *,
    axis: int = 0,
    idx: Optional[int] = None,
    title: str = "Target",
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> None:
    """Show target slice (center or specific index)."""
    alpha = target.squeeze(0).squeeze(0).cpu().numpy()
    cur = _get_slice(alpha, axis, idx)
    title = f"{title} (slice {idx or alpha.shape[axis] // 2}, axis={['X','Y','Z'][axis]})"
    return _plot_slice(cur, title=title, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax, show=show)

def show_comparison(
    state: Tensor,
    target: Tensor,
    *,
    axis: int = 0,
    idx: Optional[int] = None,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """Show NCA and target slices side-by-side (center or specific index)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    show_nca(state, axis=axis, idx=idx, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax1, show=False)
    show_target(target, axis=axis, idx=idx, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax2, show=False)

    plt.tight_layout()
    plt.show()