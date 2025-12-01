"""
3-D volume visualization utilities for NCA (matplotlib-based).
Usage:
    show_volume_nca_mpl(state)                    # Show NCA 3D scatter
    show_volume_target_mpl(target)                # Show target 3D scatter
    show_volume_comparison_mpl(state, target)     # Side-by-side 3D comparison
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from typing import Optional, Tuple

def _alpha_np(tensor: Tensor) -> np.ndarray:
    """Extract alpha channel as NumPy array [X, Y, Z]."""
    return tensor[:, -1:, ...].squeeze(0).squeeze(0).detach().cpu().numpy()

def show_volume_nca_mpl(
    state: Tensor,
    *,
    threshold: float = 0.2,
    figsize: Tuple[int, int] = (8, 8),
    cmap: str = "viridis",
    point_size: int = 6,
    title: str = "NCA Volume",
    view_angle: Optional[Tuple[float, float]] = None,
    show: bool = True,
) -> int:
    """Show NCA alpha channel as 3D scatter plot."""
    alpha = _alpha_np(state)
    xs, ys, zs = np.nonzero(alpha > threshold)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    
    if len(xs) > 0:
        ax.scatter(xs, ys, zs, s=point_size, c=alpha[alpha > threshold], cmap=cmap)
    
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

def show_volume_target_mpl(
    target: Tensor,
    *,
    threshold: float = 0.2,
    figsize: Tuple[int, int] = (8, 8),
    cmap: str = "viridis",
    point_size: int = 6,
    title: str = "Target Volume",
    view_angle: Optional[Tuple[float, float]] = None,
    show: bool = True,
) -> int:
    """Show target volume as 3D scatter plot."""
    alpha = target.squeeze(0).squeeze(0).cpu().numpy()
    xs, ys, zs = np.nonzero(alpha > threshold)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    
    if len(xs) > 0:
        ax.scatter(xs, ys, zs, s=point_size, 
                  c=alpha[alpha > threshold], cmap=cmap)
    
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

def show_volume_comparison_mpl(
    state: Tensor,
    target: Tensor,
    *,
    threshold: float = 0.2,
    figsize: Tuple[int, int] = (14, 6),
    cmap: str = "viridis",
    point_size: int = 6,
    title_prefix: str = "",
    view_angle: Optional[Tuple[float, float]] = None,
) -> Tuple[int, int]:
    """Show NCA and target volumes side-by-side as 3D scatter plots."""
    alpha = _alpha_np(state)
    target_np = target.squeeze(0).squeeze(0).cpu().numpy()
    
    xs, ys, zs = np.nonzero(alpha > threshold)
    xs_t, ys_t, zs_t = np.nonzero(target_np > threshold)
    
    fig = plt.figure(figsize=figsize)
    
    ax1 = fig.add_subplot(121, projection="3d")
    if len(xs) > 0:
        ax1.scatter(xs, ys, zs, s=point_size, c=alpha[alpha > threshold], cmap=cmap)
    ax1.set_title(f"{title_prefix}Predicted ({len(xs)} voxels)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    if view_angle:
        ax1.view_init(elev=view_angle[0], azim=view_angle[1])
    
    ax2 = fig.add_subplot(122, projection="3d")
    if len(xs_t) > 0:
        ax2.scatter(xs_t, ys_t, zs_t, s=point_size, c=target_np[target_np > threshold], cmap=cmap)
    ax2.set_title(f"{title_prefix}Target ({len(xs_t)} voxels)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    if view_angle:
        ax2.view_init(elev=view_angle[0], azim=view_angle[1])
    
    plt.tight_layout()
    plt.show()
    
    return len(xs), len(xs_t)