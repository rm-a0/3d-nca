from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from typing import Optional, Tuple

def _extract_alpha(tensor: Tensor) -> np.ndarray:
    arr = tensor.squeeze(0).cpu().numpy()
    if arr.shape[0] == 1:
        return arr[0]
    elif arr.shape[0] == 3:
        return arr.mean(axis=0)
    elif arr.shape[0] == 4:
        return arr[3]
    else:
        raise ValueError(f"Unexpected channel dimension: {arr.shape[0]}")

def _alpha_np(tensor: Tensor) -> np.ndarray:
    """Extract alpha channel as NumPy array [X, Y, Z]."""
    return tensor[:, -1:, ...].squeeze(0).squeeze(0).detach().cpu().numpy()

def show_volume_mpl(
    state: Tensor,
    *,
    threshold: float = 0.2,
    figsize: Tuple[int, int] = (8, 8),
    cmap: str = "viridis",
    point_size: int = 6,
    title: str = "Volume",
    view_angle: Optional[Tuple[float, float]] = None,
    show: bool = True,
) -> int:
    """Show Tensor alpha channel as 3D scatter plot."""
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
    target_np = _extract_alpha(target)
    
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