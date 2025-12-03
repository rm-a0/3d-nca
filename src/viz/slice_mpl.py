import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from torch import Tensor
from .utils import extract_visible, extract_alpha, extract_rgba, get_slice_2d, generate_checkerboard

def show_slice_alpha_mpl(
    tensor: Tensor,
    visible_channels: Optional[int] = None,
    axis: int = 2,
    idx: Optional[int] = None,
    title: str = "Alpha Slice",
    cmap: str = "viridis",
    vmin: float = 0,
    vmax: float = 1,
    ax=None,
    show: bool = True,
) -> None:
    visible = extract_visible(tensor, visible_channels)
    alpha = extract_alpha(visible)
    slice_2d = get_slice_2d(alpha, axis, idx)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    
    im = ax.imshow(slice_2d, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(title)
    ax.axis('off')
    
    if show:
        plt.colorbar(im, ax=ax)
        plt.show()

def show_slice_color_mpl(
    tensor: Tensor,
    visible_channels: Optional[int] = None,
    axis: int = 2,
    idx: Optional[int] = None,
    title: str = "Color Slice",
    ax=None,
    show: bool = True,
) -> None:
    visible = extract_visible(tensor, visible_channels)
    rgba = extract_rgba(visible)
    slice_2d = get_slice_2d(rgba, axis, idx)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    
    h, w = slice_2d.shape[:2]
    checker = generate_checkerboard(h, w)
    rgb = slice_2d[:, :, :3]
    alpha = slice_2d[:, :, 3:4]
    display = rgb * alpha + np.stack([checker] * 3, axis=-1) * (1 - alpha)
    
    ax.imshow(np.clip(display, 0, 1), interpolation='nearest')
    ax.set_title(title)
    ax.axis('off')
    
    if show:
        plt.show()

def show_slice_alpha_comparison_mpl(
    state: Tensor,
    target: Tensor,
    visible_channels: Optional[int] = None,
    axis: int = 2,
    idx: Optional[int] = None,
    cmap: str = "viridis",
    vmin: float = 0,
    vmax: float = 1,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    show_slice_alpha_mpl(state, visible_channels=visible_channels, axis=axis, idx=idx, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax1, show=False)
    show_slice_alpha_mpl(target, visible_channels=visible_channels, axis=axis, idx=idx, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax2, show=False)
    ax1.set_title("NCA Output (Alpha)")
    ax2.set_title("Target (Alpha)")
    plt.tight_layout()
    plt.show()

def show_slice_color_comparison_mpl(
    state: Tensor,
    target: Tensor,
    visible_channels: Optional[int] = None,
    axis: int = 2,
    idx: Optional[int] = None,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    show_slice_color_mpl(state, visible_channels=visible_channels, axis=axis, idx=idx, ax=ax1, show=False)
    show_slice_color_mpl(target, visible_channels=visible_channels, axis=axis, idx=idx, ax=ax2, show=False)
    ax1.set_title("NCA Output (Color)")
    ax2.set_title("Target (Color)")
    plt.tight_layout()
    plt.show()