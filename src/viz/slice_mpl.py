import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor


def _alpha_np(tensor: Tensor) -> np.ndarray:
    if tensor.ndim == 5:
        tensor = tensor[0]
    
    alpha = tensor[-1].detach().cpu().numpy()
    alpha = np.clip(alpha, 0, 1)
    
    return alpha


def _rgba_np(tensor: Tensor) -> np.ndarray:
    if tensor.ndim == 5:
        tensor = tensor[0]
    
    rgba = tensor[-4:].detach().cpu().numpy()
    rgba = np.transpose(rgba, (1, 2, 3, 0))
    rgba = np.clip(rgba, 0, 1)
    
    return rgba


def _get_slice(arr: np.ndarray, axis: int = 2, idx: int | None = None):
    if idx is None:
        idx = arr.shape[axis] // 2
    
    if arr.ndim == 3:
        if axis == 0:
            return arr[idx, :, :]
        elif axis == 1:
            return arr[:, idx, :]
        else:
            return arr[:, :, idx]
    elif arr.ndim == 4:
        if axis == 0:
            return arr[idx, :, :, :]
        elif axis == 1:
            return arr[:, idx, :, :]
        else:
            return arr[:, :, idx, :]
    else:
        raise ValueError(f"Expected 3D or 4D array, got shape {arr.shape}")

def show_slice_mpl(
    state: Tensor,
    *,
    axis: int = 2,
    idx: int | None = None,
    title: str = "Slice",
    cmap: str = "viridis",
    vmin: float = 0,
    vmax: float = 1,
    ax=None,
    show: bool = True,
) -> None:
    alpha = _alpha_np(state)
    cur = _get_slice(alpha, axis, idx)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    
    im = ax.imshow(cur, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(title)
    ax.axis('off')
    
    if show:
        plt.colorbar(im, ax=ax)
        plt.show()


def show_slice_comparison_mpl(
    state: Tensor,
    target: Tensor,
    *,
    axis: int = 2,
    idx: int | None = None,
    cmap: str = "viridis",
    vmin: float = 0,
    vmax: float = 1,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    show_slice_mpl(state, axis=axis, idx=idx, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax1, show=False)
    show_slice_mpl(target, axis=axis, idx=idx, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax2, show=False)
    plt.tight_layout()
    plt.show()


def show_slice_rgba_mpl(
    state: Tensor,
    *,
    axis: int = 2,
    idx: int | None = None,
    title: str = "RGBA Slice",
    ax=None,
    show: bool = True,
) -> None:
    rgba = _rgba_np(state)
    slice_2d = _get_slice(rgba, axis, idx)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    
    h, w = slice_2d.shape[:2]
    checker = np.indices((h, w)).sum(axis=0) % 2
    checker = checker.astype(float) * 0.2 + 0.4
    
    rgb = slice_2d[:, :, :3]
    alpha = slice_2d[:, :, 3:4]
    
    display = rgb * alpha + np.stack([checker]*3, axis=-1) * (1 - alpha)
    
    ax.imshow(np.clip(display, 0, 1), interpolation='nearest')
    ax.set_title(title)
    ax.axis('off')
    
    if show:
        plt.show()


def show_slice_comparison_rgba_mpl(
    state: Tensor,
    target: Tensor,
    *,
    axis: int = 2,
    idx: int | None = None,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    show_slice_rgba_mpl(state, axis=axis, idx=idx, title="NCA Output", ax=ax1, show=False)
    show_slice_rgba_mpl(target, axis=axis, idx=idx, title="Target", ax=ax2, show=False)
    plt.tight_layout()
    plt.show()