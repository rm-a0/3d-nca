# src/viz/slice.py
"""
2-D slice visualisation - centre slice of the alpha (alive) channel.

Usage:
    show_nca(state)                     # NCA only
    show_target(target)                 # target only
    show_comparison(state, target)      # side-by-side
    show_slice_at(state, idx)           # manual slice index
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from typing import Optional

def _alpha_np(tensor: Tensor) -> "np.ndarray":
    """Return alpha channel as numpy on CPU."""
    return tensor[:, -1:, ...].squeeze(0).squeeze(0).detach().cpu().numpy()


def show_nca(
    state: Tensor,
    *,
    axis: int = 0,
    title: Optional[str] = None,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """Show centre slice of the NCA alpha channel."""
    alpha = _alpha_np(state)
    mid = alpha.shape[axis] // 2
    cur = alpha[mid] if axis == 0 else alpha[:, mid, :] if axis == 1 else alpha[:, :, mid]

    plt.figure(figsize=(5, 4.5))
    im = plt.imshow(cur, vmin=vmin, vmax=vmax, cmap=cmap, origin="lower")
    plt.title(title or f"NCA alpha (slice {mid}, axis={['X','Y','Z'][axis]})")
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def show_target(
    target: Tensor,
    *,
    axis: int = 0,
    title: str = "Target",
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """Show centre slice of the target (assumed [B,1,X,Y,Z])."""
    alpha = target.squeeze(0).squeeze(0).cpu().numpy()
    mid = alpha.shape[axis] // 2
    cur = alpha[mid] if axis == 0 else alpha[:, mid, :] if axis == 1 else alpha[:, :, mid]

    plt.figure(figsize=(5, 4.5))
    im = plt.imshow(cur, vmin=vmin, vmax=vmax, cmap=cmap, origin="lower")
    plt.title(title)
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def show_comparison(
    state: Tensor,
    target: Tensor,
    *,
    axis: int = 0,
    title: Optional[str] = None,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """Show NCA and target centre slices side-by-side."""
    alpha_nca = _alpha_np(state)
    alpha_tgt = target.squeeze(0).squeeze(0).cpu().numpy()
    mid = alpha_nca.shape[axis] // 2

    def _slice(arr):
        return arr[mid] if axis == 0 else arr[:, mid, :] if axis == 1 else arr[:, :, mid]

    nca_slice = _slice(alpha_nca)
    tgt_slice = _slice(alpha_tgt)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    im1 = ax1.imshow(nca_slice, vmin=vmin, vmax=vmax, cmap=cmap, origin="lower")
    ax1.set_title(title or f"NCA α (slice {mid})")
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    im2 = ax2.imshow(tgt_slice, vmin=vmin, vmax=vmax, cmap=cmap, origin="lower")
    ax2.set_title("Target")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    plt.tight_layout()
    plt.show()

def show_slice_at(
    state: Tensor,
    idx: int,
    *,
    axis: int = 0,
    title: Optional[str] = None,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """Show NCA alpha at a user-specified slice index."""
    alpha = _alpha_np(state)
    cur = alpha[idx] if axis == 0 else alpha[:, idx, :] if axis == 1 else alpha[:, :, idx]

    plt.figure(figsize=(5, 4.5))
    im = plt.imshow(cur, vmin=vmin, vmax=vmax, cmap=cmap, origin="lower")
    plt.title(title or f"NCA alpha (slice {idx}, axis={['X','Y','Z'][axis]})")
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()