"""
Visualization utilities - tensor extraction and data preparation helpers.

Handles conversion from NCA tensors [B,C,X,Y,Z] to visualization-ready
formats: extracting alpha/RGB channels, filtering by threshold, slicing
2D planes, normalizing values, and generating checkerboard backgrounds.
"""

from typing import Optional, Tuple
import numpy as np
from torch import Tensor


def extract_visible(tensor: Tensor, visible_channels: Optional[int] = None) -> Tensor:
    """Extract visible channels from state tensor, removing batch dimension.

    Args:
        tensor: State tensor with shape [B,C,X,Y,Z]. Assumes batch size 1.
        visible_channels: If specified, extract last N channels; else return all.

    Returns:
        Visible channels [C,X,Y,Z] or [visible_channels,X,Y,Z] if filtered.

    Raises:
        ValueError: If batch size != 1 or tensor rank != 5.
    """
    if tensor.ndim != 5:
        raise ValueError(f"Expected tensor shape [B, C, X, Y, Z], got {tensor.shape}")
    if tensor.shape[0] != 1:
        raise ValueError(f"Visualization assumes batch size 1, got {tensor.shape[0]}")

    tensor = tensor.squeeze(0)

    if visible_channels is None:
        return tensor
    else:
        if tensor.shape[0] < visible_channels:
            raise ValueError(
                f"Tensor has {tensor.shape[0]} channels, but visible_channels={visible_channels}"
            )
        return tensor[-visible_channels:]


def extract_alpha(visible_tensor: Tensor) -> np.ndarray:
    """Extract alpha channel from visible channels.

    Handles 1-channel (alpha only), 3-channel (RGB - average values), and
    4-channel (RGBA - last channel) cases.

    Args:
        visible_tensor: Visible channels [C,X,Y,Z].

    Returns:
        Alpha array [X,Y,Z] clipped to [0,1].

    Raises:
        ValueError: If C not in (1, 3, 4).
    """
    arr = visible_tensor.detach().cpu().numpy()
    num_channels = arr.shape[0]
    if num_channels == 1:
        alpha = arr[0]
    elif num_channels == 3:
        alpha = np.mean(arr, axis=0)
    elif num_channels == 4:
        alpha = arr[3]
    else:
        raise ValueError(
            f"Visible channels must be 1, 3, or 4 for alpha extraction, got {num_channels}"
        )
    return np.clip(alpha, 0, 1)


def extract_rgb(visible_tensor: Tensor) -> np.ndarray:
    """Extract RGB channels and transpose to display format.

    Args:
        visible_tensor: Visible channels [C,X,Y,Z] with C in (3, 4).

    Returns:
        RGB array [X,Y,Z,3] clipped to [0,1], ready for matplotlib imshow.

    Raises:
        ValueError: If C not in (3, 4).
    """
    arr = visible_tensor.detach().cpu().numpy()
    num_channels = arr.shape[0]
    if num_channels not in (3, 4):
        raise ValueError(
            f"Visible channels must be 3 or 4 for RGB extraction, got {num_channels}"
        )
    rgb = arr[:3]
    rgb = np.transpose(rgb, (1, 2, 3, 0))
    return np.clip(rgb, 0, 1)


def extract_rgba(visible_tensor: Tensor) -> np.ndarray:
    """Extract RGBA channels and transpose to display format.

    If only 3 channels (RGB), synthesize alpha as mean of RGB.

    Args:
        visible_tensor: Visible channels [C,X,Y,Z] with C in (3, 4).

    Returns:
        RGBA array [X,Y,Z,4] clipped to [0,1], ready for display.

    Raises:
        ValueError: If C not in (3, 4).
    """
    arr = visible_tensor.detach().cpu().numpy()
    num_channels = arr.shape[0]
    if num_channels not in (3, 4):
        raise ValueError(
            f"Visible channels must be 3 or 4 for RGBA extraction, got {num_channels}"
        )
    if num_channels == 3:
        rgb = arr
        alpha = np.mean(arr, axis=0, keepdims=True)
        rgba = np.concatenate([rgb, alpha], axis=0)
    else:
        rgba = arr
    rgba = np.transpose(rgba, (1, 2, 3, 0))
    return np.clip(rgba, 0, 1)


def get_slice_2d(
    arr: np.ndarray, axis: int = 2, idx: Optional[int] = None
) -> np.ndarray:
    """Extract 2D slice from 3D or 4D array along specified axis.

    Args:
        arr: 3D [X,Y,Z] or 4D [X,Y,Z,C] array.
        axis: Which axis to slice (0, 1, or 2 for X, Y, Z).
        idx: Slice index; if None, uses center index along axis.

    Returns:
        2D slice [remaining_dims].

    Raises:
        ValueError: If arr rank not in (3, 4).
    """
    if idx is None:
        idx = arr.shape[axis] // 2
    if arr.ndim == 3:
        slices = [slice(None)] * 3
        slices[axis] = idx
        return arr[tuple(slices)]
    elif arr.ndim == 4:
        slices = [slice(None)] * 4
        slices[axis] = idx
        return arr[tuple(slices)]
    else:
        raise ValueError(f"Expected 3D or 4D array, got shape {arr.shape}")


def get_voxels_above_threshold(
    alpha: np.ndarray, threshold: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find voxel coordinates where alpha exceeds threshold.

    Args:
        alpha: Alpha values [X,Y,Z].
        threshold: Alpha threshold (default 0.05).

    Returns:
        Tuple (xs, ys, zs, values) - coordinate arrays and their alpha values.
    """
    xs, ys, zs = np.nonzero(alpha > threshold)
    values = alpha[xs, ys, zs]
    return xs, ys, zs, values


def generate_checkerboard(
    height: int, width: int, scale: float = 0.2, base: float = 0.4
) -> np.ndarray:
    """Generate checkerboard pattern for visualizing transparency.

    Args:
        height: Pattern height in pixels.
        width: Pattern width in pixels.
        scale: Amplitude of checkerboard pattern.
        base: Base intensity level.

    Returns:
        Checkerboard pattern [height,width] with values in [base, base+scale].
    """
    checker = np.indices((height, width)).sum(axis=0) % 2
    checker = checker.astype(float) * scale + base
    return checker


def normalize_values(
    values: np.ndarray, min_val: Optional[float] = None, max_val: Optional[float] = None
) -> np.ndarray:
    """Normalize array values to [0,1] range.

    Args:
        values: Input array to normalize.
        min_val: Minimum value for normalization; if None, uses array min.
        max_val: Maximum value for normalization; if None, uses array max.

    Returns:
        Normalized array scaled to [0,1].
    """
    if min_val is None:
        min_val = values.min()
    if max_val is None:
        max_val = values.max()
    denom = max_val - min_val + 1e-8
    return (values - min_val) / denom
