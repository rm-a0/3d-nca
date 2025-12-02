from typing import Optional, Tuple
import numpy as np
from torch import Tensor

def extract_visible(tensor: Tensor, visible_channels: Optional[int] = None) -> Tensor:
    if tensor.ndim != 5:
        raise ValueError(f"Expected tensor shape [B, C, X, Y, Z], got {tensor.shape}")
    if tensor.shape[0] != 1:
        raise ValueError(f"Visualization assumes batch size 1, got {tensor.shape[0]}")
    
    tensor = tensor.squeeze(0)
    
    if visible_channels is None:
        return tensor
    else:
        if tensor.shape[0] < visible_channels:
            raise ValueError(f"Tensor has {tensor.shape[0]} channels, but visible_channels={visible_channels}")
        return tensor[-visible_channels:]

def extract_alpha(visible_tensor: Tensor) -> np.ndarray:
    arr = visible_tensor.detach().cpu().numpy()
    num_channels = arr.shape[0]
    if num_channels == 1:
        alpha = arr[0]
    elif num_channels == 3:
        alpha = np.mean(arr, axis=0)
    elif num_channels == 4:
        alpha = arr[3]
    else:
        raise ValueError(f"Visible channels must be 1, 3, or 4 for alpha extraction, got {num_channels}")
    return np.clip(alpha, 0, 1)

def extract_rgb(visible_tensor: Tensor) -> np.ndarray:
    arr = visible_tensor.detach().cpu().numpy()
    num_channels = arr.shape[0]
    if num_channels not in (3, 4):
        raise ValueError(f"Visible channels must be 3 or 4 for RGB extraction, got {num_channels}")
    rgb = arr[:3]
    rgb = np.transpose(rgb, (1, 2, 3, 0))
    return np.clip(rgb, 0, 1)

def extract_rgba(visible_tensor: Tensor) -> np.ndarray:
    arr = visible_tensor.detach().cpu().numpy()
    num_channels = arr.shape[0]
    if num_channels not in (3, 4):
        raise ValueError(f"Visible channels must be 3 or 4 for RGBA extraction, got {num_channels}")
    if num_channels == 3:
        rgb = arr
        alpha = np.mean(arr, axis=0, keepdims=True)
        rgba = np.concatenate([rgb, alpha], axis=0)
    else:
        rgba = arr
    rgba = np.transpose(rgba, (1, 2, 3, 0))
    return np.clip(rgba, 0, 1)

def get_slice_2d(arr: np.ndarray, axis: int = 2, idx: Optional[int] = None) -> np.ndarray:
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

def get_voxels_above_threshold(alpha: np.ndarray, threshold: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, zs = np.nonzero(alpha > threshold)
    values = alpha[xs, ys, zs]
    return xs, ys, zs, values

def generate_checkerboard(height: int, width: int, scale: float = 0.2, base: float = 0.4) -> np.ndarray:
    checker = np.indices((height, width)).sum(axis=0) % 2
    checker = checker.astype(float) * scale + base
    return checker

def normalize_values(values: np.ndarray, min_val: float = None, max_val: float = None) -> np.ndarray:
    if min_val is None:
        min_val = values.min()
    if max_val is None:
        max_val = values.max()
    denom = max_val - min_val + 1e-8
    return (values - min_val) / denom