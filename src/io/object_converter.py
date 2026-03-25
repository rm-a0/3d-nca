import numpy as np
import torch
from torch import Tensor
from typing import Tuple, Literal
import trimesh

def obj_to_tensor(
    filepath: str,
    grid_size: Tuple[int, int, int] = (100, 100, 100),
    mode: Literal["rgba", "alpha"] = "rgba",
    device: str = "cpu",
) -> Tensor:
    """Convert an OBJ file to a voxelized tensor representation.
    
    Args:
        filepath: Path to OBJ file to load
        grid_size: Target voxel grid dimensions (D, H, W)
        mode: Output channels — 'rgba' (4 channels: R,G,B,A) or 'alpha' (1 channel: A only)
        device: Device to place tensor on ('cpu' or 'cuda')
    
    Returns:
        Tensor with shape (1, C, D, H, W):
        - Batch dimension: 1 (single seed state)
        - C channels: 4 for RGBA, 1 for alpha
        - D, H, W: voxel grid dimensions
        
        Uses internal (B,C,D,H,W) format (batch-first) for PyTorch compatibility.
        To convert to external (D,H,W,C) format: tensor[0].permute(1,2,3,0).numpy()
    """
    mesh = trimesh.load_mesh(filepath)
    
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    scale = (bounds[1] - bounds[0]).max()
    mesh.vertices = (mesh.vertices - center) / scale
    
    voxels = mesh.voxelized(pitch=2.0 / max(grid_size))
    
    voxel_matrix = voxels.matrix
    
    if voxel_matrix.shape != grid_size:
        target_grid = np.zeros(grid_size, dtype=bool)
        
        src_shape = voxel_matrix.shape
        offsets = [(g - s) // 2 for g, s in zip(grid_size, src_shape)]
        
        slices_src = []
        slices_tgt = []
        for i in range(3):
            if offsets[i] >= 0:
                s_start, s_end = 0, src_shape[i]
                t_start, t_end = offsets[i], offsets[i] + src_shape[i]
            else:
                s_start, s_end = -offsets[i], -offsets[i] + grid_size[i]
                t_start, t_end = 0, grid_size[i]
            slices_src.append(slice(s_start, s_end))
            slices_tgt.append(slice(t_start, t_end))
        
        target_grid[slices_tgt[0], slices_tgt[1], slices_tgt[2]] = \
            voxel_matrix[slices_src[0], slices_src[1], slices_src[2]]
        voxel_matrix = target_grid
    
    voxel_float = voxel_matrix.astype(np.float32)
    
    # Validate shape before stacking channels
    assert voxel_float.ndim == 3, f"Voxel matrix must be 3D (D,H,W), got {voxel_float.ndim}D"
    
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        vertex_colors = mesh.visual.vertex_colors[:, :3] / 255.0  # RGB, normalized
        avg_color = vertex_colors.mean(axis=0)
    else:
        avg_color = np.array([1.0, 1.0, 1.0])
    if mode == "rgba":
        r_channel = voxel_float * avg_color[0]
        g_channel = voxel_float * avg_color[1]
        b_channel = voxel_float * avg_color[2]
        a_channel = voxel_float
        
        tensor = np.stack([r_channel, g_channel, b_channel, a_channel], axis=0)  # (C=4, D, H, W)
    elif mode == "alpha":
        tensor = voxel_float[np.newaxis, ...]  # (C=1, D, H, W)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'rgba' or 'alpha'")
    
    # Add batch dimension: (C, D, H, W) → (1, C, D, H, W)
    result = torch.from_numpy(tensor).unsqueeze(0).to(device)
    assert result.ndim == 5, f"Result must be 5D (B,C,D,H,W), got {result.ndim}D"
    assert result.shape[0] == 1, f"Batch dimension must be 1, got {result.shape[0]}"
    assert result.shape[1] == (4 if mode == "rgba" else 1), f"Channel count mismatch for mode {mode}"
    return result