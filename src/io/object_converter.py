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
    """Convert an OBJ file to a voxelized tensor representation."""
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
    
    # Convert to float
    voxel_float = voxel_matrix.astype(np.float32)
    
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
        
        tensor = np.stack([r_channel, g_channel, b_channel, a_channel], axis=0)
    elif mode == "alpha":
        tensor = voxel_float[np.newaxis, ...]
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'rgba' or 'alpha'")
    
    return torch.from_numpy(tensor).unsqueeze(0).to(device)