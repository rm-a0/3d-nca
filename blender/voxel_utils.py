"""
Utilities for voxelizing meshes and displaying voxel arrays in Blender.

Voxelization: converts Blender mesh objects to (D,H,W,C) numpy arrays using
raycast-based inside-outside tests. Preserves material IDs and colors.

Display: renders (D,H,W,C) arrays as optimized merged-cube meshes in Blender
using vectorized geometry construction and foreach_set bulk operations for speed.

Note: voxelization core implemented with assistance from Claude Opus 4.6.
"""

import bpy
import numpy as np
import colorsys
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from typing import List, Optional, Tuple


def get_or_create_collection(name: str) -> bpy.types.Collection:
    """Get or create a named Blender collection linked to active scene.

    Args:
        name: Collection name.

    Returns:
        Existing or newly created Collection object.
    """
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def clear_collection(collection: bpy.types.Collection) -> None:
    """Remove all objects from collection and delete orphaned mesh data.

    Args:
        collection: Collection to clear.
    """
    for obj in list(collection.objects):
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data and data.users == 0 and isinstance(data, bpy.types.Mesh):
            bpy.data.meshes.remove(data)


def get_slot_offset(
    slot: int, grid_size: Tuple[int, int, int], cell_size: float
) -> float:
    """Compute X-axis offset for a layout slot.

    Slot 0: source mesh, Slot 1: voxelized target, Slot 2: NCA state.
    Objects are laid out side-by-side with calculated gap.

    Args:
        slot: Slot index (0, 1, or 2).
        grid_size: Voxel grid dimensions (D, H, W).
        cell_size: Size of each voxel.

    Returns:
        X-axis offset in world space.
    """
    extent = max(grid_size) * cell_size
    gap = extent * 0.3
    return slot * (extent + gap)


def place_source_in_scene(
    objs: list,
    grid_size: Tuple[int, int, int],
    cell_size: float,
    collection_name: str = "NCA_Source",
) -> None:
    """Place and scale source meshes to match voxel grid on display.

    Creates linked duplicates of source meshes, grouped under an empty parent,
    positioned at slot 0 and uniformly scaled to fit grid dimensions.

    Args:
        objs: List of source Mesh objects to place.
        grid_size: Voxel grid dimensions (D, H, W).
        cell_size: Size of each voxel.
        collection_name: Collection to place scaled meshes in (default "NCA_Source").
    """
    collection = get_or_create_collection(collection_name)
    clear_collection(collection)

    if not objs:
        return

    D, H, W = grid_size
    slot_extent = max(D, H, W) * cell_size
    slot_x = get_slot_offset(0, grid_size, cell_size)

    # Centre of the voxelised grid in this slot
    grid_center = Vector(
        (
            slot_x + D * cell_size * 0.5,
            H * cell_size * 0.5,
            W * cell_size * 0.5,
        )
    )

    # World-space bounding box of all source meshes
    all_corners = []
    for obj in objs:
        try:
            _ = obj.name  # validate the StructRNA reference is still alive
        except ReferenceError:
            continue
        for corner in obj.bound_box:
            all_corners.append(obj.matrix_world @ Vector(corner))

    if not all_corners:
        return

    bb_min = Vector((min(c[i] for c in all_corners) for i in range(3)))
    bb_max = Vector((max(c[i] for c in all_corners) for i in range(3)))
    src_extent = max(max(bb_max[i] - bb_min[i] for i in range(3)), 1e-8)
    src_center = (bb_min + bb_max) * 0.5

    scale_factor = slot_extent / src_extent

    # Parent empty controls the group transform
    empty = bpy.data.objects.new("NCA_Source_Root", None)
    empty.empty_display_size = 0.01
    empty.empty_display_type = 'PLAIN_AXES'
    collection.objects.link(empty)
    empty.location = grid_center
    empty.scale = (scale_factor, scale_factor, scale_factor)

    for obj in objs:
        try:
            _ = obj.name
        except ReferenceError:
            continue
        dup = obj.copy()  # linked duplicate (shares mesh data)
        collection.objects.link(dup)
        dup.parent = empty
        dup.matrix_parent_inverse = Matrix.Identity(4)
        dup.location = obj.matrix_world.translation - src_center
        dup.rotation_euler = obj.rotation_euler.copy()
        dup.scale = obj.scale.copy()


def mesh_to_voxel_array(
    obj: bpy.types.Object,
    grid_size: Tuple[int, int, int],
    visible_channels: str = "RGBA",
    offset: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List]:
    """Voxelize mesh object to array with material color preservation.

    Uses ray-casting inside-outside test to determine voxel occupancy.
    Mesh is uniformly scaled so longest axis spans (grid_axis - 2*offset) voxels,
    then centered in grid. Extracts Principled BSDF base colors from materials.

    Args:
        obj: Mesh object to voxelize.
        grid_size: Target voxel grid dimensions (D, H, W).
        visible_channels: Output format: "ALPHA" (1 channel), "RGBA" (4 channels), or "ALPHA_MATERIAL_ID" (2 channels).
        offset: Border offset in voxels (default 1).

    Returns:
        Tuple (voxel_data [D,H,W,C], material_index_map [D,H,W], materials list).
    """
    channel_map = {"ALPHA": 1, "RGBA": 4, "ALPHA_MATERIAL_ID": 2}
    n_ch = channel_map.get(visible_channels, 4)
    D, H, W = grid_size
    voxels = np.zeros((D, H, W, n_ch), dtype=np.float32)
    mat_index_map = np.full((D, H, W), -1, dtype=np.int32)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.data
    bvh = BVHTree.FromObject(eval_obj, depsgraph, epsilon=0.0)

    local_corners = [Vector(c) for c in obj.bound_box]
    bb_min = Vector(min(c[i] for c in local_corners) for i in range(3))
    bb_max = Vector(max(c[i] for c in local_corners) for i in range(3))

    mesh_span = bb_max - bb_min
    grid_dims = np.array([D, H, W], dtype=np.float64)
    usable = np.maximum(grid_dims - 2.0 * offset, 1.0)
    mesh_extents = np.maximum(
        np.array([mesh_span.x, mesh_span.y, mesh_span.z], dtype=np.float64), 1e-8
    )

    scale = float(np.min(usable / mesh_extents))
    scaled_extents = mesh_extents * scale
    grid_origin = (grid_dims - scaled_extents) * 0.5

    poly_mat_ids = np.array(
        [p.material_index for p in eval_mesh.polygons], dtype=np.int32
    )
    materials: list = [slot.material for slot in obj.material_slots]
    n_mats = max(len(materials), 1)
    mat_norm = float(max(n_mats - 1, 1))

    for i in range(D):
        tx = bb_min.x + (i - grid_origin[0] + 0.5) / scale
        for j in range(H):
            ty = bb_min.y + (j - grid_origin[1] + 0.5) / scale
            for k in range(W):
                tz = bb_min.z + (k - grid_origin[2] + 0.5) / scale
                point = Vector((tx, ty, tz))

                inside, _ray_face = _is_inside_mesh_with_face(bvh, point)
                if not inside:
                    continue

                nearest, _normal, nearest_face, _dist = bvh.find_nearest(point)
                mi = int(poly_mat_ids[nearest_face]) if nearest_face is not None else 0
                mat_index_map[i, j, k] = mi

                if visible_channels == "ALPHA_MATERIAL_ID":
                    voxels[i, j, k] = [1.0, float(mi) / mat_norm]
                elif n_ch >= 4:
                    color = _get_material_base_color(
                        materials[mi] if mi < len(materials) else None
                    )
                    voxels[i, j, k] = [color[0], color[1], color[2], 1.0]
                else:
                    voxels[i, j, k, 0] = 1.0

    return voxels, mat_index_map, materials


def _get_material_base_color(mat) -> np.ndarray:
    """Extract base color from Blender Principled BSDF material.

    Args:
        mat: Blender material or None.

    Returns:
        RGB color as [R,G,B] float32 array, or [1,1,1] if not found.
    """
    if mat and mat.use_nodes:
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                c = node.inputs["Base Color"].default_value
                return np.array([c[0], c[1], c[2]], dtype=np.float32)
    return np.array([1.0, 1.0, 1.0], dtype=np.float32)


def _is_inside_mesh_with_face(
    bvh: BVHTree, point: Vector
) -> Tuple[bool, Optional[int]]:
    """Determine if point is inside mesh using ray-casting parity test.

    Casts ray from point upward (Z direction) and counts intersections.
    Odd count = inside, even = outside (standard ray-casting test).

    Args:
        bvh: BVH tree from mesh.
        point: Point to test.

    Returns:
        Tuple (is_inside, first_hit_face_index or None).
    """
    direction = Vector((0, 0, 1))
    origin = point.copy()
    count = 0
    first_face: Optional[int] = None
    for _ in range(100):
        hit, _normal, idx, _dist = bvh.ray_cast(origin, direction)
        if hit is None:
            break
        if count == 0:
            first_face = idx
        count += 1
        origin = hit + direction * 1e-4
    return (count % 2 == 1, first_face)


_FACE_DEFS = [
    (0, 1, 2, 3),
    (4, 7, 6, 5),
    (0, 4, 5, 1),
    (2, 6, 7, 3),
    (0, 3, 7, 4),
    (1, 5, 6, 2),
]


def voxel_array_to_blender(
    voxel_data: np.ndarray,
    collection_name: str,
    object_name: str = "NCA_Voxels",
    cell_size: float = 0.1,
    alive_threshold: float = 0.02,
    material_map: Optional[np.ndarray] = None,
    materials: Optional[list] = None,
) -> Optional[bpy.types.Object]:
    """Create Blender mesh object from voxel array display data.

    Converts (D,H,W,C) voxel array to optimized merged-cube mesh using
    vectorized numpy geometry construction. Caps voxel count at 50K to prevent
    GPU crashes. Uses per-loop vertex colors for visualization.

    Args:
        voxel_data: Voxel array [D,H,W,C] with C>=1 (alpha at ...,-1]).
        collection_name: Blender collection to place mesh in.
        object_name: Name for created object (default "NCA_Voxels").
        cell_size: Size of each voxel in Blender units (default 0.1).
        alive_threshold: Alpha threshold for inclusion (default 0.02).
        material_map: Optional Material IDs [D,H,W] to assign per-voxel materials.
        materials: Optional list of Blender materials to assign to faces.

    Returns:
        Created Mesh object, or None if no voxels above threshold.
    """
    collection = get_or_create_collection(collection_name)

    for old in [o for o in collection.objects if o.name == object_name]:
        data = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        if data and data.users == 0 and isinstance(data, bpy.types.Mesh):
            bpy.data.meshes.remove(data)

    MAX_VOXELS = 50_000  # safety cap - prevent GPU driver crashes

    n_ch = voxel_data.shape[-1]
    alpha = voxel_data[..., 3] if n_ch >= 4 else voxel_data[..., 0]
    occupied = np.argwhere(alpha > alive_threshold)

    if len(occupied) == 0:
        return None

    # If too many voxels, keep only the highest-alpha ones
    if len(occupied) > MAX_VOXELS:
        alpha_vals = alpha[occupied[:, 0], occupied[:, 1], occupied[:, 2]]
        top_indices = np.argpartition(alpha_vals, -MAX_VOXELS)[-MAX_VOXELS:]
        occupied = occupied[top_indices]

    N = len(occupied)

    use_source_mats = (
        material_map is not None and materials is not None and len(materials) > 0
    )
    use_vcol = not use_source_mats and (n_ch >= 4 or n_ch == 2)

    s = cell_size * 0.5

    # --- Numpy-Vectorized Geometry ---
    # Template cube: 8 vertices relative to centre
    cube_verts = np.array(
        [
            [-s, -s, -s],
            [+s, -s, -s],
            [+s, +s, -s],
            [-s, +s, -s],
            [-s, -s, +s],
            [+s, -s, +s],
            [+s, +s, +s],
            [-s, +s, +s],
        ],
        dtype=np.float32,
    )

    # Template faces (6 quads, winding order gives outward normals)
    cube_faces = np.array(_FACE_DEFS, dtype=np.int32)

    # Centres for every occupied voxel  (N, 3)
    centres = occupied.astype(np.float32) * cell_size

    # Broadcast: (1,8,3) + (N,1,3) -> (N,8,3) -> (N*8, 3)
    all_verts = (cube_verts[np.newaxis, :, :] + centres[:, np.newaxis, :]).reshape(
        -1, 3
    )

    # Per-voxel vertex offset applied to face indices: (N,6,4) -> (N*6, 4)
    offsets = (np.arange(N, dtype=np.int32) * 8)[:, np.newaxis, np.newaxis]
    all_faces = (cube_faces[np.newaxis, :, :] + offsets).reshape(-1, 4)

    n_verts = N * 8
    n_faces = N * 6
    n_loops = n_faces * 4

    # Build Blender mesh via foreach_set (bulk C-level copy)
    mesh_data = bpy.data.meshes.new(f"{object_name}_mesh")

    mesh_data.vertices.add(n_verts)
    mesh_data.vertices.foreach_set("co", all_verts.ravel())

    mesh_data.loops.add(n_loops)
    mesh_data.loops.foreach_set("vertex_index", all_faces.ravel())

    mesh_data.polygons.add(n_faces)
    mesh_data.polygons.foreach_set("loop_start", np.arange(n_faces, dtype=np.int32) * 4)
    mesh_data.polygons.foreach_set("loop_total", np.full(n_faces, 4, dtype=np.int32))

    # Material index per polygon (6 faces share the same material per voxel)
    if use_source_mats:
        mat_ids = material_map[occupied[:, 0], occupied[:, 1], occupied[:, 2]]
        mat_ids = np.clip(mat_ids, 0, len(materials) - 1).astype(np.int32)
        mesh_data.polygons.foreach_set("material_index", np.repeat(mat_ids, 6))

    mesh_data.update()
    mesh_data.validate()

    # Per-loop (face-corner) vertex colours
    if use_vcol:
        color_attr = mesh_data.color_attributes.new("Col", 'FLOAT_COLOR', 'CORNER')
        if n_ch >= 4:
            colors = voxel_data[occupied[:, 0], occupied[:, 1], occupied[:, 2]].astype(
                np.float32
            )  # (N, 4)
        elif n_ch == 2:
            mat_vals = voxel_data[occupied[:, 0], occupied[:, 1], occupied[:, 2], 1]
            colors = np.array(
                [_matid_to_color(float(v)) for v in mat_vals], dtype=np.float32
            )  # (N, 4)
        # 6 faces x 4 corners = 24 identical color entries per voxel
        loop_colors = np.repeat(colors, 24, axis=0).astype(np.float32)
        color_attr.data.foreach_set("color", loop_colors.ravel())

    obj = bpy.data.objects.new(object_name, mesh_data)
    collection.objects.link(obj)

    if use_source_mats:
        for mat in materials:
            obj.data.materials.append(mat)
    elif use_vcol:
        _assign_vertex_color_material(obj, f"{object_name}_mat", "Col")

    return obj


def _matid_to_color(mat_val: float) -> Tuple[float, float, float, float]:
    """Convert normalized material ID to distinct RGBA color via HSV.

    Args:
        mat_val: Normalized material value in [0,1].

    Returns:
        RGBA tuple (r, g, b, a) with a=1.0, colors via HSV hue rotation.
    """
    r, g, b = colorsys.hsv_to_rgb(mat_val, 0.9, 0.9)
    return (r, g, b, 1.0)


def _assign_vertex_color_material(
    obj: bpy.types.Object, mat_name: str, attr_name: str
) -> None:
    """Create and assign material that reads color from geometry attribute.

    Args:
        obj: Mesh object to assign material to.
        mat_name: Material name to create/reuse.
        attr_name: Geometry attribute name for color (e.g., "Col").
    """
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()

    output = tree.nodes.new("ShaderNodeOutputMaterial")
    output.location = (300, 0)
    bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    attr = tree.nodes.new("ShaderNodeAttribute")
    attr.location = (-300, 0)
    attr.attribute_name = attr_name
    attr.attribute_type = "GEOMETRY"

    tree.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)


def server_state_to_voxel_array(
    raw: np.ndarray, visible_channels: int = 4
) -> np.ndarray:
    """Convert server state tensor to display-ready voxel array.

    Extracts visible channels from server tensor, transposes from internal
    (B,C,D,H,W) format to external (D,H,W,C) format, and clamps to [0,1]
    for correct rendering (NCA outputs may be in [-1,1] range).

    Args:
        raw: State tensor [B,C,D,H,W] from server or [C,D,H,W] (batch=1).
        visible_channels: Number of visible channels to extract (default 4).

    Returns:
        Voxel array [D,H,W,visible_channels] clipped to [0,1].
    """
    if raw.ndim == 5:
        raw = raw[0]
    vis = raw[-visible_channels:]
    arr = np.transpose(vis, (1, 2, 3, 0)).astype(np.float32)
    np.clip(arr, 0.0, 1.0, out=arr)
    return arr
