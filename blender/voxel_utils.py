"""
Utilities for voxelizing meshes and displaying voxel arrays in Blender.

Voxelization: converts Blender mesh objects to (D,H,W,C) numpy arrays using
raycast-based inside-outside tests. Preserves material IDs and colors.

Display: renders (D,H,W,C) arrays as optimized merged-cube meshes in Blender
using vectorized geometry construction and foreach_set bulk operations for speed.

Color modes:
  - BSDF: reads Principled BSDF Base Color (fast, solid colors only)
  - TEXTURE: samples image texture at nearest surface UV (slower, works with
    any image-textured mesh including imports)

Note: voxelization core implemented with assistance from Claude Opus 4.6.
"""

import bpy
import numpy as np
import colorsys
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Color / material helpers
# ---------------------------------------------------------------------------

def _get_material_base_color(mat) -> np.ndarray:
    """Extract base color from Blender Principled BSDF material.

    Args:
        mat: Blender material or None.

    Returns:
        RGB color as float32 array [R,G,B], or [1,1,1] if not found.
    """
    if mat and mat.use_nodes:
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                c = node.inputs["Base Color"].default_value
                return np.array([c[0], c[1], c[2]], dtype=np.float32)
    return np.array([1.0, 1.0, 1.0], dtype=np.float32)


def _get_material_image(mat) -> Optional[bpy.types.Image]:
    """Find the primary image texture in a material's node tree.

    Search order:
    1. Image Texture node wired into Principled BSDF 'Base Color'
    2. Any Image Texture node that has an image loaded

    Args:
        mat: Blender material or None.

    Returns:
        The first matching Image datablock, or None.
    """
    if mat is None or not mat.use_nodes:
        return None

    tree = mat.node_tree

    # Prefer the node directly feeding BSDF Base Color
    for node in tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            inp = node.inputs.get("Base Color")
            if inp and inp.is_linked:
                from_node = inp.links[0].from_node
                if from_node.type == "TEX_IMAGE" and from_node.image:
                    return from_node.image

    # Fallback: any loaded image texture
    for node in tree.nodes:
        if node.type == "TEX_IMAGE" and node.image:
            return node.image

    return None


def _build_image_cache(materials: list) -> Dict[int, Tuple[np.ndarray, int, int]]:
    """Pre-load all material images into numpy pixel arrays.

    Loading image pixels into numpy once avoids per-voxel Python overhead when
    doing many texture lookups during voxelization.

    Args:
        materials: List of Blender material objects (may contain None entries).

    Returns:
        Dict mapping material index -> (pixels[H,W,4], img_width, img_height).
        Only materials that have an image texture are included.
    """
    cache: Dict[int, Tuple[np.ndarray, int, int]] = {}
    for idx, mat in enumerate(materials):
        img = _get_material_image(mat)
        if img is None:
            continue
        w, h = img.size
        if w == 0 or h == 0:
            continue
        # img.pixels is a flat RGBA sequence; copy once into numpy
        pixels = np.array(img.pixels, dtype=np.float32).reshape(h, w, 4)
        cache[idx] = (pixels, w, h)
    return cache


def _barycentric_coords(
    p: Vector, a: Vector, b: Vector, c: Vector
) -> Tuple[float, float, float]:
    """Compute barycentric coordinates (wa, wb, wc) for point p in triangle abc.

    Uses the Möller dot-product method, which is numerically stable and fast.

    Args:
        p: Query point (should be on or very close to the triangle plane).
        a, b, c: Triangle vertices.

    Returns:
        (wa, wb, wc) barycentric weights summing to ~1.  Falls back to
        centroid weights (1/3, 1/3, 1/3) for degenerate triangles.
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10:
        return 1 / 3, 1 / 3, 1 / 3

    wb = (d11 * d20 - d01 * d21) / denom
    wc = (d00 * d21 - d01 * d20) / denom
    wa = 1.0 - wb - wc
    return wa, wb, wc


def _sample_texture_color(
    eval_mesh,
    poly_idx: int,
    surface_point: Vector,
    materials: list,
    img_cache: Dict[int, Tuple[np.ndarray, int, int]],
) -> np.ndarray:
    """Sample texture color at a surface point using UV interpolation.

    Algorithm:
    1. Determine the material index of the hit polygon.
    2. Look up the pre-cached image pixel array for that material.
    3. Fan-triangulate the polygon and find which triangle contains
       the surface point using barycentric coordinates.
    4. Interpolate UV from vertex UVs, then sample the image with wrap.
    5. Fall back to BSDF base color or white when any step is unavailable.

    Args:
        eval_mesh: Evaluated (post-modifier) mesh data.
        poly_idx: Index of the hit polygon.
        surface_point: Exact surface hit position in object local space
                       (as returned by BVHTree.find_nearest).
        materials: List of Blender materials for the object.
        img_cache: Pre-built image pixel cache from _build_image_cache().

    Returns:
        RGB color as float32 array [R,G,B] in linear color space.
    """
    poly = eval_mesh.polygons[poly_idx]
    mat_idx = poly.material_index

    # No cached image for this material -> fall back to BSDF color
    if mat_idx not in img_cache:
        mat = materials[mat_idx] if mat_idx < len(materials) else None
        return _get_material_base_color(mat)

    pixels, img_w, img_h = img_cache[mat_idx]

    # Need a UV layer to do the lookup
    uv_layer = eval_mesh.uv_layers.active
    if uv_layer is None:
        mat = materials[mat_idx] if mat_idx < len(materials) else None
        return _get_material_base_color(mat)

    loops = list(poly.loop_indices)
    n_loops = len(loops)
    if n_loops < 3:
        return np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # Gather per-loop vertex positions and UVs
    vert_cos = [
        Vector(eval_mesh.vertices[eval_mesh.loops[li].vertex_index].co)
        for li in loops
    ]
    vert_uvs = [uv_layer.data[li].uv for li in loops]  # mathutils.Vector(2D)

    # Fan triangulation from vertex 0; find the triangle that best contains p
    best_u, best_v = vert_uvs[0].x, vert_uvs[0].y  # sensible default
    best_match = False

    for tri_i in range(1, n_loops - 1):
        v0, v1, v2 = vert_cos[0], vert_cos[tri_i], vert_cos[tri_i + 1]
        uv0, uv1, uv2 = vert_uvs[0], vert_uvs[tri_i], vert_uvs[tri_i + 1]

        wa, wb, wc = _barycentric_coords(surface_point, v0, v1, v2)

        # Loose tolerance handles floating-point imprecision near edges
        if -0.05 <= wa <= 1.05 and -0.05 <= wb <= 1.05 and -0.05 <= wc <= 1.05:
            best_u = wa * uv0.x + wb * uv1.x + wc * uv2.x
            best_v = wa * uv0.y + wb * uv1.y + wc * uv2.y
            best_match = True
            break

    if not best_match:
        # Point didn't land cleanly in any triangle (edge/precision issue);
        # use the UV of the nearest vertex as a reasonable fallback.
        best_u = vert_uvs[0].x
        best_v = vert_uvs[0].y

    # Tile/wrap UV, then nearest-pixel sample (bilinear isn't worth it at
    # voxel resolution since each voxel maps to many texels anyway)
    u = best_u % 1.0
    v = best_v % 1.0
    px = int(u * img_w) % img_w
    py = int(v * img_h) % img_h

    return pixels[py, px, :3].copy()


# ---------------------------------------------------------------------------
# Core voxelizer
# ---------------------------------------------------------------------------

def mesh_to_voxel_array(
    obj: bpy.types.Object,
    grid_size: Tuple[int, int, int],
    visible_channels: str = "RGBA",
    offset: int = 1,
    color_mode: str = "BSDF",
) -> Tuple[np.ndarray, np.ndarray, List]:
    """Voxelize mesh object to array with material color preservation.

    Uses ray-casting inside-outside test to determine voxel occupancy.
    Mesh is uniformly scaled so longest axis spans (grid_axis - 2*offset) voxels,
    then centered in grid.

    Color modes (only relevant when visible_channels == 'RGBA'):
        'BSDF'    – reads the Base Color socket of a Principled BSDF node.
                    Fast, works for simple solid-color materials.
        'TEXTURE' – UV-samples image textures at the nearest surface point.
                    Slower first-time (image cache build), but handles any
                    imported mesh that uses image-texture materials.

    Args:
        obj: Mesh object to voxelize.
        grid_size: Target voxel grid dimensions (D, H, W).
        visible_channels: Output format: "ALPHA" (1ch), "RGBA" (4ch), or
                          "ALPHA_MATERIAL_ID" (2ch).
        offset: Border offset in voxels (default 1).
        color_mode: 'BSDF' or 'TEXTURE' (see above).

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

    # --- Texture mode: pre-load all images into numpy arrays once ---
    use_texture_sampling = (color_mode == "TEXTURE") and (n_ch >= 4)
    img_cache: Dict[int, Tuple[np.ndarray, int, int]] = {}
    if use_texture_sampling:
        img_cache = _build_image_cache(materials)

    # --- Progress bar (shown in Blender status bar) ---
    wm = bpy.context.window_manager
    wm.progress_begin(0, D)

    try:
        for i in range(D):
            wm.progress_update(i)
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
                    mi = (
                        int(poly_mat_ids[nearest_face])
                        if nearest_face is not None
                        else 0
                    )
                    mat_index_map[i, j, k] = mi

                    if visible_channels == "ALPHA_MATERIAL_ID":
                        voxels[i, j, k] = [1.0, float(mi) / mat_norm]

                    elif n_ch >= 4:
                        if (
                            use_texture_sampling
                            and nearest is not None
                            and nearest_face is not None
                        ):
                            color = _sample_texture_color(
                                eval_mesh,
                                nearest_face,
                                nearest,
                                materials,
                                img_cache,
                            )
                        else:
                            color = _get_material_base_color(
                                materials[mi] if mi < len(materials) else None
                            )
                        voxels[i, j, k] = [color[0], color[1], color[2], 1.0]

                    else:
                        voxels[i, j, k, 0] = 1.0

    finally:
        wm.progress_end()

    return voxels, mat_index_map, materials


# ---------------------------------------------------------------------------
# Ray-cast inside/outside test
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Blender mesh display
# ---------------------------------------------------------------------------

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

    cube_faces = np.array(_FACE_DEFS, dtype=np.int32)

    centres = occupied.astype(np.float32) * cell_size

    all_verts = (cube_verts[np.newaxis, :, :] + centres[:, np.newaxis, :]).reshape(
        -1, 3
    )

    offsets = (np.arange(N, dtype=np.int32) * 8)[:, np.newaxis, np.newaxis]
    all_faces = (cube_faces[np.newaxis, :, :] + offsets).reshape(-1, 4)

    n_verts = N * 8
    n_faces = N * 6
    n_loops = n_faces * 4

    mesh_data = bpy.data.meshes.new(f"{object_name}_mesh")

    mesh_data.vertices.add(n_verts)
    mesh_data.vertices.foreach_set("co", all_verts.ravel())

    mesh_data.loops.add(n_loops)
    mesh_data.loops.foreach_set("vertex_index", all_faces.ravel())

    mesh_data.polygons.add(n_faces)
    mesh_data.polygons.foreach_set("loop_start", np.arange(n_faces, dtype=np.int32) * 4)
    mesh_data.polygons.foreach_set("loop_total", np.full(n_faces, 4, dtype=np.int32))

    if use_source_mats:
        mat_ids = material_map[occupied[:, 0], occupied[:, 1], occupied[:, 2]]
        mat_ids = np.clip(mat_ids, 0, len(materials) - 1).astype(np.int32)
        mesh_data.polygons.foreach_set("material_index", np.repeat(mat_ids, 6))

    mesh_data.update()
    mesh_data.validate()

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


# ---------------------------------------------------------------------------
# Server state conversion
# ---------------------------------------------------------------------------

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