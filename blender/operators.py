import bpy
import numpy as np

from .client import NCAClient

from .voxel_utils import (
    mesh_to_voxel_array,
    voxel_array_to_blender,
    server_state_to_voxel_array,
    clear_collection,
)

_client : NCAClient | None = None
_target_array: np.ndarray | None = None

TARGET_COLLECTION = "NCA_Target"
STATE_COLLECTION  = "NCA_State"

def setup_configs(context):
    cell_props = context.scene.nca_cell_props
    perception_props = context.scene.nca_perception_props
    update_props = context.scene.nca_update_props
    grid_props = context.scene.nca_grid_props

    visible_channels_map = {
        'ALPHA': 1,
        'RGBA': 4,
        'ALPHA_MATERIAL_ID': 2,
    }

    cfg = {
        "cell": {
            "hidden_channels": int(cell_props.hidden_channels),
            "visible_channels": visible_channels_map.get(cell_props.visible_channels, 4),
            "alive_threshold": float(cell_props.alive_threshold),
        },
        "perception": {
            "kernel_radius": int(perception_props.kernel_radius),
            "channel_groups": int(perception_props.channel_groups),
        },
        "update": {
            "hidden_dim": int(update_props.hidden_dim),
            "stochastic_update": bool(update_props.stochastic_update),
            "fire_rate": float(update_props.fire_rate),
        },
        "grid": {
            "size": tuple(int(x) for x in grid_props.grid_size),
        },
        "training": {
            "learning_rate": float(context.scene.nca_training_props.learning_rate),
            "num_epochs": int(context.scene.nca_training_props.num_epochs),
            "batch_size": int(context.scene.nca_training_props.batch_size),
        },
    }
    return cfg

def selected_meshes_to_voxel_array():
    """Return the cached target array."""
    global _target_array
    if _target_array is not None:
        return _target_array
    return np.zeros((32, 32, 32, 4), dtype=np.float32)

def visualize_voxel(tensor):
    """Display a voxel array in the viewport (called for server state updates)."""
    vis = bpy.context.scene.nca_visualization_props
    cell = bpy.context.scene.nca_cell_props
    voxel_array_to_blender(
        tensor,
        collection_name=STATE_COLLECTION,
        object_name="NCA_State",
        cell_size=vis.cell_size,
        alive_threshold=cell.alive_threshold,
    )

def _on_state(array: np.ndarray):
    """Listener callback - runs on a background thread."""
    vis_channels_map = {'ALPHA': 1, 'RGBA': 4, 'ALPHA_MATERIAL_ID': 2}
    vis_ch = vis_channels_map.get(
        bpy.context.scene.nca_cell_props.visible_channels, 4
    )
    display_arr = server_state_to_voxel_array(array, visible_channels=vis_ch)
    bpy.app.timers.register(lambda: _update_state_display(display_arr))

def _update_state_display(arr: np.ndarray):
    """Timer callback on Blender's main thread."""
    try:
        visualize_voxel(arr)
    except Exception as e:
        print(f"Error updating state visualization: {e}")
    return None

def _on_error(error_msg: str):
    """Handle server error messages."""
    print(f"Error from NCA server: {error_msg}")

def _on_disconnect():
    """Handle server disconnection."""
    print("Disconnected from NCA server")

class NCA_OT_StartTraining(bpy.types.Operator):
    bl_idname = "nca.start_training"
    bl_label = "Start Training"
    bl_description = "Connect to NCA server and start training"

    def execute(self, context):
        global _client

        if _client and _client.connected:
            self.report({'WARNING'}, "Already connected to NCA server")
            return {'CANCELLED'}

        config = setup_configs(context)
        target = selected_meshes_to_voxel_array()

        _client = NCAClient(host="127.0.0.1", port=5555)
        try:
            _client.connect()
        except Exception as e:
            self.report({'ERROR'}, f"Failed to connect to NCA server: {e}")
            _client = None
            return {'CANCELLED'}
        
        _client.send_init(config, target)
        _client.start_listener(
            on_state=_on_state, 
            on_error=_on_error, 
            on_disconnect=_on_disconnect,
        )

        self.report({'INFO'}, "Started training on NCA server")
        return {'FINISHED'}

class NCA_OT_StopTraining(bpy.types.Operator):
    bl_idname = "nca.stop_training"
    bl_label = "Stop Training"
    bl_description = "Stop training and disconnect from NCA server"

    def execute(self, context):
        global _client

        if _client and _client.connected:
            try:
                _client.send_stop()
            except Exception:
                pass
            _client.disconnect()
            _client = None
            self.report({'INFO'}, "Stopped training and disconnected from NCA server")
        else:
            self.report({'WARNING'}, "Not connected to NCA server")
        return {'FINISHED'}   

class NCA_OT_PauseTraining(bpy.types.Operator):
    bl_idname = "nca.pause_training"
    bl_label = "Pause Training"
    bl_description = "Pause training on NCA server"

    def execute(self, context):
        global _client

        if _client and _client.connected:
            _client.send_pause()
            self.report({'INFO'}, "Paused training on NCA server")
        else:
            self.report({'WARNING'}, "Not connected to NCA server")
        return {'FINISHED'}

class NCA_OT_ResumeTraining(bpy.types.Operator):
    bl_idname = "nca.resume_training"
    bl_label = "Resume Training"
    bl_description = "Resume training on NCA server"

    def execute(self, context):
        global _client

        if _client and _client.connected:
            _client.send_resume()
            self.report({'INFO'}, "Resumed training on NCA server")
        else:
            self.report({'WARNING'}, "Not connected to NCA server")
        return {'FINISHED'}

class NCA_OT_VoxelizeTarget(bpy.types.Operator):
    bl_idname = "nca.voxelize_target"
    bl_label = "Voxelize"
    bl_description = "Voxelize the selected viewport mesh into a voxel grid and display it"

    def execute(self, context):
        global _target_array

        cell_props = context.scene.nca_cell_props
        grid_props = context.scene.nca_grid_props
        vis_props = context.scene.nca_visualization_props

        objs = [o for o in context.selected_objects if o.type == 'MESH']
        if not objs:
            self.report({'ERROR'}, "No mesh objects selected in the viewport")
            return {'CANCELLED'}

        grid_size = tuple(int(x) for x in grid_props.grid_size)
        vis_ch = cell_props.visible_channels

        offset = int(grid_props.grid_offset)
        results = [mesh_to_voxel_array(obj, grid_size, vis_ch, offset=offset) for obj in objs]

        combined_data = results[0][0].copy()
        combined_mat_map = results[0][1].copy()
        all_materials = list(results[0][2])

        for data, mat_map, mats in results[1:]:
            offset = len(all_materials)
            all_materials.extend(mats)
            new_occupied = mat_map >= 0
            combined_mat_map[new_occupied] = mat_map[new_occupied] + offset
            combined_data = np.maximum(combined_data, data)

        _target_array = combined_data

        voxel_array_to_blender(
            combined_data,
            collection_name=TARGET_COLLECTION,
            object_name="NCA_Target",
            cell_size=vis_props.cell_size,
            alive_threshold=cell_props.alive_threshold,
            material_map=combined_mat_map,
            materials=all_materials,
        )

        n_ch = combined_data.shape[-1]
        alpha = combined_data[..., 3] if n_ch >= 4 else combined_data[..., 0]
        n_alive = int((alpha > cell_props.alive_threshold).sum())
        names = ", ".join(o.name for o in objs)
        self.report({'INFO'}, f"Voxelized [{names}] → {n_alive} voxels")
        return {'FINISHED'}

class NCA_OT_ClearTargetVoxels(bpy.types.Operator):
    bl_idname = "nca.clear_target_voxels"
    bl_label = "Clear"
    bl_description = "Remove the voxelized target from the viewport"

    def execute(self, context):
        global _target_array

        if TARGET_COLLECTION in bpy.data.collections:
            clear_collection(bpy.data.collections[TARGET_COLLECTION])

        _target_array = None
        self.report({'INFO'}, "Cleared target voxels")
        return {'FINISHED'}

classes = (
    NCA_OT_StartTraining,
    NCA_OT_StopTraining,
    NCA_OT_PauseTraining,
    NCA_OT_ResumeTraining,
    NCA_OT_VoxelizeTarget,
    NCA_OT_ClearTargetVoxels,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
