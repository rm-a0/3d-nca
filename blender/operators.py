import bpy
import numpy as np

from .client import NCAClient

# Module level client instance
_client : NCAClient | None = None

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
    # TODO: implement mesh -> np array conversion
    # For now, just return a dummy array
    return np.zeros((32, 32, 32, 4), dtype=np.float32)

def visualize_voxel(tensor):
    # TODO: implement Blender visualization logic (voxels, RGBA grids, etc.)
    return

def _on_state(array: np.ndarray, epoch: int):
    print(f"Received state for epoch {epoch}, shape: {array.shape}")

def _on_error(error_msg: str):
    print(f"Error from NCA server: {error_msg}")

def _on_disconnect():
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

classes = (
    NCA_OT_StartTraining,
    NCA_OT_StopTraining,
    NCA_OT_PauseTraining,
    NCA_OT_ResumeTraining,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
