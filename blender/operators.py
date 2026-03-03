import bpy
import numpy as np
import threading
from typing import List, Tuple

from .client import NCAClient

from .voxel_utils import (
    mesh_to_voxel_array,
    voxel_array_to_blender,
    server_state_to_voxel_array,
    clear_collection,
    get_slot_offset,
    place_source_in_scene,
)

_client: NCAClient | None = None
_target_array: np.ndarray | None = None

_pending_state: np.ndarray | None = None
_pending_lock = threading.Lock()
_timer_running = False

SOURCE_COLLECTION = "NCA_Source"
TARGET_COLLECTION = "NCA_Target"
STATE_COLLECTION  = "NCA_State"

VISIBLE_CHANNELS_MAP = {
    'ALPHA': 1,
    'RGBA': 4,
    'ALPHA_MATERIAL_ID': 2,
}

def is_training_active() -> bool:
    """Return True when a training session is connected (running or paused).

    Used by panels to gray-out settings that must not change mid-training.
    """
    return _client is not None and _client.connected

def setup_configs(context) -> dict:
    """Build the full config dict from Blender scene properties."""
    cell_props = context.scene.nca_cell_props
    perception_props = context.scene.nca_perception_props
    update_props = context.scene.nca_update_props
    grid_props = context.scene.nca_grid_props

    return {
        "cell": {
            "hidden_channels": int(cell_props.hidden_channels),
            "visible_channels": VISIBLE_CHANNELS_MAP.get(cell_props.visible_channels, 4),
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

def get_target_array(context) -> np.ndarray:
    """Return the cached target array, or a zero array matching grid_size."""
    if _target_array is not None:
        return _target_array
    grid_props = context.scene.nca_grid_props
    size = tuple(int(x) for x in grid_props.grid_size)
    vis = VISIBLE_CHANNELS_MAP.get(
        context.scene.nca_cell_props.visible_channels, 4
    )
    return np.zeros((*size, vis), dtype=np.float32)

def voxelize_and_display(
    context, objs: List[bpy.types.Object]
) -> Tuple[bool, str]:
    """Voxelize *objs*, display in viewport, update target properties.

    Returns (success: bool, message: str).
    """
    global _target_array

    cell_props = context.scene.nca_cell_props
    grid_props = context.scene.nca_grid_props
    target_props = context.scene.nca_target_props

    if not objs:
        return False, "No mesh objects provided"

    grid_size = tuple(int(x) for x in grid_props.grid_size)
    vis_ch = cell_props.visible_channels
    grid_offset = int(grid_props.grid_offset)

    results = [
        mesh_to_voxel_array(obj, grid_size, vis_ch, offset=grid_offset)
        for obj in objs
    ]

    combined_data = results[0][0].copy()
    combined_mat_map = results[0][1].copy()
    all_materials = list(results[0][2])

    for data, mat_map, mats in results[1:]:
        mat_offset = len(all_materials)
        all_materials.extend(mats)
        new_occupied = mat_map >= 0
        combined_mat_map[new_occupied] = mat_map[new_occupied] + mat_offset
        combined_data = np.maximum(combined_data, data)

    _target_array = combined_data

    place_source_in_scene(objs, grid_size, target_props.cell_size)

    vox_obj = voxel_array_to_blender(
        combined_data,
        collection_name=TARGET_COLLECTION,
        object_name="NCA_Target",
        cell_size=target_props.cell_size,
        alive_threshold=cell_props.alive_threshold,
        material_map=combined_mat_map,
        materials=all_materials,
    )
    if vox_obj:
        vox_obj.location.x = get_slot_offset(1, grid_size, target_props.cell_size)

    n_ch = combined_data.shape[-1]
    alpha = combined_data[..., 3] if n_ch >= 4 else combined_data[..., 0]
    n_alive = int((alpha > cell_props.alive_threshold).sum())
    names = ", ".join(o.name for o in objs)

    target_props.voxel_count = n_alive
    target_props.is_voxelized = True

    return True, f"Voxelized [{names}] → {n_alive} voxels"

def clear_target(context) -> None:
    """Remove target/source collections and reset target state."""
    global _target_array

    if TARGET_COLLECTION in bpy.data.collections:
        clear_collection(bpy.data.collections[TARGET_COLLECTION])
    if SOURCE_COLLECTION in bpy.data.collections:
        clear_collection(bpy.data.collections[SOURCE_COLLECTION])

    _target_array = None

    target_props = context.scene.nca_target_props
    target_props.voxel_count = 0
    target_props.is_voxelized = False

def visualize_voxel(tensor: np.ndarray) -> None:
    """Display a voxel array in the viewport (called for server state updates)."""
    target_props = bpy.context.scene.nca_target_props
    cell = bpy.context.scene.nca_cell_props
    grid = bpy.context.scene.nca_grid_props
    obj = voxel_array_to_blender(
        tensor,
        collection_name=STATE_COLLECTION,
        object_name="NCA_State",
        cell_size=target_props.cell_size,
        alive_threshold=cell.alive_threshold,
    )
    if obj:
        grid_size = tuple(int(x) for x in grid.grid_size)
        obj.location.x = get_slot_offset(2, grid_size, target_props.cell_size)

def _on_state(array: np.ndarray) -> None:
    """Listener callback — runs on a background thread."""
    global _pending_state, _timer_running
    with _pending_lock:
        _pending_state = array
        if not _timer_running:
            _timer_running = True
            bpy.app.timers.register(_poll_state, first_interval=0.05)

def _poll_state() -> float | None:
    """Recurring timer on Blender's main thread — drains the pending buffer."""
    global _pending_state, _timer_running

    with _pending_lock:
        arr = _pending_state
        _pending_state = None

    if arr is None:
        if _client is None or not _client.connected:
            _timer_running = False
            return None
        return 0.1

    try:
        vis_ch = VISIBLE_CHANNELS_MAP.get(
            bpy.context.scene.nca_cell_props.visible_channels, 4
        )
        display_arr = server_state_to_voxel_array(arr, visible_channels=vis_ch)
        visualize_voxel(display_arr)
    except Exception as e:
        print(f"Error updating state visualization: {e}")

    if _client is not None and _client.connected:
        return 0.1
    _timer_running = False
    return None

def _on_error(error_msg: str) -> None:
    """Handle server error messages."""
    print(f"Error from NCA server: {error_msg}")

def _on_disconnect() -> None:
    """Handle server disconnection."""
    print("Disconnected from NCA server")

class NCA_OT_StartTraining(bpy.types.Operator):
    bl_idname = "nca.start_training"
    bl_label = "Start Training"
    bl_description = "Connect to NCA server and start training"

    @classmethod
    def poll(cls, context):
        return (_client is None or not _client.connected) and _target_array is not None

    def execute(self, context):
        global _client

        if _client and _client.connected:
            self.report({'WARNING'}, "Already connected to NCA server")
            return {'CANCELLED'}

        config = setup_configs(context)
        target = get_target_array(context)

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

    @classmethod
    def poll(cls, context):
        return _client is not None and _client.connected

    def execute(self, context):
        global _client, _timer_running

        if _client and _client.connected:
            try:
                _client.send_stop()
            except Exception:
                pass
            _client.disconnect()
            _client = None
            _timer_running = False
            self.report({'INFO'}, "Stopped training and disconnected from NCA server")
        else:
            self.report({'WARNING'}, "Not connected to NCA server")
        return {'FINISHED'}   

class NCA_OT_PauseTraining(bpy.types.Operator):
    bl_idname = "nca.pause_training"
    bl_label = "Pause Training"
    bl_description = "Pause training on NCA server"

    @classmethod
    def poll(cls, context):
        return _client is not None and _client.connected

    def execute(self, context):
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

    @classmethod
    def poll(cls, context):
        return _client is not None and _client.connected

    def execute(self, context):
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
        objs = [o for o in context.selected_objects if o.type == 'MESH']
        if not objs:
            self.report({'ERROR'}, "No mesh objects selected in the viewport")
            return {'CANCELLED'}

        success, message = voxelize_and_display(context, objs)
        if success:
            self.report({'INFO'}, message)
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, message)
            return {'CANCELLED'}

class NCA_OT_ClearTargetVoxels(bpy.types.Operator):
    bl_idname = "nca.clear_target_voxels"
    bl_label = "Clear"
    bl_description = "Remove the voxelized target from the viewport"

    def execute(self, context):
        clear_target(context)

        target_props = context.scene.nca_target_props
        target_props["source_object"] = None

        self.report({'INFO'}, "Cleared target voxels")
        return {'FINISHED'}

class NCA_OT_AddScheduleEvent(bpy.types.Operator):
    bl_idname = "nca.add_schedule_event"
    bl_label = "Add Event"
    bl_description = "Add a new event to the training schedule"

    def execute(self, context):
        sched = context.scene.nca_schedule_props
        item = sched.events.add()
        item.epoch = -1
        item.event_type = 'LEARNING_RATE'
        item.value = 0.01
        sched.active_event_index = len(sched.events) - 1
        return {'FINISHED'}

class NCA_OT_RemoveScheduleEvent(bpy.types.Operator):
    bl_idname = "nca.remove_schedule_event"
    bl_label = "Remove Event"
    bl_description = "Remove the selected event from the training schedule"

    @classmethod
    def poll(cls, context):
        sched = context.scene.nca_schedule_props
        return len(sched.events) > 0

    def execute(self, context):
        sched = context.scene.nca_schedule_props
        idx = sched.active_event_index
        sched.events.remove(idx)
        sched.active_event_index = max(0, idx - 1)
        return {'FINISHED'}

class NCA_OT_SendSchedule(bpy.types.Operator):
    bl_idname = "nca.send_schedule"
    bl_label = "Send Schedule"
    bl_description = "Send the current schedule to the training server"

    @classmethod
    def poll(cls, context):
        return is_training_active()

    def execute(self, context):
        sched = context.scene.nca_schedule_props
        events = []
        for ev in sched.events:
            events.append({
                "epoch": ev.epoch,
                "event_type": ev.event_type,
                "value": ev.value,
            })
        try:
            _client.send_schedule(events)
            self.report({'INFO'}, f"Sent schedule with {len(events)} event(s)")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to send schedule: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}

classes = (
    NCA_OT_StartTraining,
    NCA_OT_StopTraining,
    NCA_OT_PauseTraining,
    NCA_OT_ResumeTraining,
    NCA_OT_VoxelizeTarget,
    NCA_OT_ClearTargetVoxels,
    NCA_OT_AddScheduleEvent,
    NCA_OT_RemoveScheduleEvent,
    NCA_OT_SendSchedule,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
