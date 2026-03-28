"""
NCA Blender Operators.

Defines Blender operators and runtime callbacks for remote training, inference,
target voxelization, schedule updates, and state visualization.

Connection flow:
    Disconnected -> Connected (idle) via connect.
    Connected (idle) -> Connected (training) via start training.
    Connected (idle) -> Connected (inference) via run model.
    Connected (training or inference) -> Connected (idle) via stop.
    Connected -> Disconnected via disconnect.
"""

from __future__ import annotations

import os
import threading
from typing import List, Optional, Tuple

import bpy
import numpy as np

from .client import NCAClient
from .voxel_utils import (
    mesh_to_voxel_array,
    voxel_array_to_blender,
    server_state_to_voxel_array,
    clear_collection,
    get_slot_offset,
    place_source_in_scene,
)

# --- Global State ---

_client: Optional[NCAClient] = None
_server_mode: str = "idle"  # 'idle' | 'training' | 'inference'
_target_array: Optional[np.ndarray] = None

_pending_state: Optional[np.ndarray] = None
_pending_lock = threading.Lock()
_timer_running = False

SOURCE_COLLECTION = "NCA_Source"
TARGET_COLLECTION = "NCA_Target"
STATE_COLLECTION = "NCA_State"

VISIBLE_CHANNELS_MAP = {
    "ALPHA": 1,
    "RGBA": 4,
    "ALPHA_MATERIAL_ID": 2,
}

# --- State Queries ---


def is_connected() -> bool:
    """Return True if client socket exists and reports connected state."""
    return _client is not None and _client.connected


def is_server_busy() -> bool:
    """Return True while server is running training or inference."""
    return _server_mode in ("training", "inference")


def is_training_active() -> bool:
    """Alias kept for panel compatibility."""
    return _server_mode == "training"


# --- Listener Callbacks (Background Thread) ---


def _on_state(array: np.ndarray) -> None:
    """Store latest state and start UI timer if not already running.

    Args:
        array: Server state tensor broadcast.
    """
    global _pending_state, _timer_running
    with _pending_lock:
        _pending_state = array
        if not _timer_running:
            _timer_running = True
            bpy.app.timers.register(_poll_state, first_interval=0.05)


def _on_error(message: str) -> None:
    """Print server error message received from listener thread."""
    print(f"[NCA] Server error: {message}")


def _on_disconnect() -> None:
    """Reset client state after remote disconnect."""
    global _client, _server_mode
    _client = None
    _server_mode = "idle"
    print("[NCA] Server disconnected")


def _poll_state() -> Optional[float]:
    """Main-thread timer - drains pending state buffer and renders it."""
    global _pending_state, _timer_running

    with _pending_lock:
        arr = _pending_state
        _pending_state = None

    if arr is not None:
        try:
            vis_ch = VISIBLE_CHANNELS_MAP.get(
                bpy.context.scene.nca_cell_props.visible_channels, 4
            )
            display = server_state_to_voxel_array(arr, visible_channels=vis_ch)
            _visualize_state(display)
        except Exception as exc:
            print(f"[NCA] Visualization error: {exc}")

    if not is_connected():
        _timer_running = False
        return None
    return 0.1


def _visualize_state(tensor: np.ndarray) -> None:
    """Render server state tensor in Blender state slot.

    Args:
        tensor: Display-ready tensor in external (D,H,W,C) format.
    """
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


# --- Target Helpers ---


def setup_configs(context) -> dict:
    """Build training config dict from Blender scene properties.

    Args:
        context: Blender context containing NCA property groups.

    Returns:
        Nested configuration dictionary for server init message.
    """
    cell = context.scene.nca_cell_props
    perc = context.scene.nca_perception_props
    upd = context.scene.nca_update_props
    grid = context.scene.nca_grid_props
    train = context.scene.nca_training_props
    return {
        "cell": {
            "hidden_channels": int(cell.hidden_channels),
            "visible_channels": VISIBLE_CHANNELS_MAP.get(cell.visible_channels, 4),
            "alive_threshold": float(cell.alive_threshold),
        },
        "perception": {
            "kernel_radius": int(perc.kernel_radius),
            "channel_groups": int(perc.channel_groups),
        },
        "update": {
            "hidden_dim": int(upd.hidden_dim),
            "stochastic_update": bool(upd.stochastic_update),
            "fire_rate": float(upd.fire_rate),
        },
        "grid": {
            "size": tuple(int(x) for x in grid.grid_size),
        },
        "training": {
            "learning_rate": float(train.learning_rate),
            "num_epochs": int(train.num_epochs),
            "batch_size": int(train.batch_size),
        },
    }


def get_target_array(context) -> np.ndarray:
    """Return cached target array or allocate empty target from grid settings."""
    if _target_array is not None:
        return _target_array
    grid = context.scene.nca_grid_props
    size = tuple(int(x) for x in grid.grid_size)
    vis = VISIBLE_CHANNELS_MAP.get(context.scene.nca_cell_props.visible_channels, 4)
    return np.zeros((*size, vis), dtype=np.float32)


def voxelize_and_display(context, objs: List[bpy.types.Object]) -> Tuple[bool, str]:
    """Voxelize source meshes, cache target tensor, and display source/target.

    Args:
        context: Blender context with NCA properties.
        objs: Mesh objects to voxelize and combine.

    Returns:
        Tuple (success, message) with operation status and user-facing text.
    """
    global _target_array
    cell = context.scene.nca_cell_props
    grid = context.scene.nca_grid_props
    target = context.scene.nca_target_props

    if not objs:
        return False, "No mesh objects provided"

    grid_size = tuple(int(x) for x in grid.grid_size)
    vis_ch = cell.visible_channels
    grid_offset = int(grid.grid_offset)

    results = [
        mesh_to_voxel_array(obj, grid_size, vis_ch, offset=grid_offset) for obj in objs
    ]

    combined_data = results[0][0].copy()
    combined_mat_map = results[0][1].copy()
    all_materials = list(results[0][2])

    for data, mat_map, mats in results[1:]:
        mat_offset = len(all_materials)
        all_materials.extend(mats)
        occupied = mat_map >= 0
        combined_mat_map[occupied] = mat_map[occupied] + mat_offset
        combined_data = np.maximum(combined_data, data)

    _target_array = combined_data
    place_source_in_scene(objs, grid_size, target.cell_size)

    vox_obj = voxel_array_to_blender(
        combined_data,
        collection_name=TARGET_COLLECTION,
        object_name="NCA_Target",
        cell_size=target.cell_size,
        alive_threshold=cell.alive_threshold,
        material_map=combined_mat_map,
        materials=all_materials,
    )
    if vox_obj:
        vox_obj.location.x = get_slot_offset(1, grid_size, target.cell_size)

    n_ch = combined_data.shape[-1]
    alpha = combined_data[..., 3] if n_ch >= 4 else combined_data[..., 0]
    n_alive = int((alpha > cell.alive_threshold).sum())
    names = ", ".join(o.name for o in objs)

    target.voxel_count = n_alive
    target.is_voxelized = True
    return True, f"Voxelized [{names}] -> {n_alive} voxels"


def clear_target(context) -> None:
    """Clear cached target tensor and remove source/target visualization objects."""
    global _target_array
    for name in (TARGET_COLLECTION, SOURCE_COLLECTION):
        if name in bpy.data.collections:
            clear_collection(bpy.data.collections[name])
    _target_array = None
    target = context.scene.nca_target_props
    target.voxel_count = 0
    target.is_voxelized = False


# --- Connection Operators ---


class NCA_OT_Connect(bpy.types.Operator):
    bl_idname = "nca.connect"
    bl_label = "Connect"
    bl_description = "Connect to the NCA training server"

    @classmethod
    def poll(cls, context):
        return not is_connected()

    def execute(self, context):
        global _client, _server_mode
        conn = context.scene.nca_connection_props
        client = NCAClient(host=conn.host, port=conn.port)
        try:
            client.connect()
        except Exception as exc:
            self.report({"ERROR"}, f"Connection failed: {exc}")
            return {"CANCELLED"}
        client.start_listener(
            on_state=_on_state,
            on_error=_on_error,
            on_disconnect=_on_disconnect,
        )
        _client = client
        _server_mode = "idle"
        self.report({"INFO"}, f"Connected to {conn.host}:{conn.port}")
        return {"FINISHED"}


class NCA_OT_Disconnect(bpy.types.Operator):
    bl_idname = "nca.disconnect"
    bl_label = "Disconnect"
    bl_description = "Disconnect from the NCA server (stops any active session)"

    @classmethod
    def poll(cls, context):
        return is_connected()

    def execute(self, context):
        global _client, _server_mode, _timer_running
        if _client:
            if is_server_busy():
                try:
                    _client.send_stop()
                except Exception:
                    pass
            _client.disconnect()
            _client = None
        _server_mode = "idle"
        _timer_running = False
        self.report({"INFO"}, "Disconnected")
        return {"FINISHED"}


# --- Training Operators ---


class NCA_OT_StartTraining(bpy.types.Operator):
    bl_idname = "nca.start_training"
    bl_label = "Start Training"
    bl_description = "Send training config and target to the connected server"

    @classmethod
    def poll(cls, context):
        return is_connected() and not is_server_busy() and _target_array is not None

    def execute(self, context):
        global _server_mode
        try:
            _client.send_init(setup_configs(context), get_target_array(context))
        except Exception as exc:
            self.report({"ERROR"}, f"Failed to start training: {exc}")
            return {"CANCELLED"}
        _server_mode = "training"
        self.report({"INFO"}, "Training started")
        return {"FINISHED"}


class NCA_OT_StopTraining(bpy.types.Operator):
    bl_idname = "nca.stop_training"
    bl_label = "Stop"
    bl_description = "Stop the active training or inference session (stays connected)"

    @classmethod
    def poll(cls, context):
        return is_connected() and is_server_busy()

    def execute(self, context):
        global _server_mode
        try:
            _client.send_stop()
        except Exception:
            pass
        _server_mode = "idle"
        self.report({"INFO"}, "Stopped")
        return {"FINISHED"}


class NCA_OT_PauseTraining(bpy.types.Operator):
    bl_idname = "nca.pause_training"
    bl_label = "Pause"
    bl_description = "Pause the active training session"

    @classmethod
    def poll(cls, context):
        return is_connected() and _server_mode == "training"

    def execute(self, context):
        try:
            _client.send_pause()
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, "Paused")
        return {"FINISHED"}


class NCA_OT_ResumeTraining(bpy.types.Operator):
    bl_idname = "nca.resume_training"
    bl_label = "Resume"
    bl_description = "Resume a paused training session"

    @classmethod
    def poll(cls, context):
        return is_connected() and _server_mode == "training"

    def execute(self, context):
        try:
            _client.send_resume()
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, "Resumed")
        return {"FINISHED"}


# --- Inference Operator ---


class NCA_OT_RunModel(bpy.types.Operator):
    bl_idname = "nca.run_model"
    bl_label = "Run Model"
    bl_description = "Load a saved .pt model and run it on the server"

    @classmethod
    def poll(cls, context):
        infer = context.scene.nca_inference_props
        return is_connected() and not is_server_busy() and bool(infer.model_path)

    def execute(self, context):
        global _server_mode
        infer = context.scene.nca_inference_props
        model_path = bpy.path.abspath(infer.model_path)

        if not os.path.isfile(model_path):
            self.report({"ERROR"}, f"Model file not found: {model_path}")
            return {"CANCELLED"}

        try:
            _client.send_run_model(
                model_path,
                infer.steps_per_phase,
                infer.broadcast_every,
            )
        except Exception as exc:
            self.report({"ERROR"}, f"Failed to run model: {exc}")
            return {"CANCELLED"}

        _server_mode = "inference"
        self.report({"INFO"}, "Inference started")
        return {"FINISHED"}


# --- Target Operators ---


class NCA_OT_VoxelizeTarget(bpy.types.Operator):
    bl_idname = "nca.voxelize_target"
    bl_label = "Voxelize"
    bl_description = "Voxelize selected mesh objects into the NCA grid"

    def execute(self, context):
        objs = [o for o in context.selected_objects if o.type == "MESH"]
        if not objs:
            self.report({"ERROR"}, "No mesh objects selected")
            return {"CANCELLED"}
        ok, msg = voxelize_and_display(context, objs)
        self.report({"INFO"} if ok else {"ERROR"}, msg)
        return {"FINISHED"} if ok else {"CANCELLED"}


class NCA_OT_ClearTargetVoxels(bpy.types.Operator):
    bl_idname = "nca.clear_target_voxels"
    bl_label = "Clear"
    bl_description = "Remove the voxelized target from the viewport"

    def execute(self, context):
        clear_target(context)
        context.scene.nca_target_props["source_object"] = None
        self.report({"INFO"}, "Target cleared")
        return {"FINISHED"}


class NCA_OT_ExportTarget(bpy.types.Operator):
    bl_idname = "nca.export_target"
    bl_label = "Export Target"
    bl_description = (
        "Save the voxelized target as .npz for use in Colab/external training"
    )

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")  # type: ignore

    @classmethod
    def poll(cls, context):
        return context.scene.nca_target_props.is_voxelized and _target_array is not None

    def invoke(self, context, event):
        src = context.scene.nca_target_props.source_object
        name = bpy.path.clean_name(src.name if src else "nca_target") or "nca_target"
        self.filepath = bpy.path.abspath(f"//{name}.npz")
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        if _target_array is None:
            self.report({"ERROR"}, "No voxelized target to export")
            return {"CANCELLED"}
        path = bpy.path.abspath(self.filepath)
        if not path.lower().endswith(".npz"):
            path += ".npz"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(
            path, target=_target_array.astype(np.float32), config=setup_configs(context)
        )
        self.report({"INFO"}, f"Exported to {path}")
        return {"FINISHED"}


# --- Schedule Operators ---


class NCA_OT_AddScheduleEvent(bpy.types.Operator):
    bl_idname = "nca.add_schedule_event"
    bl_label = "Add Event"

    def execute(self, context):
        sched = context.scene.nca_schedule_props
        item = sched.events.add()
        item.epoch = -1
        item.event_type = "LEARNING_RATE"
        item.value = 0.01
        sched.active_event_index = len(sched.events) - 1
        return {"FINISHED"}


class NCA_OT_RemoveScheduleEvent(bpy.types.Operator):
    bl_idname = "nca.remove_schedule_event"
    bl_label = "Remove Event"

    @classmethod
    def poll(cls, context):
        return len(context.scene.nca_schedule_props.events) > 0

    def execute(self, context):
        sched = context.scene.nca_schedule_props
        idx = sched.active_event_index
        sched.events.remove(idx)
        sched.active_event_index = max(0, idx - 1)
        return {"FINISHED"}


class NCA_OT_SendSchedule(bpy.types.Operator):
    bl_idname = "nca.send_schedule"
    bl_label = "Send Schedule"
    bl_description = "Send the event schedule to the training server"

    @classmethod
    def poll(cls, context):
        return is_connected() and _server_mode == "training"

    def execute(self, context):
        from .protocol import tensor_to_b64

        sched = context.scene.nca_schedule_props
        events = []
        for ev in sched.events:
            entry = {"epoch": ev.epoch, "event_type": ev.event_type, "value": ev.value}
            if ev.event_type == "TARGET_CHANGE":
                if ev.target_object is None:
                    self.report({"ERROR"}, "TARGET_CHANGE event has no mesh assigned")
                    return {"CANCELLED"}
                ok, msg = voxelize_and_display(context, [ev.target_object])
                if not ok:
                    self.report({"ERROR"}, msg)
                    return {"CANCELLED"}
                if context.scene.nca_target_props.voxel_count == 0:
                    self.report(
                        {"ERROR"}, f"'{ev.target_object.name}' produced no alive voxels"
                    )
                    return {"CANCELLED"}
                entry["target"] = tensor_to_b64(_target_array)
                entry["target_shape"] = list(_target_array.shape)
            events.append(entry)

        try:
            _client.send_schedule(events)
            self.report({"INFO"}, f"Sent {len(events)} event(s)")
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


# --- Registration ---

classes = (
    NCA_OT_Connect,
    NCA_OT_Disconnect,
    NCA_OT_StartTraining,
    NCA_OT_StopTraining,
    NCA_OT_PauseTraining,
    NCA_OT_ResumeTraining,
    NCA_OT_RunModel,
    NCA_OT_VoxelizeTarget,
    NCA_OT_ClearTargetVoxels,
    NCA_OT_ExportTarget,
    NCA_OT_AddScheduleEvent,
    NCA_OT_RemoveScheduleEvent,
    NCA_OT_SendSchedule,
)


def register():
    """Register Blender operator classes."""
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    """Unregister Blender operator classes in reverse order."""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
