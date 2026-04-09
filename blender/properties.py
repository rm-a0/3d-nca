"""
Blender PropertyGroup definitions for NCA add-on UI.

Defines data structures for:
- NCA model configuration (cell, perception, update, grid, training)
- Target mesh and voxelization state
- Training schedule events
- Server connection settings
- Inference parameters

Properties are registered and stored on bpy.context.scene for persistence.
"""

import bpy

NCA_COLLECTIONS = {"NCA_Source", "NCA_Target", "NCA_State"}


def _is_source_mesh_candidate(self, obj: bpy.types.Object) -> bool:
    """Filter for mesh poll callback - exclude NCA-generated objects.

    Args:
        obj: Candidate object to check.

    Returns:
        True if obj is a user mesh not in NCA collections, False otherwise.
    """
    if obj.type != "MESH":
        return False
    if obj.name.startswith("NCA_"):
        return False
    for col in obj.users_collection:
        if col.name in NCA_COLLECTIONS:
            return False
    return True


class NCA_PG_CellProperties(bpy.types.PropertyGroup):
    hidden_channels: bpy.props.IntProperty(
        name="Hidden Channels",
        default=16,
        min=1,
    )  # type: ignore
    visible_channels: bpy.props.EnumProperty(
        name="Visible Channels",
        items=[
            ("ALPHA", "Alpha", "1 channel"),
            ("RGBA", "RGBA", "4 channels"),
            ("ALPHA_MATERIAL_ID", "Alpha + Material ID", "2 channels"),
        ],
        default="RGBA",
    )  # type: ignore
    alive_threshold: bpy.props.FloatProperty(
        name="Alive Threshold",
        default=0.02,
        min=0.0,
    )  # type: ignore


class NCA_PG_PerceptionProperties(bpy.types.PropertyGroup):
    kernel_radius: bpy.props.IntProperty(
        name="Kernel Radius",
        default=1,
        min=1,
    )  # type: ignore
    channel_groups: bpy.props.IntProperty(
        name="Channel Groups",
        default=3,
        min=1,
    )  # type: ignore


class NCA_PG_UpdateProperties(bpy.types.PropertyGroup):
    hidden_dim: bpy.props.IntProperty(
        name="Hidden Dimension",
        default=96,
        min=1,
    )  # type: ignore
    stochastic_update: bpy.props.BoolProperty(
        name="Stochastic Update",
        default=False,
    )  # type: ignore
    fire_rate: bpy.props.FloatProperty(
        name="Fire Rate",
        default=0.5,
        min=0.0,
        max=1.0,
    )  # type: ignore


class NCA_PG_GridProperties(bpy.types.PropertyGroup):
    grid_size: bpy.props.IntVectorProperty(
        name="Grid Size",
        default=(32, 32, 32),
        size=3,
        min=1,
    )  # type: ignore
    grid_offset: bpy.props.IntProperty(
        name="Grid Offset",
        default=1,
        min=0,
    )  # type: ignore


class NCA_PG_TrainingProperties(bpy.types.PropertyGroup):
    learning_rate: bpy.props.FloatProperty(
        name="Learning Rate",
        default=0.001,
        min=0.0,
    )  # type: ignore
    batch_size: bpy.props.IntProperty(
        name="Batch Size",
        default=4,
        min=1,
    )  # type: ignore
    num_epochs: bpy.props.IntProperty(
        name="Epochs",
        default=5000,
        min=1,
    )  # type: ignore


class NCA_PG_TargetProperties(bpy.types.PropertyGroup):
    source_object: bpy.props.PointerProperty(
        name="Source Mesh",
        type=bpy.types.Object,
        poll=_is_source_mesh_candidate,
        update=lambda self, ctx: _on_source_changed(self, ctx),
    )  # type: ignore
    is_voxelized: bpy.props.BoolProperty(name="Is Voxelized", default=False)  # type: ignore
    voxel_count: bpy.props.IntProperty(name="Voxel Count", default=0)  # type: ignore
    cell_size: bpy.props.FloatProperty(name="Cell Size", default=0.1, min=0.01)  # type: ignore


def _on_source_changed(target_props, context):
    """Update callback when source mesh property changes.

    Voxelizes new source or clears voxels if source is removed.

    Args:
        target_props: NCA_PG_TargetProperties instance.
        context: Blender context.
    """
    from . import operators as ops

    if target_props.source_object is not None:
        ops.voxelize_and_display(context, [target_props.source_object])
    else:
        ops.clear_target(context)


SCHEDULE_EVENT_TYPES = [
    ("LEARNING_RATE", "Learning Rate", "Change optimizer learning rate"),
    ("BATCH_SIZE", "Batch Size", "Change training batch size"),
    ("ALPHA_WEIGHT", "Alpha Weight", "Change alpha loss weight"),
    ("COLOR_WEIGHT", "Color Weight", "Change color loss weight"),
    ("OVERFLOW_WEIGHT", "Overflow Weight", "Change overflow loss weight"),
    ("TARGET_CHANGE", "Target Change", "Swap training target"),
]


class NCA_PG_ScheduleEvent(bpy.types.PropertyGroup):
    epoch: bpy.props.IntProperty(
        name="Epoch",
        description="Epoch when the event fires (-1 = NOW)",
        default=-1,
        min=-1,
    )  # type: ignore
    event_type: bpy.props.EnumProperty(
        name="Type",
        items=SCHEDULE_EVENT_TYPES,
        default="LEARNING_RATE",
    )  # type: ignore
    value: bpy.props.FloatProperty(name="Value", default=0, precision=6)  # type: ignore
    target_object: bpy.props.PointerProperty(
        name="Target Mesh",
        type=bpy.types.Object,
        poll=_is_source_mesh_candidate,
    )  # type: ignore


class NCA_PG_ScheduleProperties(bpy.types.PropertyGroup):
    events: bpy.props.CollectionProperty(type=NCA_PG_ScheduleEvent)  # type: ignore
    active_event_index: bpy.props.IntProperty(name="Active Event", default=0)  # type: ignore


class NCA_PG_ConnectionProperties(bpy.types.PropertyGroup):
    host: bpy.props.StringProperty(
        name="Host",
        description="Server hostname or IP (ngrok host for remote training)",
        default="127.0.0.1",
    )  # type: ignore
    port: bpy.props.IntProperty(
        name="Port",
        default=5555,
        min=1,
        max=65535,
    )  # type: ignore


class NCA_PG_InferenceProperties(bpy.types.PropertyGroup):
    """Settings for running a pre-trained model."""

    model_path: bpy.props.StringProperty(
        name="Model Path",
        description="Path to a saved .pt model file",
        subtype="FILE_PATH",
    )  # type: ignore
    steps_per_phase: bpy.props.IntProperty(
        name="Steps per Phase",
        description="NCA steps to run per growth phase (or total steps if no task channels)",
        default=32,
        min=1,
        max=256,
    )  # type: ignore
    broadcast_every: bpy.props.IntProperty(
        name="Broadcast Every N Steps",
        description="How often to send the state to Blender during inference",
        default=4,
        min=1,
        max=64,
    )  # type: ignore
    send_delay_ms: bpy.props.IntProperty(
        name="Send Delay (ms)",
        description="Delay before each state broadcast during inference",
        default=40,
        min=0,
        max=1000,
    )  # type: ignore


classes = (
    NCA_PG_CellProperties,
    NCA_PG_PerceptionProperties,
    NCA_PG_UpdateProperties,
    NCA_PG_GridProperties,
    NCA_PG_TrainingProperties,
    NCA_PG_TargetProperties,
    NCA_PG_ScheduleEvent,
    NCA_PG_ScheduleProperties,
    NCA_PG_ConnectionProperties,
    NCA_PG_InferenceProperties,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.nca_cell_props = bpy.props.PointerProperty(
        type=NCA_PG_CellProperties
    )
    bpy.types.Scene.nca_perception_props = bpy.props.PointerProperty(
        type=NCA_PG_PerceptionProperties
    )
    bpy.types.Scene.nca_update_props = bpy.props.PointerProperty(
        type=NCA_PG_UpdateProperties
    )
    bpy.types.Scene.nca_grid_props = bpy.props.PointerProperty(
        type=NCA_PG_GridProperties
    )
    bpy.types.Scene.nca_training_props = bpy.props.PointerProperty(
        type=NCA_PG_TrainingProperties
    )
    bpy.types.Scene.nca_target_props = bpy.props.PointerProperty(
        type=NCA_PG_TargetProperties
    )
    bpy.types.Scene.nca_schedule_props = bpy.props.PointerProperty(
        type=NCA_PG_ScheduleProperties
    )
    bpy.types.Scene.nca_connection_props = bpy.props.PointerProperty(
        type=NCA_PG_ConnectionProperties
    )
    bpy.types.Scene.nca_inference_props = bpy.props.PointerProperty(
        type=NCA_PG_InferenceProperties
    )


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.nca_cell_props
    del bpy.types.Scene.nca_perception_props
    del bpy.types.Scene.nca_update_props
    del bpy.types.Scene.nca_grid_props
    del bpy.types.Scene.nca_training_props
    del bpy.types.Scene.nca_target_props
    del bpy.types.Scene.nca_schedule_props
    del bpy.types.Scene.nca_connection_props
    del bpy.types.Scene.nca_inference_props
