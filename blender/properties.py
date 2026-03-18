import bpy

NCA_COLLECTIONS = {"NCA_Source", "NCA_Target", "NCA_State"}

def _is_source_mesh_candidate(self, obj: bpy.types.Object) -> bool:
    """Poll filter for the source mesh picker.

    Returns True only for MESH objects that are NOT generated NCA voxels:
    - Must be a MESH type
    - Must not have a name starting with 'NCA_'
    - Must not belong to any NCA-managed collection
    """
    if obj.type != 'MESH':
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
        description="Number of hidden channels in the NCA",
        default=16,
        min=1
    ) # type: ignore
    visible_channels: bpy.props.EnumProperty(
        name="Visible Channels",
        description="Channels that are visible",
        items=[
            ('ALPHA', "Alpha", "Alpha Channel Only (1 channel)"),
            ('RGBA', "RGBA", "RGBA Channels (4 channels)"),
            ('ALPHA_MATERIAL_ID', "Alpha + Material ID", "Alpha + Material ID Channels (2 channels)"),
        ],
        default='RGBA'
    ) # type: ignore
    alive_threshold: bpy.props.FloatProperty(
        name="Alive Threshold",
        description="Threshold for cell survival",
        default=0.02,
        min=0.0
    ) # type: ignore

class NCA_PG_PerceptionProperties(bpy.types.PropertyGroup):
    kernel_radius: bpy.props.IntProperty(
        name="Kernel Radius",
        description="Radius of the perception kernel",
        default=1,
        min=1
    ) # type: ignore
    channel_groups: bpy.props.IntProperty(
        name="Channel Groups",
        description="Number of channel groups for perception",
        default=3,
        min=1
    ) # type: ignore

class NCA_PG_UpdateProperties(bpy.types.PropertyGroup):
    hidden_dim: bpy.props.IntProperty(
        name="Hidden Dimension",
        description="Dimension of the hidden layer",
        default=96,
        min=1
    ) # type: ignore
    stochastic_update: bpy.props.BoolProperty(
        name="Stochastic Update",
        description="Enable stochastic updates",
        default=False
    ) # type: ignore
    fire_rate: bpy.props.FloatProperty(
        name="Fire Rate",
        description="Probability of a cell updating",
        default=0.5,
        min=0.0,
        max=1.0
    ) # type: ignore

class NCA_PG_GridProperties(bpy.types.PropertyGroup):
    grid_size: bpy.props.IntVectorProperty(
        name="Grid Size",
        description="Size of the NCA grid",
        default=(32, 32, 32),
        size=3,
        min=1
    ) # type: ignore
    grid_offset: bpy.props.IntProperty(
        name="Grid Offset",
        description="Voxel offset from grid edges",
        default=1,
        min=0
    ) # type: ignore

class NCA_PG_TrainingProperties(bpy.types.PropertyGroup):
    learning_rate: bpy.props.FloatProperty(
        name="Learning Rate",
        description="Learning rate for training",
        default=0.001,
        min=0.0
    ) # type: ignore
    batch_size: bpy.props.IntProperty(
        name="Batch Size",
        description="Batch size for training",
        default=4,
        min=1
    ) # type: ignore
    num_epochs: bpy.props.IntProperty(
        name="Number of Epochs",
        description="Number of epochs for training",
        default=5000,
        min=1
    ) # type: ignore

class NCA_PG_TargetProperties(bpy.types.PropertyGroup):
    source_object: bpy.props.PointerProperty(
        name="Source Mesh",
        description="Mesh object to voxelize as the NCA target",
        type=bpy.types.Object,
        poll=_is_source_mesh_candidate,
        update=lambda self, ctx: _on_source_changed(self, ctx),
    ) # type: ignore
    is_voxelized: bpy.props.BoolProperty(
        name="Is Voxelized",
        description="Whether a target has been voxelized",
        default=False
    ) # type: ignore
    voxel_count: bpy.props.IntProperty(
        name="Voxel Count",
        description="Number of alive voxels in the target",
        default=0
    ) # type: ignore
    cell_size: bpy.props.FloatProperty(
        name="Cell Size",
        description="Size of each cell in the visualization",
        default=0.1,
        min=0.01
    ) # type: ignore

def _on_source_changed(target_props, context):
    """Auto-voxelize when a new source mesh is picked, or clear when set to None."""
    # Import here to avoid circular imports
    from . import operators as ops

    if target_props.source_object is not None:
        ops.voxelize_and_display(context, [target_props.source_object])
    else:
        ops.clear_target(context)

SCHEDULE_EVENT_TYPES = [
    ('LEARNING_RATE',   "Learning Rate",   "Change the optimizer learning rate"),
    ('BATCH_SIZE',      "Batch Size",      "Change the training batch size"),
    ('ALPHA_WEIGHT',    "Alpha Weight",    "Change the alpha/occupancy loss weight"),
    ('COLOR_WEIGHT',    "Color Weight",    "Change the color loss weight"),
    ('OVERFLOW_WEIGHT', "Overflow Weight", "Change the overflow loss weight"),
    ('TARGET_CHANGE',   "Target Change",   "Swap the training target to a different mesh"),
]

class NCA_PG_ScheduleEvent(bpy.types.PropertyGroup):
    """A single scheduled parameter change."""
    epoch: bpy.props.IntProperty(
        name="Epoch",
        description="Epoch when the event fires (-1 = NOW, executes immediately)",
        default=-1,
        min=-1,
    ) # type: ignore
    event_type: bpy.props.EnumProperty(
        name="Type",
        description="Which parameter to change",
        items=SCHEDULE_EVENT_TYPES,
        default='LEARNING_RATE',
    ) # type: ignore
    value: bpy.props.FloatProperty(
        name="Value",
        description="New value for the parameter",
        default=0,
        precision=6,
    ) # type: ignore
    target_object: bpy.props.PointerProperty(
        name="Target Mesh",
        description="Mesh to voxelize as the new training target",
        type=bpy.types.Object,
        poll=_is_source_mesh_candidate,
    ) # type: ignore

class NCA_PG_ScheduleProperties(bpy.types.PropertyGroup):
    """Container for the event list shown in the Schedule panel."""
    events: bpy.props.CollectionProperty(
        type=NCA_PG_ScheduleEvent,
    ) # type: ignore
    active_event_index: bpy.props.IntProperty(
        name="Active Event",
        default=0,
    ) # type: ignore

class NCA_PG_ConnectionProperties(bpy.types.PropertyGroup):
    host: bpy.props.StringProperty(
        name="Host",
        description="NCA server hostname or IP (use the ngrok host when tunnelling)",
        default="127.0.0.1",
    )  # type: ignore
    port: bpy.props.IntProperty(
        name="Port",
        description="NCA server port",
        default=5555,
        min=1,
        max=65535,
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
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.nca_cell_props = bpy.props.PointerProperty(type=NCA_PG_CellProperties)
    bpy.types.Scene.nca_perception_props = bpy.props.PointerProperty(type=NCA_PG_PerceptionProperties)
    bpy.types.Scene.nca_update_props = bpy.props.PointerProperty(type=NCA_PG_UpdateProperties)
    bpy.types.Scene.nca_grid_props = bpy.props.PointerProperty(type=NCA_PG_GridProperties)
    bpy.types.Scene.nca_training_props = bpy.props.PointerProperty(type=NCA_PG_TrainingProperties)
    bpy.types.Scene.nca_target_props = bpy.props.PointerProperty(type=NCA_PG_TargetProperties)
    bpy.types.Scene.nca_schedule_props = bpy.props.PointerProperty(type=NCA_PG_ScheduleProperties)
    bpy.types.Scene.nca_connection_props = bpy.props.PointerProperty(type=NCA_PG_ConnectionProperties)

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