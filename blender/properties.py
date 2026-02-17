import bpy

class NCA_PG_CellProperties(bpy.types.PropertyGroup):
    hidden_channels: bpy.props.IntProperty(
        name="Hidden Channels",
        description="Number of hidden channels in the NCA",
        default=8,
        min=0
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
        default=4,
        min=1
    ) # type: ignore
    

class NCA_PG_UpdateProperties(bpy.types.PropertyGroup):
    hidden_dim: bpy.props.IntProperty(
        name="Hidden Dimension",
        description="Dimension of the hidden layer",
        default=64,
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
        default=16,
        min=1
    ) # type: ignore
    num_epochs: bpy.props.IntProperty(
        name="Number of Epochs",
        description="Number of epochs for training",
        default=2000,
        min=1
    ) # type: ignore

class NCA_PG_TargetProperties(bpy.types.PropertyGroup):
    is_voxelized: bpy.props.BoolProperty(
        name="Is Voxelized",
        description="Whether a target has been voxelized",
        default=False
    ) # type: ignore
    source_names: bpy.props.StringProperty(
        name="Source Names",
        description="Names of the source meshes that were voxelized",
        default=""
    ) # type: ignore
    voxel_count: bpy.props.IntProperty(
        name="Voxel Count",
        description="Number of alive voxels in the target",
        default=0
    ) # type: ignore

class NCA_PG_VisualizationProperties(bpy.types.PropertyGroup):
    cell_shape: bpy.props.EnumProperty(
        name="Cell Shape",
        description="Shape used to visualize cells",
        items=[
            ('CUBE', "Cube", "Visualize cells as cubes"),
            ('SPHERE', "Sphere", "Visualize cells as spheres"),
        ],
        default='CUBE'
    ) # type: ignore
    cell_size: bpy.props.FloatProperty(
        name="Cell Size",
        description="Size of each cell in the visualization",
        default=0.1,
        min=0.01
    ) # type: ignore
    show_grid: bpy.props.BoolProperty(
        name="Show Grid",
        description="Toggle grid visibility",
        default=True
    ) # type: ignore
    animation_speed: bpy.props.FloatProperty(
        name="Animation Speed",
        description="Speed of the NCA animation",
        default=1.0,
        min=0.1
    ) # type: ignore

classes = (
    NCA_PG_CellProperties,
    NCA_PG_PerceptionProperties,
    NCA_PG_UpdateProperties,
    NCA_PG_GridProperties,
    NCA_PG_TrainingProperties,
    NCA_PG_TargetProperties,
    NCA_PG_VisualizationProperties,
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
    bpy.types.Scene.nca_visualization_props = bpy.props.PointerProperty(type=NCA_PG_VisualizationProperties)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.nca_cell_props
    del bpy.types.Scene.nca_perception_props
    del bpy.types.Scene.nca_update_props
    del bpy.types.Scene.nca_grid_props
    del bpy.types.Scene.nca_training_props
    del bpy.types.Scene.nca_target_props
    del bpy.types.Scene.nca_visualization_props