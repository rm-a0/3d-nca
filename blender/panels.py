import bpy

class NCA_PT_BasePanel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'NCA'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

class NCA_PT_MainPanel(NCA_PT_BasePanel):
    bl_label = "3D NCA Settings"
    bl_idname = "NCA_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'NCA'

    def draw(self, context):
        pass

class NCA_PT_ControlPanel(NCA_PT_BasePanel):
    bl_label = "Control"
    bl_idname = "NCA_PT_control_panel"
    bl_parent_id = "NCA_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 0

    def draw(self, context):
        super().draw(context)

        layout = self.layout
        layout.operator("nca.start_training", text="Start Training")
        layout.operator("nca.stop_training", text="Stop Training")
        layout.operator("nca.pause_training", text="Pause Training")
        layout.operator("nca.resume_training", text="Resume Training")

class NCA_PT_TargetPanel(NCA_PT_BasePanel):
    bl_label = "Target"
    bl_idname = "NCA_PT_target_panel"
    bl_parent_id = "NCA_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        super().draw(context)
        layout = self.layout
        target_props = context.scene.nca_target_props

        box = layout.box()
        box.label(text="Step 1: Select source mesh", icon='RESTRICT_SELECT_OFF')
        meshes = [o for o in context.selected_objects if o.type == 'MESH']
        if meshes:
            names = ", ".join(o.name for o in meshes)
            box.label(text=names, icon='MESH_DATA')
        else:
            box.label(text="No mesh selected", icon='INFO')

        box = layout.box()
        box.label(text="Step 2: Voxelize", icon='MESH_GRID')
        row = box.row(align=True)
        row.operator("nca.voxelize_target", text="Voxelize", icon='MESH_GRID')
        row.operator("nca.clear_target_voxels", text="Clear", icon='X')
        row.enabled = len(meshes) > 0 or target_props.is_voxelized

        if target_props.is_voxelized:
            status_box = layout.box()
            status_box.label(text="Target ready to send", icon='CHECKMARK')
            status_box.label(text=f"Source: {target_props.source_names}")
            status_box.label(text=f"Voxels: {target_props.voxel_count}")
        else:
            status_box = layout.box()
            status_box.label(text="No target voxelized", icon='ERROR')
            status_box.label(text="Voxelize a mesh before training")


class NCA_PT_CellSettings(NCA_PT_BasePanel):
    bl_label = "Cell Settings"
    bl_idname = "NCA_PT_cell_settings"
    bl_parent_id = "NCA_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        super().draw(context)

        layout = self.layout
        cell_props = context.scene.nca_cell_props

        layout.prop(cell_props, "hidden_channels", text="Hidden Channels")
        layout.prop(cell_props, "visible_channels", text="Visible Channels")
        layout.prop(cell_props, "alive_threshold", text="Alive Threshold")

class NCA_PT_PerceptionSettings(NCA_PT_BasePanel):
    bl_label = "Perception Settings"
    bl_idname = "NCA_PT_perception_settings"
    bl_parent_id = "NCA_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        super().draw(context)

        layout = self.layout
        perc_props = context.scene.nca_perception_props

        layout.prop(perc_props, "kernel_radius", text="Kernel Radius")
        layout.prop(perc_props, "channel_groups", text="Channel Groups")

class NCA_PT_UpdateSettings(NCA_PT_BasePanel):
    bl_label = "Update Settings"
    bl_idname = "NCA_PT_update_settings"
    bl_parent_id = "NCA_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        super().draw(context)

        layout = self.layout
        upd_props = context.scene.nca_update_props

        layout.prop(upd_props, "hidden_dim", text="Hidden Dimension")
        layout.prop(upd_props, "stochastic_update", text="Stochastic Update")
        layout.prop(upd_props, "fire_rate", text="Fire Rate")

class NCA_PT_GridSettings(NCA_PT_BasePanel):
    bl_label = "Grid Settings"
    bl_idname = "NCA_PT_grid_settings"
    bl_parent_id = "NCA_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        super().draw(context)

        layout = self.layout
        grid_props = context.scene.nca_grid_props

        layout.prop(grid_props, "grid_size", text="Grid Size")
        layout.prop(grid_props, "grid_offset", text="Grid Offset")

class NCA_PT_TrainingSettings(NCA_PT_BasePanel):
    bl_label = "Training Settings"
    bl_idname = "NCA_PT_training_settings"
    bl_parent_id = "NCA_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        super().draw(context)

        layout = self.layout
        train_props = context.scene.nca_training_props

        layout.prop(train_props, "learning_rate", text="Learning Rate")
        layout.prop(train_props, "batch_size", text="Batch Size")
        layout.prop(train_props, "num_epochs", text="Number of Epochs")

class NCA_PT_VisualizationSettings(NCA_PT_BasePanel):
    bl_label = "Visualization Settings"
    bl_idname = "NCA_PT_visualization_settings"
    bl_parent_id = "NCA_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        super().draw(context)

        layout = self.layout
        vis_props = context.scene.nca_visualization_props

        layout.prop(vis_props, "cell_shape", text="Cell Shape")
        layout.prop(vis_props, "cell_size", text="Cell Size")
        layout.prop(vis_props, "show_grid", text="Show Grid")
        layout.prop(vis_props, "animation_speed", text="Animation Speed")

classes = (
    NCA_PT_MainPanel,
    NCA_PT_ControlPanel,
    NCA_PT_TargetPanel,
    NCA_PT_CellSettings,
    NCA_PT_PerceptionSettings,
    NCA_PT_UpdateSettings,
    NCA_PT_GridSettings,
    NCA_PT_TrainingSettings,
    NCA_PT_VisualizationSettings,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)