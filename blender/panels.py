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

        layout.prop(perc_props, "channel_groups", text="Channel Groups")
        layout.prop(perc_props, "perception_radius", text="Perception Radius")

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

class NCA_PT_RunPanel(NCA_PT_BasePanel):
    bl_label = "Run NCA"
    bl_idname = "NCA_PT_run_panel"
    bl_parent_id = "NCA_PT_main_panel"
    bl_options = {'HIDE_HEADER'}
    bl_order = 10

    def draw(self, context):
        super().draw(context)

        layout = self.layout
        layout.operator("nca.run_simulation", text="Run Simulation")
        layout.operator("nca.stop_simulation", text="Stop Simulation")

classes = (
    NCA_PT_MainPanel,
    NCA_PT_CellSettings,
    NCA_PT_PerceptionSettings,
    NCA_PT_UpdateSettings,
    NCA_PT_GridSettings,
    NCA_PT_TrainingSettings,
    NCA_PT_VisualizationSettings,
    NCA_PT_RunPanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)