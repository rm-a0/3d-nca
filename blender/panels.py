import bpy

class NCA_PT_BasePanel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'NCA'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

class NCA_PT_ConnectionPanel(NCA_PT_BasePanel):
    bl_label = "Connection"
    bl_idname = "NCA_PT_connection_panel"
    bl_parent_id = "NCA_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = -1

    def draw(self, context):
        from .operators import is_training_active
        super().draw(context)

        layout = self.layout
        layout.enabled = not is_training_active()
        conn = context.scene.nca_connection_props

        layout.prop(conn, "host", text="Host")
        layout.prop(conn, "port", text="Port")

        layout.separator()
        layout.label(
            text="Paste the ngrok host/port here when training remotely.",
            icon='INFO',
        )

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
        from .operators import is_training_active
        super().draw(context)
        layout = self.layout
        target_props = context.scene.nca_target_props

        active = not is_training_active()

        layout.prop(target_props, "cell_size")

        row = layout.row(align=True)
        row.enabled = active
        row.prop(target_props, "source_object", text="Source Mesh")
        row.operator("nca.clear_target_voxels", text="", icon='X')

        if target_props.is_voxelized:
            box = layout.box()
            box.label(text="Target ready", icon='CHECKMARK')
            if target_props.source_object:
                box.label(text=f"Source: {target_props.source_object.name}")
            box.label(text=f"Voxels: {target_props.voxel_count}")
            row = box.row()
            row.enabled = active
            row.operator("nca.export_target", text="Export Target")
        else:
            box = layout.box()
            box.label(text="No target voxelized", icon='ERROR')

class NCA_PT_CellSettings(NCA_PT_BasePanel):
    bl_label = "Cell Settings"
    bl_idname = "NCA_PT_cell_settings"
    bl_parent_id = "NCA_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        from .operators import is_training_active
        super().draw(context)

        layout = self.layout
        layout.enabled = not is_training_active()
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
        from .operators import is_training_active
        super().draw(context)

        layout = self.layout
        layout.enabled = not is_training_active()
        perc_props = context.scene.nca_perception_props

        layout.prop(perc_props, "kernel_radius", text="Kernel Radius")
        layout.prop(perc_props, "channel_groups", text="Channel Groups")

class NCA_PT_UpdateSettings(NCA_PT_BasePanel):
    bl_label = "Update Settings"
    bl_idname = "NCA_PT_update_settings"
    bl_parent_id = "NCA_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        from .operators import is_training_active
        super().draw(context)

        layout = self.layout
        layout.enabled = not is_training_active()
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
        from .operators import is_training_active
        super().draw(context)

        layout = self.layout
        layout.enabled = not is_training_active()
        grid_props = context.scene.nca_grid_props

        layout.prop(grid_props, "grid_size", text="Grid Size")
        layout.prop(grid_props, "grid_offset", text="Grid Offset")

class NCA_PT_TrainingSettings(NCA_PT_BasePanel):
    bl_label = "Training Settings"
    bl_idname = "NCA_PT_training_settings"
    bl_parent_id = "NCA_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        from .operators import is_training_active
        super().draw(context)

        layout = self.layout
        layout.enabled = not is_training_active()
        train_props = context.scene.nca_training_props

        layout.prop(train_props, "learning_rate", text="Learning Rate")
        layout.prop(train_props, "batch_size", text="Batch Size")
        layout.prop(train_props, "num_epochs", text="Number of Epochs")

class NCA_UL_ScheduleEventList(bpy.types.UIList):
    """Draws one row per schedule event."""
    bl_idname = "NCA_UL_schedule_event_list"

    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            if item.epoch == -1:
                row.label(text="Now")
            else:
                row.prop(item, "epoch", text="", emboss=False)
            row.prop(item, "event_type", text="", emboss=False)
            if item.event_type == 'TARGET_CHANGE':
                row.prop(item, "target_object", text="", emboss=False)
            else:
                row.prop(item, "value", text="", emboss=False)
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon='EVENT_S')

class NCA_PT_SchedulePanel(NCA_PT_BasePanel):
    bl_label = "Schedule"
    bl_idname = "NCA_PT_schedule_panel"
    bl_parent_id = "NCA_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        from .operators import is_training_active
        super().draw(context)

        layout = self.layout
        sched = context.scene.nca_schedule_props

        row = layout.row()
        row.template_list(
            "NCA_UL_schedule_event_list", "",
            sched, "events",
            sched, "active_event_index",
            rows=3,
        )

        col = row.column(align=True)
        col.operator("nca.add_schedule_event", icon='ADD', text="")
        col.operator("nca.remove_schedule_event", icon='REMOVE', text="")

        if sched.events and 0 <= sched.active_event_index < len(sched.events):
            ev = sched.events[sched.active_event_index]
            box = layout.box()
            box.prop(ev, "epoch")
            box.prop(ev, "event_type")
            if ev.event_type == 'TARGET_CHANGE':
                box.prop(ev, "target_object")
            else:
                box.prop(ev, "value")

        row = layout.row()
        row.enabled = is_training_active()
        row.operator("nca.send_schedule", icon='EXPORT')

classes = (
    NCA_PT_MainPanel,
    NCA_PT_ConnectionPanel,
    NCA_PT_ControlPanel,
    NCA_PT_TargetPanel,
    NCA_PT_CellSettings,
    NCA_PT_PerceptionSettings,
    NCA_PT_UpdateSettings,
    NCA_PT_GridSettings,
    NCA_PT_TrainingSettings,
    NCA_UL_ScheduleEventList,
    NCA_PT_SchedulePanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)