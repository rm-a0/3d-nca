"""
Blender UI panels for NCA training and inference.

Provides organized interface panels for:
- Connection management (host, port, connect/disconnect)
- Training control (start, stop, pause, resume)
- Target voxel editing and export
- Inference model loading and execution

Panels are organized hierarchically under a main NCA Settings panel.
"""

import bpy


class NCA_PT_BasePanel(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NCA"

    def draw(self, context):
        self.layout.use_property_split = True
        self.layout.use_property_decorate = False


class NCA_PT_MainPanel(NCA_PT_BasePanel):
    bl_label = "3D NCA Settings"
    bl_idname = "NCA_PT_main_panel"

    def draw(self, context):
        pass


class NCA_PT_ConnectionPanel(NCA_PT_BasePanel):
    bl_label = "Connection"
    bl_idname = "NCA_PT_connection_panel"
    bl_parent_id = "NCA_PT_main_panel"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 0

    def draw(self, context):
        from .operators import is_connected, is_server_busy

        super().draw(context)
        layout = self.layout
        conn = context.scene.nca_connection_props
        connected = is_connected()

        # Host / port - disabled while connected
        col = layout.column()
        col.enabled = not connected
        col.prop(conn, "host", text="Host")
        col.prop(conn, "port", text="Port")

        layout.separator()

        # Status row
        row = layout.row(align=True)
        if connected:
            row.label(text="Connected", icon="CHECKMARK")
            row.operator("nca.disconnect", text="Disconnect", icon="UNLINKED")
        else:
            row.label(text="Disconnected", icon="X")
            row.operator("nca.connect", text="Connect", icon="LINKED")

        if not connected:
            layout.label(
                text="Use ngrok host/port for remote training.",
                icon="INFO",
            )


class NCA_PT_ControlPanel(NCA_PT_BasePanel):
    bl_label = "Training Control"
    bl_idname = "NCA_PT_control_panel"
    bl_parent_id = "NCA_PT_main_panel"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 1

    def draw(self, context):
        from .operators import is_connected, is_server_busy

        super().draw(context)
        layout = self.layout
        connected = is_connected()
        busy = is_server_busy()

        col = layout.column(align=True)
        col.enabled = connected

        row = col.row(align=True)
        row.enabled = not busy
        row.operator("nca.start_training", icon="PLAY")

        row = col.row(align=True)
        row.enabled = busy
        row.operator("nca.stop_training", icon="SNAP_FACE")
        row.operator("nca.pause_training", icon="PAUSE")
        row.operator("nca.resume_training", icon="PLAY")


class NCA_PT_TargetPanel(NCA_PT_BasePanel):
    bl_label = "Target"
    bl_idname = "NCA_PT_target_panel"
    bl_parent_id = "NCA_PT_main_panel"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 2

    def draw(self, context):
        from .operators import is_server_busy

        super().draw(context)
        layout = self.layout
        target = context.scene.nca_target_props
        cell = context.scene.nca_cell_props
        editable = not is_server_busy()

        layout.prop(target, "cell_size")

        # Color mode only makes sense when exporting RGBA channels
        if cell.visible_channels == "RGBA":
            col = layout.column(align=True)
            col.enabled = editable
            col.prop(target, "color_mode", text="Color Mode")
            if target.color_mode == "TEXTURE":
                col.label(
                    text="Tip: images are cached before the loop.",
                    icon="INFO",
                )
                col.label(
                    text="Progress shown in the status bar.",
                    icon="INFO",
                )

        layout.separator()

        row = layout.row(align=True)
        row.enabled = editable
        row.prop(target, "source_object", text="Source Mesh")
        row.operator("nca.clear_target_voxels", text="", icon="X")

        if target.is_voxelized:
            box = layout.box()
            box.label(text="Target ready", icon="CHECKMARK")
            if target.source_object:
                box.label(text=f"Source: {target.source_object.name}")
            box.label(text=f"Voxels: {target.voxel_count}")
            row = box.row()
            row.enabled = editable
            row.operator("nca.export_target", text="Export .npz")
        else:
            layout.box().label(text="No target voxelized", icon="ERROR")


class NCA_PT_InferencePanel(NCA_PT_BasePanel):
    bl_label = "Run Trained Model"
    bl_idname = "NCA_PT_inference_panel"
    bl_parent_id = "NCA_PT_main_panel"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 3

    def draw(self, context):
        from .operators import is_connected, is_server_busy

        super().draw(context)
        layout = self.layout
        infer = context.scene.nca_inference_props
        connected = is_connected()
        busy = is_server_busy()

        col = layout.column()
        col.enabled = not busy
        col.prop(infer, "model_path", text="Model (.pt)")
        col.prop(infer, "steps_per_phase", text="Steps")
        col.prop(infer, "broadcast_every", text="Broadcast Every")
        col.prop(infer, "send_delay_ms", text="Send Delay (ms)")

        layout.separator()
        row = layout.row()
        row.enabled = connected and not busy and bool(infer.model_path)
        row.operator("nca.run_model", icon="PLAY")

        row = layout.row()
        row.enabled = connected and busy
        row.operator("nca.stop_training", text="Stop Inference", icon="SNAP_FACE")


class NCA_PT_CellSettings(NCA_PT_BasePanel):
    bl_label = "Cell Settings"
    bl_idname = "NCA_PT_cell_settings"
    bl_parent_id = "NCA_PT_main_panel"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 4

    def draw(self, context):
        from .operators import is_server_busy

        super().draw(context)
        layout = self.layout
        layout.enabled = not is_server_busy()
        cell = context.scene.nca_cell_props
        layout.prop(cell, "hidden_channels", text="Hidden Channels")
        layout.prop(cell, "visible_channels", text="Visible Channels")
        layout.prop(cell, "alive_threshold", text="Alive Threshold")


class NCA_PT_PerceptionSettings(NCA_PT_BasePanel):
    bl_label = "Perception Settings"
    bl_idname = "NCA_PT_perception_settings"
    bl_parent_id = "NCA_PT_main_panel"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 5

    def draw(self, context):
        from .operators import is_server_busy

        super().draw(context)
        layout = self.layout
        layout.enabled = not is_server_busy()
        perc = context.scene.nca_perception_props
        layout.prop(perc, "kernel_radius", text="Kernel Radius")
        layout.prop(perc, "channel_groups", text="Channel Groups")


class NCA_PT_UpdateSettings(NCA_PT_BasePanel):
    bl_label = "Update Settings"
    bl_idname = "NCA_PT_update_settings"
    bl_parent_id = "NCA_PT_main_panel"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 6

    def draw(self, context):
        from .operators import is_server_busy

        super().draw(context)
        layout = self.layout
        layout.enabled = not is_server_busy()
        upd = context.scene.nca_update_props
        layout.prop(upd, "hidden_dim", text="Hidden Dimension")
        layout.prop(upd, "stochastic_update", text="Stochastic Update")
        layout.prop(upd, "fire_rate", text="Fire Rate")


class NCA_PT_GridSettings(NCA_PT_BasePanel):
    bl_label = "Grid Settings"
    bl_idname = "NCA_PT_grid_settings"
    bl_parent_id = "NCA_PT_main_panel"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 7

    def draw(self, context):
        from .operators import is_server_busy

        super().draw(context)
        layout = self.layout
        layout.enabled = not is_server_busy()
        grid = context.scene.nca_grid_props
        layout.prop(grid, "grid_size", text="Grid Size")
        layout.prop(grid, "grid_offset", text="Grid Offset")


class NCA_PT_TrainingSettings(NCA_PT_BasePanel):
    bl_label = "Training Settings"
    bl_idname = "NCA_PT_training_settings"
    bl_parent_id = "NCA_PT_main_panel"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 8

    def draw(self, context):
        from .operators import is_server_busy

        super().draw(context)
        layout = self.layout
        layout.enabled = not is_server_busy()
        train = context.scene.nca_training_props
        layout.prop(train, "learning_rate", text="Learning Rate")
        layout.prop(train, "batch_size", text="Batch Size")
        layout.prop(train, "num_epochs", text="Epochs")


class NCA_UL_ScheduleEventList(bpy.types.UIList):
    bl_idname = "NCA_UL_schedule_event_list"

    def draw_item(
        self, context, layout, data, item, icon, active_data, active_property, index
    ):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            row.label(text="Now" if item.epoch == -1 else str(item.epoch))
            row.prop(item, "event_type", text="", emboss=False)
            if item.event_type == "TARGET_CHANGE":
                row.prop(item, "target_object", text="", emboss=False)
            else:
                row.prop(item, "value", text="", emboss=False)


class NCA_PT_SchedulePanel(NCA_PT_BasePanel):
    bl_label = "Schedule"
    bl_idname = "NCA_PT_schedule_panel"
    bl_parent_id = "NCA_PT_main_panel"
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 9

    def draw(self, context):
        from .operators import is_connected, is_server_busy

        super().draw(context)
        layout = self.layout
        sched = context.scene.nca_schedule_props

        row = layout.row()
        row.template_list(
            "NCA_UL_schedule_event_list",
            "",
            sched,
            "events",
            sched,
            "active_event_index",
            rows=3,
        )
        col = row.column(align=True)
        col.operator("nca.add_schedule_event", icon="ADD", text="")
        col.operator("nca.remove_schedule_event", icon="REMOVE", text="")

        if sched.events and 0 <= sched.active_event_index < len(sched.events):
            ev = sched.events[sched.active_event_index]
            box = layout.box()
            box.prop(ev, "epoch")
            box.prop(ev, "event_type")
            if ev.event_type == "TARGET_CHANGE":
                box.prop(ev, "target_object")
            else:
                box.prop(ev, "value")

        row = layout.row()
        row.enabled = is_connected() and is_server_busy()
        row.operator("nca.send_schedule", icon="EXPORT")


# --- Registration ---

classes = (
    NCA_PT_MainPanel,
    NCA_PT_ConnectionPanel,
    NCA_PT_ControlPanel,
    NCA_PT_TargetPanel,
    NCA_PT_InferencePanel,
    NCA_PT_CellSettings,
    NCA_PT_PerceptionSettings,
    NCA_PT_UpdateSettings,
    NCA_PT_GridSettings,
    NCA_PT_TrainingSettings,
    NCA_UL_ScheduleEventList,
    NCA_PT_SchedulePanel,
)


def register():
    """Register Blender panel classes."""
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    """Unregister Blender panel classes in reverse order."""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)