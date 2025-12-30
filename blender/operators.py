import bpy
import threading
import time

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
            "channel_groups": int(perception_props.channel_groups),
            "kernel_radius": int(perception_props.kernel_radius),
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
        }
    }
    return cfg

def selected_meshes_to_tensor():
    # TODO: implement mesh -> tensor conversion
    return None

def visualize_tensor_in_blender(tensor):
    # TODO: implement Blender visualization logic (voxels, RGBA grids, etc.)
    return

class NCA_OT_RunSimulationOperator(bpy.types.Operator):
    bl_idname = "nca.run_simulation"
    bl_label = "Run Simulation"
    bl_description = "Send NCA configuration and mesh to trainer and visualize updates"

    _is_running = False
    _thread = None

    def execute(self, context):
        if self._is_running:
            self.report({'WARNING'}, "Simulation already running!")
            return {'CANCELLED'}

        # 1. Gather config and mesh tensor
        config = setup_configs(context)
        tensor = selected_meshes_to_tensor()

        # 2. Start a thread to handle communication with trainer
        self._is_running = True
        self._thread = threading.Thread(target=self.run_trainer_comm, args=(config, tensor))
        self._thread.start()

        return {'FINISHED'}

    def run_trainer_comm(self, config, tensor):
        # TODO: implement ZeroMQ / socket communication
        # if trainer not running: start subprocess
        # connect to trainer socket
        # send config and tensor
        # loop: receive snapshot -> visualize_tensor_in_blender(snapshot)
        while self._is_running:
            # placeholder loop for receiving snapshots
            time.sleep(0.1)
        self._is_running = False

    def cancel(self, context):
        self._is_running = False
        if self._thread:
            self._thread.join()
            self._thread = None

classes = (
    NCA_OT_RunSimulationOperator,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
