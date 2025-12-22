import bpy
import torch
import time
from torch.nn import functional as F
from src.core import CellConfig, PerceptionConfig, UpdateConfig, GridConfig, Grid3D

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

    cell_cfg = CellConfig(
        hidden_channels=int(cell_props.hidden_channels),
        visible_channels=visible_channels_map.get(cell_props.visible_channels, 4),
        alive_threshold=float(cell_props.alive_threshold),
    )
    
    perc_cfg = PerceptionConfig(
        channel_groups=int(perception_props.channel_groups),
        kernel_radius=int(perception_props.kernel_radius),
    )
    
    upd_cfg = UpdateConfig(
        hidden_dim=int(update_props.hidden_dim),
        stochastic_update=bool(update_props.stochastic_update),
        fire_rate=float(update_props.fire_rate),
    )
    
    grid_cfg = GridConfig(
        size=(
            int(grid_props.grid_size[0]),
            int(grid_props.grid_size[1]),
            int(grid_props.grid_size[2])
        ),
    )
    
    return cell_cfg, perc_cfg, upd_cfg, grid_cfg

def selected_meshes_to_tensor():
    # TODO: Implement function to convert selected Blender meshes to PyTorch tensors
    return 

def visualize_tensor_in_blender(tensor):
    # TODO: Implement function to visualize PyTorch tensors in Blender
    return

class NCA_OT_RunSimulationOperator(bpy.types.Operator):
    bl_idname = "nca.run_simulation"
    bl_label = "Run Simulation"
    bl_description = "Execute the NCA simulation with current settings"

    _timer = None
    _is_running = False
    _model = None
    _optimizer = None
    _target = None
    _device = None
    _current_iter = 0
    _max_iters = 0
    _cell_cfg = None
    _start_time = None

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        
        if not self._is_running:
            self.cancel(context)
            return {'CANCELLED'}
        
        if self._current_iter >= self._max_iters:
            elapsed = time.time() - self._start_time
            self.report({'INFO'}, f"Training complete! {self._max_iters} iterations in {elapsed:.1f}s")
            self.cancel(context)
            return {'CANCELLED'}
        
        try:
            self.training_step()
            self._current_iter += 1
        except Exception as e:
            self.report({'ERROR'}, f"Error during simulation: {str(e)}")
            import traceback
            traceback.print_exc()
            self.cancel(context)
            return {'CANCELLED'}
        
        return {'PASS_THROUGH'}
    
    def training_step(self):
        self._optimizer.zero_grad()

        state = self._model.seed_center(1, self._device)
        c = 16
        state[:, :8, c:c+1, c:c+1, c:c+1] = torch.randn(1, 8, 1, 1, 1, device=self._device) * 0.05
        
        state = self._model(state, steps=64 + self._current_iter // 100)
        
        rgba_pred = state[:, -self._cell_cfg.visible_channels:]
        rgba_target = self._target[:, :self._cell_cfg.visible_channels]
        
        loss = F.mse_loss(
            rgba_pred[:, :, 10:22, 10:22, 10:22],
            rgba_target[:, :, 10:22, 10:22, 10:22]
        )
        
        loss.backward()
        self._optimizer.step()

        visualize_tensor_in_blender(rgba_pred.detach().cpu())

    def execute(self, context):
        if self._is_running:
            self.report({'WARNING'}, "Simulation already running!")
            return {'CANCELLED'}

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._target = selected_meshes_to_tensor().to(self._device)

        self._cell_cfg, perc_cfg, upd_cfg, grid_cfg = setup_configs(context)
        self._model = Grid3D(self._cell_cfg, perc_cfg, upd_cfg, grid_cfg).to(self._device)

        learning_rate = float(context.scene.nca_training_props.learning_rate)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)

        self._max_iters = int(context.scene.nca_training_props.num_epochs)
        self._current_iter = 0
        self._start_time = time.time()

        self._is_running = True
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window) # 10 FPS
        wm.modal_handler_add(self)

        return {'RUNNING_MODAL'}
    
    def cancel(self, context):
        if self._timer:
            wm = context.window_manager
            wm.event_timer_remove(self._timer)
            self._timer = None
        
        self._is_running = False
        self._current_iter = 0


classes = (
    NCA_OT_RunSimulationOperator,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)