# 3D Neural Cellular Automata

A modular PyTorch framework for training **3D Neural Cellular Automata** — self-organizing systems that grow volumetric structures from learned update rules. Given a target voxel mesh, the system learns a differentiable cellular automaton to reproduce it through local, parallel cell interactions.

Try the interactive Blender integration: Launch Blender with the addon, connect the training server, and watch the NCA grow your target in real-time.

---

## Core Features

| Feature | Description |
|---|---|
| **Modular NCA Engine** | Perception kernels, learnable update rules, differentiable grid simulation with gradient checkpointing |
| **3D Voxel Space** | Arbitrary grid size; supports multi-channel state (hidden + visible) with alive masking via neighbor max-pool |
| **Curriculum Learning** | Automatic ramping of training steps per epoch for stable convergence (8→32 steps, 16→64 steps) |
| **Multi-part Loss** | Alpha (emergence), color (fidelity), overflow (background suppression) over batched pool |
| **Real-time Server** | TCP socket protocol; background thread orchestrates training, foreground sends/receives state updates |
| **Blender Addon** | Visualize NCA growth live during training; voxel rendering with pygame-based UI; no polling overhead |
| **Flexible I/O** | Mesh voxelization via trimesh; checkpoint save/load; exports to external formats |

---

## Quick Start

### 1. Setup

```bash
git clone https://github.com/rm-a0/3d-nca
cd 3d-nca
```

Choose your environment manager:

**Conda (one-click, GPU-aware):**
```bash
conda env create -f conda_env.yml
conda activate nca3d
```

**Poetry (developers):**
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install poetry
poetry install --with dev
```

### 2. Run Training (Jupyter Notebook)

```bash
code notebooks/01_basic_functionality.ipynb
```

Or in Python:

```python
from src.core.nca_model import NCAModel
from src.io.object_converter import obj_to_tensor
import numpy as np

# Load target mesh
target = obj_to_tensor("path/to/mesh.obj", grid_size=(100, 100, 100), mode="rgba")

# Create and train model
model = NCAModel()
model.train(target.numpy(), epochs=5000)

# Export result
state = model.export_state()  # shape (D, H, W, C)
```

### 3. Real-time Blender Visualization (Optional)

1. Open Blender
2. Edit → Preferences → Add-ons → Install from file → select `blender/` folder
3. Start training server: `python -c "from src.server.server import serve_forever; serve_forever()"`
4. In Blender, click "Connect to Server" in the NCA panel
5. Watch the NCA grow in real-time

---

## Requirements

- **Python** >= 3.11
- **PyTorch** >= 2.0 (CPU or CUDA)
- **NumPy, SciPy** — numerical computing
- **trimesh** — mesh voxelization
- **Blender** >= 3.0 (optional, for visualization addon)

| Package | Purpose |
|---|---|
| `torch` | Neural network training, autodiff, Conv3d |
| `numpy` | Tensor I/O, data processing |
| `trimesh` | Load mesh files, voxelization |
| `scipy` | Sparse matrices, scientific ops |
| `tqdm` | Progress bars during training |

---

## Installation & Run

### Local Training

**Data Preparation:**
Place your target mesh files in `assets/`:

```bash
assets/
├── donut/
│   └── donut.obj
├── your_mesh/
│   └── your_mesh.obj
└── ...
```

**Train from Jupyter:**
```bash
code notebooks/01_basic_functionality.ipynb
```

or from CLI:
```bash
python -c "
from src.core.nca_model import NCAModel
from src.io.object_converter import obj_to_tensor

target = obj_to_tensor('assets/donut/donut.obj', mode='rgba')
model = NCAModel(config={'training': {'num_epochs': 5000}})
model.train(target.numpy())
"
```

### Server Mode (with Blender Integration)

Terminal 1 — Start training server:
```bash
python -c "from src.server.server import NCAServer; s = NCAServer(port=9999); s.serve_forever()"
```

Terminal 2 — Blender connects via addon:
- Start Blender
- Click "Connect" in the NCA panel → connects to `localhost:9999`
- Server spawns background training thread
- Training loop broadcasts state updates every ~100ms

The server accepts commands:
- `start`: Begin training with optional schedule
- `pause`: Pause training
- `resume`: Resume from pause
- `stop`: Stop and join training thread
- `update_schedule`: Inject runtime parameter changes (LR, batch size, target swap)

---

## Project Structure

```
3d-nca/
│
├── src/
│   ├── core/                      # Core NCA algorithm
│   │   ├── cell.py                # CellState: channels, alive mask logic
│   │   ├── grid.py                # Grid3D: main NCA simulator (forward pass)
│   │   ├── perception.py          # Perception3D: 3×3×3 depthwise convolutions
│   │   ├── update.py              # UpdateRule: learnable MLP (Conv + GroupNorm + tanh)
│   │   ├── nca_model.py           # NCAModel: high-level user API
│   │   ├── runner.py              # NCARunner: training loop, curriculum, pool management
│   │   └── schedule.py            # Schedule: thread-safe event dispatch (target changes, LR)
│   │
│   ├── server/                    # Training server & Blender integration
│   │   ├── server.py              # NCAServer: TCP socket listener
│   │   ├── trainer.py             # NCATrainer: background thread orchestration
│   │   ├── protocol.py            # Wire protocol: JSON + base64 tensor serialization
│   │   └── logger.py              # Logging: epoch metrics, checkpoints, state broadcasts
│   │
│   ├── io/                        # Input/output utilities
│   │   └── object_converter.py    # obj_to_tensor: mesh voxelization
│   │
│   └── viz/                       # Visualization backends
│       ├── volume_pv.py           # PyVista 3D volume rendering
│       ├── volume_mpl.py          # Matplotlib isosurface rendering
│       └── slice_mpl.py           # Matplotlib slice viewer
│
├── blender/                       # Blender add-on (Python 3.11 bundled)
│   ├── __init__.py
│   ├── operator.py                # UI operators: start/stop/pause
│   ├── panel.py                   # Blender UI panel in N-sidebar
│   ├── client.py                  # Socket client: connection management
│   ├── protocol.py                # Wire protocol (shared with server)
│   ├── voxel_utils.py             # Voxel→mesh conversion for Blender viewport
│   └── properties.py              # Configuration parameters (grid size, channels)
│
├── notebooks/                     # Jupyter experiments
│   ├── 01_basic_functionality.ipynb         # Simple NCA training
│   ├── 02_sphere_training_alpha.ipynb       # Sphere with alpha only
│   ├── 03_donut_training_alpha.ipynb        # Donut with alpha only
│   ├── 04_donut_training_rgba.ipynb         # Full RGBA channel training
│   └── colab_run_*.py             # Google Colab entry points
│
├── scripts/                       # Utility scripts
│   ├── plot_loss.py               # Loss curve visualization
│   └── plot_loss_phased.py        # Phase-wise loss breakdown
│
├── runs/                          # Training run outputs
│   ├── run_000/
│   │   ├── loss.csv               # Per-epoch metrics
│   │   ├── meta.json              # Run metadata (config, timestamp)
│   │   └── checkpoints/           # Model weights at intervals
│   └── ...
│
├── assets/                        # Sample mesh files
│   └── donut/donut.obj & .mtl
│
├── documentation/                 # Thesis & figures
│   ├── thesis/
│   │   └── 3d-nca.tex             # LaTeX thesis source
│   └── README.md                  # Documentation index
│
├── tests/                         # Unit & integration tests (future)
├── conda_env.yml                  # Reproducible Conda environment
├── pyproject.toml                 # Poetry metadata, dependencies
└── README.md                      # This file
```

---

## Core Concepts

### Tensor Shapes: Two Conventions

The codebase uses **two tensor shape conventions** for good reason:

#### External I/O Format: `(D, H, W, C)` — Channels-Last

Used for:
- **Input targets** from mesh voxelization
- **Blender visualization** (intuitive for 3D data)
- **Checkpoint files** and exports
- **Wire protocol** messages

**Format**: `(Depth, Height, Width, Channels)`
- First 3 dims are spatial extents
- Last dim is features (RGBA, alpha-only, etc.)

Example:
```python
target = obj_to_tensor("mesh.obj", grid_size=(100, 100, 100), mode="rgba")
# → external shape (100, 100, 100, 4) if you extract it: target.numpy()[0].transpose(1,2,3,0)
```

#### Internal NCA Format: `(B, C, D, H, W)` — Channels-First, Batched

Used for:
- **All NCA computations** inside forward pass
- **PyTorch Conv3d operations** (standard)
- **Training pool** of candidate states
- **Gradients and backprop**

**Format**: `(Batch, Channels, Depth, Height, Width)`
- Batch dimension enables parallel training
- Channels-first is PyTorch's native format
- Crucial for GPU efficiency

Example:
```python
state = torch.randn(batch=4, channels=20, depth=32, height=32, width=32)  # (4, 20, 32, 32, 32)
state = model(state, steps=16)  # stays (4, 20, 32, 32, 32)
```

#### Conversions

| Operation | From | To | Code |
|-----------|------|----|----|
| Load target | (D,H,W,C) | (1,C,D,H,W) | `np.transpose(target, (3,0,1,2)).unsqueeze(0)` |
| Export state | (1,C,D,H,W) | (D,H,W,C) | `state[0].permute(1,2,3,0).numpy()` |
| Schedule event | (D,H,W,C) | (1,C,D,H,W) | Same as load |

**Key Rule**: Outside training loop, always use `(D,H,W,C)`. Inside training loop, always use `(B,C,D,H,W)`.

---

## NCA Algorithm

### Cell State

Each voxel is a vector of length `C = hidden_channels + visible_channels` (default: 16 + 4 = 20).

- **Hidden channels** (16): Internal memory, not rendered
- **Visible channels** (4): RGBA output
  - R, G, B: Color
  - A: Alpha (alive signal)

### Alive Masking

Cells update only if they or a neighbor is "alive" (alpha > threshold):

```python
alpha = state[:, -1:, ...]  # Extract alpha [B, 1, D, H, W]
pooled = max_pool3d(alpha, kernel=3, padding=1)  # 3×3×3 neighborhood
alive = pooled > 0.1  # Boolean mask [B, 1, D, H, W]
state_updated = state * alive  # Only alive cells update
```

### Perception

Fixed depthwise 3×3×3 kernels (no learning):
- **Identity**: Current cell
- **6-neighbor sum**: Adjacent cells (±x, ±y, ±z)
- **Laplacian**: Difference from neighborhood

→ Input to update rule: 3× perceived state

### Update Rule

Learnable MLP applied 1×1 spatially:

```
perception (3×20) → Conv(3×20 → 64) → GroupNorm(64) → ReLU 
                  → Conv(64 → 20) → tanh × 0.1
                  → stochastic alive gating (optional)
```

Output: delta state, superimposed on input (residual connection).

### Growth Through Unrolling

Training loop:
1. Sample batch from state pool
2. Forward pass: `state = model(state, steps=n)` where `n` increases with curriculum
3. Compute 3-part loss (see below)
4. Backprop through entire unroll
5. Update model weights
6. Replace worst pool members with reseeded fresh cells

---

## Training

### Runner (Single Process)

`NCARunner` orchestrates one training session:

```python
runner = NCARunner()
runner.init(config, target)

for epoch in range(1, num_epochs + 1):
    metrics = runner._step()
    # metrics = {loss_alpha, loss_color, loss_overflow, loss_total, best_np}
    schedule.check_and_execute(epoch, runner)  # Handle events
    yield metrics
```

Each step:
1. **Curriculum**: `steps = random.randint(step_min, step_max)` where ranges ramp over 2000 epochs
2. **Sample batch**: `batch = [pool[i] for i in random.choices(0, pool_size, batch_size)]`
3. **Replace worst**: `pool[worst_idx] = model.seed_center()` (restart with seed if too high loss)
4. **Forward**: `state = model(state, steps)`
5. **Loss**:
   - Alpha: `MSE(pred_alpha, target_alpha)` — emergence
   - Color: `MSE(pred_rgb * target_alpha, target_rgb * target_alpha)` — fidelity in foreground
   - Overflow: `MSE(pred_alpha * (1 - target_alpha), 0)` — suppression in background
6. **Backprop**: `backward()`, `clip_grad_norm(1.0)`, `optimizer.step()`
7. **Checkpoint**: Save model if improvement

### Server (Multi-threaded)

`NCAServer` and `NCATrainer` enable real-time control:

```
Main Thread (socket listener)
  ↓
  Receives commands (start, pause, resume, stop, update_schedule)
  ↓
  → Background Thread (training loop)
     Calls runner.train(schedule)
     Every N epochs, broadcasts state to Blender
     Pauses on command
```

**Thread Safety**: Schedule uses `threading.Lock` to protect concurrent reads/writes from main thread and background training loop.

---

## Server & Protocol

### Wire Protocol

**Message Format**: `[4-byte big-endian length] [JSON payload]`

**Serialization**: Tensors encoded as base64 + shape metadata

```json
{
  "type": "start",
  "config": { "cell": {...}, "training": {...} },
  "target": {
    "b64": "base64-encoded-float32-bytes",
    "shape": [100, 100, 100, 4]
  }
}
```

### Events

Schedule events allow runtime parameter changes without stopping training:

| Event | Effect |
|-------|--------|
| `LEARNING_RATE` | Modify optimizer LR at epoch N |
| `BATCH_SIZE` | Adjust pool sampling size |
| `ALPHA_WEIGHT` | Rebalance loss terms |
| `TARGET_CHANGE` | Swap target mesh mid-training |

Thread-safe: main thread updates schedule, background loop reads with lock.

---

## Blender Integration

### Addon Features

1. **Connection Manager**: Connect/disconnect from training server
2. **Parameter Panel**: Configure grid size, channels, learning rate
3. **Control Buttons**: Start, pause, resume, stop training
4. **Voxel Visualization**: Real-time volumetric rendering in viewport
5. **Progress Display**: Current epoch, loss, pool statistics

### Workflow

```
Blender UI (main thread) ←→ Socket Client (bg thread)
    ↓                          ↓
[UI updates]          [receives state every 100ms]
[sends commands]      [thread lock protects state]
                             ↓
                      Training Server
```

### Performance

- **Update latency**: ~100ms per frame (soft real-time)
- **Render**: Efficient voxel→mesh conversion via marching cubes
- **Memory**: State only stored once on GPU/CPU; broadcast as base64 over socket

---

## Algorithms & Loss Functions

### Curriculum Learning

Automatically ramp training complexity:

| Epoch Ranges | Min Steps | Max Steps | Rationale |
|---|---|---|---|
| 0–500 | 8 | 16 | Start short: low computational cost, fast feedback |
| 500–1500 | 16 | 32 | Middle: longer unrolls, stable gradients |
| 1500–2000 | 32 | 64 | Late: long horizons for fine structure |

Random step count sampled per batch to encourage robustness.

### Multi-Part Loss

Weighted combination of three terms:

```
L_total = w_alpha * L_alpha + w_color * L_color + w_overflow * L_overflow
        = 4.0 * MSE(α_pred, α_target)
        + 1.0 * MSE(rgb_pred ⊙ mask, rgb_target ⊙ mask)
        + 2.0 * MSE(α_pred ⊙ ¬mask, 0)
```

- **L_alpha** (weight 4.0): Emergence — encourage alpha to match target
- **L_color** (weight 1.0): Fidelity — color should match *only where target has alpha > 0.1*
- **L_overflow** (weight 2.0): Background suppression — discourage predictions outside target boundary

### Gradient Checkpointing

Memory-efficient for long unrolls:

```python
if use_checkpointing and state.requires_grad:
    state = torch.utils.checkpoint.checkpoint(
        self.step, state, use_reentrant=False
    )
```

Trades compute for memory: recompute forward activations during backward instead of storing them. ~95% memory savings for 64-step unrolls.

---

## Development

### Running Tests (if test suite exists)

```bash
pytest tests/ -xvs
```

### Adding a New Strategy / Feature

1. Create file in appropriate module
2. Follow naming convention: `noun_verb.py` or `strategy_name.py`
3. Import in parent `__init__.py` if needed
4. Docstring with shape conventions (`(B,C,D,H,W)` or `(D,H,W,C)`)
5. Test locally in notebook before integrating

### Code Style

- Type hints on public methods
- Docstrings with Args/Returns/Raises
- Inline comments for non-obvious logic
- Class constants in CAPS
- Follow existing imports pattern

---

## TODO / Known Limitations

- **Docstring standardization**: Some older modules lack detailed docstrings; standardize across entire codebase
- **Unit tests**: No pytest suite yet; add tests for core ops (grid.step, alive_mask)
- **Logging**: Currently uses print + CSV; consider structured logging (loguru/structlog)
- **Memory profiling**: Profile peak memory during large grid sizes to enable better checkpointing heuristics
- **Data pipeline**: Currently single-mesh per run; could enable multi-target curriculum
- **Async I/O**: Blender addon uses polling; consider websocket for lower latency
- **Type annotations**: Add full type hints to remaining files (perception.py, update.py)
- **Error handling**: Server protocol needs better error messages and recovery
- **CI/CD**: Add GitHub Actions for lint (ruff), type check (mypy), test runs
- **Documentation**: Add more notebook examples (multi-material, symmetry constraints, etc.)
