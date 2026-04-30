# 3D Neural Cellular Automata

A modular PyTorch framework for training **3D Neural Cellular Automata** — self-organizing systems that learn to grow volumetric structures through local, parallel cell interactions.

---

## Architecture Overview

### Package Structure

```
src/
├── core/           # NCA engine and training logic
│   ├── cell.py           # Cell state and configuration
│   ├── grid.py           # 3D grid simulation
│   ├── perception.py     # 3D convolution kernels for local perception
│   ├── update.py         # Learnable update rules (MLP)
│   ├── nca_model.py      # High-level NCA wrapper
│   ├── schedule.py       # Training event scheduling
│   └── runners/          # Training orchestrators (Morph, Regen)
├── viz/            # Visualization backends
│   ├── slice_mpl.py      # 2D slice plotting (matplotlib)
│   ├── volume_mpl.py     # 3D volume plotting (matplotlib)
│   └── volume_pv.py      # 3D volume rendering (pyvista)
├── io/             # Mesh I/O utilities
│   └── object_converter.py  # Voxelization (trimesh)
└── server/         # TCP training server (for Blender integration)
    ├── server.py         # Socket server
    ├── trainer.py        # Background training thread
    ├── protocol.py       # Message protocol
    └── logger.py         # Training logger
```

### Core Concepts

**1. Cell State**
- Each voxel stores a multi-channel state vector `[alive, r, g, b, h1, h2, ...]`
- Channel 0: alive mask (binary, learned)
- Channels 1-3: visible RGB color
- Channels 4+: hidden channels for internal computation

**2. Perception**
- 3D Sobel filters detect spatial gradients in each channel
- Creates perception vectors from local neighborhood (3×3×3 kernel)
- Differentiable convolution operations

**3. Update Rule**
- Learnable MLP maps `perception → state_delta`
- Cell updates conditionally: only if cell or neighbor is alive
- Stochastic masking prevents synchronization artifacts

**4. Training Loop**
- Start from seed (single alive cell or small region)
- Apply NCA update for N steps
- Compute loss vs target voxel mesh (alpha + color + overflow)
- Backprop through time to train the update rule

---

## Installation

### Option 1: Pip Install (Minimal Package)

For using the NCA library in your own projects:

```bash
git clone https://github.com/rm-a0/3d-nca
cd 3d-nca

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install core only (torch, numpy, tqdm)
pip install -e .

# Or with visualization support
pip install -e ".[viz]"

# Or with 3D mesh I/O
pip install -e ".[io]"

# Or all optional features
pip install -e ".[all]"

# For development (includes pytest, black, ruff, etc.)
pip install -e ".[dev]"
```

**Optional Dependency Groups:**
- `viz`: matplotlib, pyvista (for visualization)
- `io`: trimesh (for mesh voxelization)
- `all`: all optional features
- `dev`: development tools (pytest, black, isort, mypy, ruff, pre-commit)

### Option 2: Conda (Full Development Environment)

For developing this project with all tools (notebooks, scripts, GPU support):

```bash
# Create conda environment with CUDA support
conda env create -f conda_env.yml
conda activate nca3d

# Install the package in editable mode
pip install -e .
```

**What's included in `conda_env.yml`:**
- PyTorch with CUDA 12.1
- All visualization dependencies (matplotlib, pyvista)
- Mesh I/O (trimesh)
- Jupyter ecosystem (for notebooks/)
- Plotting tools (pandas for scripts/)
- Development tools (pytest, black, ruff, mypy, pre-commit)

**What's in `environment.yml`:**
- Minimal conda environment for pip package install
- Use this if you want conda for PyTorch but pip for everything else

---

## Quick Start

### Basic Training Example

```python
from src import NCAModel, NCAConfig
from src.core.runners import MorphRunner
from src.io import obj_to_tensor
import torch

# Load target mesh as voxel tensor
target = obj_to_tensor("path/to/mesh.obj", grid_size=32)

# Configure NCA
config = NCAConfig(
    grid_size=32,
    n_channels=16,  # 4 visible (alive, R, G, B) + 12 hidden
    hidden_size=96,
    n_perception_kernels=3,
)

# Create model and runner
model = NCAModel(config)
runner = MorphRunner(
    model=model,
    target=target,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

# Train
for snapshot in runner.train(epochs=1000, steps_per_epoch=32):
    if snapshot.epoch % 100 == 0:
        print(f"Epoch {snapshot.epoch}, Loss: {snapshot.loss:.4f}")
```

### Visualization

```python
from src.viz import show_volume_alpha_pv

# Visualize current state
state = runner.grid.state.detach().cpu()
show_volume_alpha_pv(state, opacity_channel=0)
```

---

## Training Modes

### 1. Morphogenesis (Default)
Grow structure from a single seed cell.

```python
from src.core.runners import MorphRunner

runner = MorphRunner(model, target, device)
```

### 2. Regeneration
Damage the structure mid-training and learn to regenerate.

```python
from src.core.runners import RegenRunner

runner = RegenRunner(
    model, target, device,
    damage_start_epoch=500,
    damage_radius=8,
)
```

---

## Development Workflow

### Running Tests
```bash
pytest tests/
pytest --cov=src tests/  # with coverage
```

### Code Formatting
```bash
black src/ tests/
isort src/ tests/
ruff check src/ tests/
```

### Pre-commit Hooks
```bash
pre-commit install
pre-commit run --all-files
```

### Using Notebooks
Jupyter notebooks in `notebooks/` demonstrate:
- Basic functionality
- Alpha-only training (structure without color)
- RGBA training (structure + color)
- Advanced experiments

Start JupyterLab:
```bash
jupyter lab
```

### Plotting Scripts
Analyze training logs:
```bash
python scripts/plot_loss.py runs/experiment_name/loss.csv
python scripts/plot_loss_phased.py runs/experiment_name/loss.csv
```

---

## Blender Integration

A TCP server allows real-time visualization in Blender during training.

**1. Start training server:**
```python
from src.server import NCAServer

server = NCAServer(model, target, host="localhost", port=8765)
server.start()
```

**2. Launch Blender with addon:**
- Install the addon from `blender/`
- Connect to `localhost:8765`
- Watch the NCA grow in real-time

The server runs training in a background thread and streams state updates to Blender without blocking.

---

## How It Works

### Forward Pass
1. **Perceive**: 3D Sobel filters extract gradients → perception vectors
2. **Update**: MLP maps perception → state delta
3. **Mask**: Only update alive cells (and their neighbors)
4. **Apply**: Add delta to current state

### Training
1. Start from seed configuration
2. Run NCA for N steps (curriculum: 8→32 or 16→64)
3. Compute loss:
   - **Alpha loss**: encourage structure emergence
   - **Color loss**: match RGB channels
   - **Overflow loss**: suppress background
4. Backprop through differentiable grid ops
5. Update MLP weights

### Key Techniques
- **Gradient checkpointing**: reduce memory for long rollouts
- **Stochastic masking**: prevent grid-wide synchronization
- **Alive masking**: binary mask via max-pool over neighbors
- **Pool-based training**: sample from pool of growth trajectories

---

## Project Files

| File/Directory | Purpose |
|---|---|
| `src/` | Installable Python package (NCA library) |
| `tests/` | Pytest test suite |
| `notebooks/` | Jupyter notebooks with examples |
| `scripts/` | Utility scripts (plotting, analysis) |
| `blender/` | Blender addon for real-time visualization |
| `runs/` | Training outputs (checkpoints, logs, renders) |
| `pyproject.toml` | Package metadata and pip dependencies |
| `conda_env.yml` | Full conda environment (development) |
| `environment.yml` | Minimal conda environment (pip package) |

---

## Environment Files Explained

### `conda_env.yml` — Development Environment
- **Purpose**: Full development setup with all tools
- **Use when**: Developing this project, running notebooks, using Blender addon
- **Includes**: PyTorch+CUDA, Jupyter, pandas, trimesh, pyvista, dev tools
- **Setup**: `conda env create -f conda_env.yml && conda activate nca3d`

### `environment.yml` — Minimal Package Environment
- **Purpose**: Minimal environment for using the library
- **Use when**: Installing as a dependency or testing pip install
- **Includes**: PyTorch+CUDA, numpy, tqdm (core only)
- **Setup**: `conda env create -f environment.yml && pip install -e ".[all]"`

### `pyproject.toml` — Pip Package Definition
- **Purpose**: Defines the installable Python package
- **Use when**: `pip install` or building wheels
- **Includes**: Minimal core deps + optional groups (viz, io, dev)

**TL;DR:**
- **Developing this project?** → Use `conda_env.yml`
- **Using as a library?** → Use `pip install -e ".[all]"`
- **Want conda for PyTorch only?** → Use `environment.yml` + pip

---

## License

MIT License — see [LICENSE](LICENSE)

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{3d-nca,
  author = {Michal Repcik},
  title = {3D Neural Cellular Automata Framework},
  year = {2026},
  url = {https://github.com/rm-a0/3d-nca}
}
```
