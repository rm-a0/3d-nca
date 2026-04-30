# 3D Neural Cellular Automata

A modular PyTorch framework for training 3D Neural Cellular Automata on volumetric targets.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)

---

## Overview

3D-NCA is a PyTorch library for training 3D Neural Cellular Automata on volumetric targets. It provides a composable model API, two built-in training runners (morphogenesis and regeneration), visualization helpers for matplotlib and pyvista, and a TCP server for real-time Blender integration.

For theoretical background and architecture details see the [thesis](documentation/thesis/) or the [API docs](documentation/docs/).

---

## Requirements

- Python 3.11+
- PyTorch 2.0+
- CUDA-capable GPU (optional, but strongly recommended)

---

## Installation

Three setup paths depending on your use case:

| Goal | Environment file | Command |
|---|---|---|
| Use as a library | `pyproject.toml` | `pip install -e ".[all]"` |
| Develop this project | `conda_env.yml` | `conda env create -f conda_env.yml` |
| Conda + pip hybrid | `environment.yml` | `conda env create -f environment.yml` |

### Option 1 - pip (library users)

```bash
git clone https://github.com/rm-a0/3d-nca
cd 3d-nca

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e .                 # core only: torch, numpy, tqdm
```

Install optional feature groups as needed:

```bash
pip install -e ".[viz]"          # matplotlib + pyvista
pip install -e ".[io]"           # trimesh (mesh voxelization)
pip install -e ".[dev]"          # pytest, black, ruff, mypy, pre-commit
pip install -e ".[all]"          # viz + io
```

> **CUDA (pip only):** Install a CUDA-enabled PyTorch build before `pip install -e .`:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> pip install -e ".[all]"
> ```

### Option 2 - conda (full development)

Includes PyTorch+CUDA 12.1, Jupyter, pandas, trimesh, pyvista, wandb, and all dev tools.

```bash
conda env create -f conda_env.yml
conda activate nca3d
pip install -e .
```

### Option 3 - conda + pip hybrid

Conda handles PyTorch+CUDA; pip handles everything else.

```bash
conda env create -f environment.yml
conda activate nca3d
pip install -e ".[viz,io,dev]"
```

See [ENVIRONMENT_GUIDE.md](ENVIRONMENT_GUIDE.md) for a full comparison of all three options.

---

## Quick Start

`NCAModel` is a standard `nn.Module` -- drop it into any PyTorch training loop.

```python
import torch
import torch.nn.functional as F
from src import NCAModel, NCAConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

config = NCAConfig(
    grid_size=(32, 32, 32),
    hidden_channels=8,
    visible_channels=1,
    update_hidden_dim=64,
)
model = NCAModel(config).to(device)

# Spherical target -- shape [1, 1, D, H, W]
D, H, W = config.grid_size
axes = [torch.linspace(-1, 1, s) for s in (D, H, W)]
grid = torch.stack(torch.meshgrid(*axes, indexing="ij"))
target = (grid.norm(dim=0) < 0.6).float().unsqueeze(0).unsqueeze(0).to(device)

optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
for step in range(1, 2001):
    state = model.seed_center(batch_size=1, device=device)
    optim.zero_grad()
    state = model(state, steps=32)
    loss = F.mse_loss(state[:, -1:], target)   # last channel = alpha/alive
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()
    if step % 500 == 0:
        print(f"step {step:4d}  loss {loss.item():.4f}")
```

### Load a mesh target

```python
from src.io import obj_to_tensor    # requires pip install -e ".[io]"

target = obj_to_tensor("mesh.obj", grid_size=32)
```

### Use the built-in training runners

`MorphRunner` and `RegenRunner` manage the training loop, sample pool, LR schedule, and loss internally.

```python
from src.core.runners import MorphRunner

runner = MorphRunner()
runner.init(
    config={
        "cell":       {"hidden_channels": 16, "visible_channels": 4},
        "perception": {"kernel_radius": 1, "channel_groups": 3},
        "update":     {"hidden_dim": 128, "stochastic_update": False, "fire_rate": 0.5},
        "grid":       {"size": [32, 32, 32]},
        "training":   {"num_epochs": 2000, "learning_rate": 1e-3, "batch_size": 4},
    },
    target=target_numpy_array,   # (D, H, W, C) float32 numpy array
)
for metrics in runner.train():
    print(runner.current_epoch, metrics["loss_total"])
```

---

## Project Structure

```
3d-nca/
+-- src/                        # Installable Python package
|   +-- __init__.py             # Public API: NCAModel, NCAConfig, low-level components
|   +-- core/                   # NCA engine
|   |   +-- cell.py             # Cell state and channel layout
|   |   +-- grid.py             # Grid3D -- main nn.Module
|   |   +-- perception.py       # 3D Sobel convolution kernels
|   |   +-- update.py           # Learnable MLP update rule
|   |   +-- nca_model.py        # NCAModel high-level wrapper
|   |   +-- schedule.py         # Training event scheduling
|   |   +-- runners/            # MorphRunner, RegenRunner
|   +-- viz/                    # Visualization (matplotlib + pyvista)
|   +-- io/                     # Mesh voxelization (trimesh)
|   +-- server/                 # TCP server for Blender integration
+-- tests/                      # Pytest test suite
+-- notebooks/                  # Jupyter training examples
+-- scripts/                    # Loss plotting utilities
+-- blender/                    # Blender addon
+-- documentation/              # Thesis and Sphinx API docs
+-- pyproject.toml              # Package definition and pip dependencies
+-- conda_env.yml               # Full development conda environment
+-- environment.yml             # Minimal conda + pip environment
```

---

## Visualization

Requires `pip install -e ".[viz]"`.

| Function | Backend | Description |
|---|---|---|
| `show_slice_alpha_mpl` | matplotlib | 2D cross-section, alpha channel |
| `show_slice_color_mpl` | matplotlib | 2D cross-section, RGB |
| `show_slice_alpha_comparison_mpl` | matplotlib | Side-by-side state vs target |
| `show_volume_alpha_mpl` | matplotlib | 3D voxel scatter, alpha |
| `show_volume_rgba_mpl` | matplotlib | 3D voxel scatter, RGBA |
| `show_state_target_comparison_mpl` | matplotlib | 3D side-by-side comparison |
| `show_volume_alpha_pv` | pyvista | Interactive 3D rendering, alpha |
| `show_volume_color_pv` | pyvista | Interactive 3D rendering, color |

```python
from src.viz import show_slice_alpha_mpl, show_volume_alpha_pv

show_slice_alpha_mpl(state, visible_channels=1)
show_volume_alpha_pv(state, visible_channels=1)
```

---

## Blender Integration

A TCP server streams per-epoch state updates to a Blender addon for real-time visualization.

```python
from src.server import NCAServer

server = NCAServer()
server.start(host="localhost", port=8765)
```

Install the addon from `blender/`, connect to `localhost:8765`, and start training. The growing structure updates live in the 3D viewport.

---

## Development

### Setup

```bash
conda env create -f conda_env.yml
conda activate nca3d
pip install -e .
pre-commit install
```

### Tests

```bash
pytest tests/
pytest --cov=src tests/
```

### Linting and Formatting

```bash
black src/ tests/
isort src/ tests/
ruff check src/ tests/
mypy src/
```

Tool configuration is in `pyproject.toml` under `[tool.black]`, `[tool.ruff]`, and `[tool.mypy]`.

### Notebooks

```bash
jupyter lab
```

| Notebook | Description |
|---|---|
| `01_basic_functionality.ipynb` | Component overview and sanity checks |
| `02_sphere_training_alpha.ipynb` | Alpha-only training on a sphere |
| `03_donut_training_alpha.ipynb` | Alpha-only training on a torus |
| `04_donut_training_rgba.ipynb` | RGBA training on a torus |

### Loss Plots

```bash
python scripts/plot_loss.py runs/experiment/loss.csv
python scripts/plot_loss_phased.py runs/experiment/loss.csv
```

---

## API Reference

Full docs: `cd documentation/docs && make html`

**`from src import ...`**

| Symbol | Description |
|---|---|
| `NCAModel` | High-level `nn.Module` -- recommended entry point |
| `NCAConfig` | Dataclass bundling all configuration |
| `Grid3D` / `GridConfig` | Low-level 3D grid module |
| `CellState` / `CellConfig` | Cell state and channel configuration |
| `Perception3D` / `PerceptionConfig` | 3D Sobel perception module |
| `UpdateRule` / `UpdateConfig` | Learnable MLP update rule |

**`from src.core.runners import ...`** -- `MorphRunner`, `RegenRunner`, `NCARunner`, `TrainingSnapshot`

**`from src.viz import ...`** -- see Visualization table above

**`from src.io import ...`** -- `obj_to_tensor`

---

## License

MIT -- see [LICENSE](LICENSE).

## Citation

```bibtex
@software{3d-nca,
  author = {Michal Repcik},
  title  = {3D Neural Cellular Automata Framework},
  year   = {2026},
  url    = {https://github.com/rm-a0/3d-nca}
}
```
