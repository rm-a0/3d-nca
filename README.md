# 3D Neural Cellular Automata

A modular PyTorch framework for training 3D Neural Cellular Automata on volumetric targets.

![Python](https://img.shields.io/badge/python-3.3%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)
![Blender](https://img.shields.io/badge/blender-3.6%2B-orange)

## Overview

**3D-NCA** provides a composable model API, built-in training runners (morphogenesis and regeneration), visualization helpers (matplotlib/pyvista), and a TCP server for real-time Blender integration.

For theoretical background and architecture details, see the [Thesis](documentation/thesis/) or the [Framework API Docs](https://www.stud.fit.vutbr.cz/~xrepcim00/3d-nca-docs/).

---

## ⚙️ Requirements

- **Python:** 3.11+
- **PyTorch:** 2.0+ (CUDA-capable GPU strongly recommended)
- **Blender:** 3.3+ (Only required for the visualization add-on)

---

## 📦 Installation

Choose the setup path that matches your goal.

### 1. Library Usage (pip)
For users who just want to import `nca3d` into their own projects:
```bash
git clone https://github.com/rm-a0/3d-nca
cd 3d-nca

# Install core (torch, numpy, tqdm)
pip install -e .

# Install with optional feature groups
pip install -e ".[viz]"    # matplotlib + pyvista
pip install -e ".[io]"     # trimesh (mesh voxelization)
pip install -e ".[all]"    # installs all of the above
```

### 2. Development (Conda)
For full development (includes PyTorch+CUDA 12.1, Jupyter, data science tools, and dev tools):
```bash
conda env create -f conda_env.yml
conda activate nca3d
pip install -e .
pre-commit install
```
> [!NOTE]  
> A minimal hybrid environment is also available via `environment.yml`.

---

## 🎨 Blender Integration

A TCP server streams per-epoch state updates to a Blender add-on for real-time 3D visualization.

### Add-on Installation
1. Download `nca3d_blender_addon_<latest>.zip` from the **[Releases](../../releases)** page. *(Do not unzip it).*
2. Open Blender and go to `Edit > Preferences > Add-ons`.
3. Click **Install...**, select the `.zip` file, and check the box to enable the add-on.

### Usage
Start the TCP server in your Python training script:
```python
from nca3d.server import NCAServer

server = NCAServer()
server.start(host="localhost", port=8765)
```
In Blender, open the 3D Viewport side panel (press `N`), locate the **3D-NCA** tab, and click **Connect**. The growing structure will update live as your model trains.

---

## 🚀 Quick Start

`NCAModel` is a standard PyTorch `nn.Module`. You can drop it into any custom training loop or use the high-level runners to manage the sample pool and loss logic.

### Using the MorphRunner
The `MorphRunner` automatically manages the training loop, sample pool, and learning rate schedule.

```python
import torch
from nca3d.core import NCAConfig, MorphRunner

# 1. Setup your configuration
config = NCAConfig(
    grid_size=(32, 32, 32),
    hidden_channels=16,
    visible_channels=4,
    num_epochs=2000,
    batch_size=4,
    learning_rate=1e-3
)

runner = MorphRunner()

# 2. Initialize
runner.init(config.as_dict(), target="your_numpy_array")

# 3. Start training
runner.train()

# 4. Export the model
model = runner.get_model()
```

> [!TIP]
> For target preparation, use the Blender add-on to export models directly to the recommended `.npz` format.

---

## 🗂️ Project Structure
```text
3d-nca/
├── nca3d/                 # Installable Python package
│   ├── core/              # NCA engine (cell, grid, perception, update, runners)
│   ├── viz/               # Visualization tools (matplotlib, pyvista)
│   ├── io/                # Mesh voxelization tools (trimesh)
│   └── server/            # TCP server for Blender integration
├── blender/               # Blender addon source code
├── scripts/               # Loss plotting and other utilities
├── notebooks/             # Jupyter training examples
├── tests/                 # Pytest suite
└── pyproject.toml         # Package definition
```

---

## 📚 Documentation & API Reference

Full API documentation is available [here](https://www.stud.fit.vutbr.cz/~xrepcim00/3d-nca-docs/). 
To build the docs locally:
```bash
cd documentation/docs
python -m sphinx.cmd.build -b html . _build/html
```

**Key Modules:**
* `from nca3d import NCAModel, NCAConfig` - Core architecture.
* `from nca3d.core.runners import MorphRunner, RegenRunner` - Training managers.
* `from nca3d.io import obj_to_tensor` - Mesh processing.
* `from nca3d.viz import show_volume_alpha_pv, show_slice_alpha_mpl` - Rendering.

---

## 📜 Citation & License

This project is released under the **MIT License**.

If you use this work in your research, please cite:
```bibtex
@mastersthesis{3d-nca-thesis,
  author = {Michal Repcik},
  title  = {3D Neural Cellular Automata},
  school = {Brno University of Technology, Faculty of Information Technology},
  year   = {2026},
  url    = {[https://www.vut.cz/en/students/final-thesis/detail/171171](https://www.vut.cz/en/students/final-thesis/detail/171171)},
  note   = {Supervisor: Ing. Karel Fritz}
}
```