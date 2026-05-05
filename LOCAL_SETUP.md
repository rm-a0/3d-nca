# Local Setup Guide

> **Recommendation:** If you have access to GitHub, the [repository README](https://github.com/rm-a0/3d-nca) provides a better experience with rendered visuals, direct release downloads, and links to the full online documentation. This guide exists as a fallback for setting up the project **entirely from the submitted source files**, without any internet access or GitHub.

---

## Prerequisites

- **Python** 3.11 (3.12+ not supported)
- **Conda** (recommended) or pip
- **CUDA-capable GPU** (strongly recommended for training; CPU works but is slow)
- **Blender 3.3+** (only required for the live visualization add-on)

---

## 1. Install the Python Package

Choose one of the two paths below.

### Option A - Conda (recommended, includes CUDA)

```bash
conda env create -f conda_env.yml
conda activate nca3d
pip install -e .
```

> Use `environment.yml` instead of `conda_env.yml` for a minimal environment without dev tools.

### Option B - pip only

```bash
pip install -e .                  # core only (torch, numpy, tqdm)
pip install -e ".[viz]"           # add matplotlib + pyvista
pip install -e ".[io]"            # add trimesh (mesh voxelization)
pip install -e ".[all]"           # install all of the above
```

> Run these commands from the root of the project directory (where `pyproject.toml` is located).

---

## 2. Verify the Installation

```bash
python -c "import nca3d; print('OK')"
```

---

## 3. Run the Notebooks

Jupyter is included in the Conda environment. Launch it from the project root:

```bash
jupyter lab
```

Open any notebook from the `notebooks/` directory. A good starting point is `01_basic_functionality.ipynb`.

The `kaggle_run_*.ipynb` and `colab_run_*.py` notebooks are self-contained - you can upload them directly to Kaggle or Google Colab and run them as-is. Repo cloning and package installation are already handled inside those notebooks.

---

## 4. Blender Add-on Installation

The add-on `.zip` is included in the submission - **do not unzip it**.

1. Open Blender and go to `Edit > Preferences > Add-ons`.
2. Click **Install...** and select the `.zip` file.
3. Enable the add-on by checking the box next to **3D-NCA**.
4. Open the 3D Viewport sidebar (press **N**) and select the **NCA** tab.

---

## 5. Live Training with Blender (Optional)

Start the NCA TCP server from the project root:

```bash
python -m nca3d.server.app
```

Then in Blender: enter the IP/Port in the **Connection** panel and click **Connect**. The voxel structure will update live as the model trains.

For detailed add-on usage (target export, training controls, configuration panels) see `documentation/add-on/README.md`.

---

## 6. API Documentation (Offline)

Pre-built HTML docs are included at `documentation/docs/_build/html/index.html` - open this file directly in a browser.

To rebuild the docs from source:

```bash
cd documentation/docs
python -m sphinx.cmd.build -b html . _build/html
```
