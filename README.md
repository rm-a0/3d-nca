# 3D Neural Cellular Automata

## Overview

Modular PyTorch framework for **3D Neural Cellular Automata**.

## How to Use

1. Clone & setup

    ```bash
    git clone https://github.com/rm-a0/3d-nca
    cd 3d-nca
    ```

2. Create environment (choose one)

    - a. Conda (GPU-aware, one-click)

        ```bash
        conda env create -f conda_env.yml
        conda activate nca3d
        ```

    - b. Pure Poetry (developers / CI / any OS)

        ```bash
        python -m venv .venv
        source .venv/bin/activate
        pip install poetry
        poetry install --with dev
        ```

3. Run (VS Code or Jupyter)

    ```bash
    code notebooks/01_test_notebook.ipynb
    ```

## Project Structure

```txt
src/
├── core/          # Core NCA logic: cell state, perception kernels, update rules, grid management
├── training/      # Training loops, loss functions, callbacks, logging (wandb)
├── io/            # Load/save states, export to Blender/Mesh formats
├── viz/           # Real-time 3D rendering, Jupyter helpers, interactive plots
└── experiments/   # High-level scripts: grow from seed, target matching, ablation studies
notebooks/         # Interactive demos, debugging, visualization notebooks
tests/             # Unit & integration tests (pytest)
scripts/           # Standalone tools: export, benchmark, data generation
documentation/     # LaTeX thesis/paper, figures, graphs, write-ups
conda_env.yml      # Exact reproducible Conda environment
pyproject.toml     # Package metadata, dependencies, lint/type config
```
