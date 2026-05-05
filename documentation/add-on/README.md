# 🧩 3D-NCA Blender Add-on Guide

This guide covers the installation and usage of the 3D-NCA Blender add-on. The tool serves two primary purposes: generating voxelized `.npz` training targets and providing a real-time interface to control and monitor training sessions.

---

## 📥 Installation & Setup

1. **Download:** Get `nca3d_blender_addon_<latest>.zip` from the [Releases](https://github.com/rm-a0/3d-nca/releases/tag/v0.1.0) page.
2. **Install:** In Blender, go to `Edit > Preferences > Add-ons > Install...`. Select the `.zip` and enable it.
3. **Locate:** Open the 3D Viewport sidebar (Press **N**) and select the **NCA** tab.

> [!CAUTION]
> **Server Requirement:** To use the live training features (Pause, Resume, Real-time updates), you must have a training script running the `NCAServer`. 
> * **Local:** Ensure your script and Blender are on the same machine/network.
> * **Remote:** If training on **Kaggle** or **Colab**, you must use a tunnel. See [notebooks](../../notebooks/colab_run_ngrok.py).

---

## 📖 User Guide

### 1. Target Preparation (No Server Required)
You can create training targets entirely offline. This process converts standard Blender meshes into the compressed `.npz` format expected by the `MorphRunner`.

![Target Export](./assets/target_export.gif)

* **Workflow:** Create your Mesh (Shift+A) -> Select created Mesh in **Target** panel -> Click **Export (.npz)**.

---

### 2. Training Quickstart
Connect Blender to a running training session to visualize the NCA growth from step zero.

![Quickstart](./assets/training_quickstart.gif)

* **Workflow:** Start your Python server -> Enter the IP/Port in **Connection** panel -> Click **Connect** -> Prepare target -> Click **Start Training**.

---

### 3. Training Controls
The add-on provides a remote control interface for your Python training loop, functioning similarly to a media player.

![Controls](./assets/training_controls.gif)

* **Pause/Resume:** Halts the training loop on the server to inspect the current state.
* **Stop:** Safely terminates the training process and saves the current model state.
* **Dynamic Reset:** You can swap targets or reset the grid without restarting your Python script.

---

### 4. Advanced Configuration
The UI panels allow you to tweak the underlying `NCAConfig` parameters. These settings directly map to the core architecture.

![Setup](./assets/training_setup.gif)

* **Panels:** Cell, Perception, Update, Grid and Training settings.
* **Voxel Sync:** Changing grid dimensions in the UI automatically updates the voxelization preview.
* **Deep Dive:** For a technical breakdown of what each slider does, refer to the [NCAConfig API Documentation](https://www.stud.fit.vutbr.cz/~xrepcim00/3d-nca-docs/).

---

### 5. Real-time Schedule Adjustment
You can modify training hyperparameters (like learning rate or batch size) mid-training without interrupting the loop.

![Schedule](./assets/schedule_setup.gif)

* **Dynamic Schedules:** Adjust the "Events" schedule to change how the model behaves during different phases of morphogenesis.
* **Documentation:** See the [Schedule API Guide](https://www.stud.fit.vutbr.cz/~xrepcim00/3d-nca-docs/) for advanced event handling.

---

> [!IMPORTANT]
> **Apply Scale:** Before voxelization, always select your object, go to **Edit Mode** (Tab) and **Select All** (Ctrl+A) to apply Scale. If the scale is not `(1, 1, 1)`, the resulting voxel grid may be distorted.