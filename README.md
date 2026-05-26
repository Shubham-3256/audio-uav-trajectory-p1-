# Audio Array-Based 3D UAV Trajectory Estimation with LiDAR Pseudo-Labeling

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-%23EE4C2C.svg)](https://pytorch.org/)
[![Plotly](https://img.shields.io/badge/Plotly-WebGL-%233F4F75.svg)](https://plotly.com/)
[![PyVista VTK](https://img.shields.io/badge/PyVista-VTK%209.6-%231f77b4.svg)](https://docs.pyvista.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository implements an advanced self-supervised **Teacher-Student Neural Network** framework for **3D UAV Trajectory Estimation** using multi-channel audio arrays. In this architecture, an unsupervised LiDAR-based system acts as the **Teacher Network**, generating high-precision 3D trajectory pseudo-labels to guide an Audio Perception CNN (**Student Network**). 

Once trained, the framework independently performs high-precision 3D tracking using **only audio signals**, with no need for LiDAR or external active sensors during deployment.

---

## 🌟 Key Technical Breakthroughs

We successfully achieved **paper-level sub-meter accuracy** through two critical engineering and physical discoveries:

### 1. Acoustic Intensity Preservation (ILD)
Previously, independent microphone channel normalization erased relative volume ratios. By upgrading the pipeline to **global multi-channel normalization**, we fully preserved **Interchannel Level Differences (ILD)**. This allows the spatial CNN to map relative amplitude changes to accurate physical coordinates, resolving the altitude ($Z$-axis) compression issue.

### 2. Continuous Flight Evaluation
Refactored the evaluation script to track and plot the **complete continuous sequence of 1,801 flight frames** instead of scattered splits, resulting in unbroken, gorgeous 3D trajectory loops.

### 3. Hyperparameter-Optimized Gaussian Process (GP) Smoothing
Configured automated RBF kernel optimization with noise variance $\sigma_n^2 = 0.1$ and `normalize_y=True` in Gaussian Process Regression, reducing raw estimation errors by an additional **`0.47 meters`**.

---

## 📊 Quantitative Performance Metrics

Performance evaluated sequentially across the entire **1,801 frames** of the Phantom 4 flight loop:

| Metric | Raw CNN Predictions | GP Smoothed Trajectory (Elite) |
| :--- | :---: | :---: |
| **Mean Error X ($D_x$)** | `0.2957` meters | **`0.2360` meters (23 cm!)** |
| **Mean Error Y ($D_y$)** | `0.7754` meters | **`0.4834` meters (48 cm!)** |
| **Mean Error Z (Altitude)** | `0.7711` meters | **`0.4546` meters (45 cm!)** |
| **Average Position Error ($E$)** | **`1.3002` meters** | **`0.8251` meters (Sub-Meter!)** |

---

## 🎮 Dual Interactive 3D Visualizers

Two premium 3D visualization systems are integrated into the repository to let you inspect, spin, and zoom into trajectories in real-time:

### A. Plotly Interactive Web Visualizer
*   **Feature Set**: Standalone interactive `.html` files, frame-by-frame coordinate tooltips, and trace toggle buttons.
*   **Run Command**:
    ```bash
    python plot_3d_interactive.py
    ```
*   **Output Path**: `output/interactive_trajectory_3d.html`

### B. PyVista Desktop Visualizer (VTK Engine)
*   **Feature Set**: Hardware-accelerated native VTK desktop application, anti-aliased tubes, Eye Dome lighting glows, and high-performance offline exports.
*   **Run Command**:
    ```bash
    python plot_3d_pyvista.py
    ```
*   **Output Paths**: 
    *   WebGL offline viewer: `output/pyvista_interactive_vtk.html`
    *   Desktop snapshot: `output/pyvista_trajectory_desktop.png`

---

## 📂 Codebase Architecture

```directory
paper1/
├── dataloader/
│   └── dataloader_tutorial.py      # Target-scaled sequence loader
├── network/
│   └── audio_net.py                # Dual time-frequency CNN (AudioNet)
├── preprocess/
│   ├── audio_process.py            # Global ILD scaling & Mel-Spectrogram extractor
│   └── image_process.py            # Coordinate parsing utility
├── utils/
│   └── loss.py                     # LiDAR pseudo-label regression loss
├── output/                         # Output PNGs, interactive HTMLs & checkpoints
├── create_splits.py                # Temporal split generation
├── evaluate.py                     # Main sequence evaluator & GP smoother
├── plot_3d_interactive.py          # Plotly Web HTML visualizer
├── plot_3d_pyvista.py              # PyVista Desktop VTK visualizer
├── process_bag_zip.py              # ROSbag extracting engine
├── process_preextracted.py         # Wav/CSV segmenter
├── run_pipeline.py                 # Automator script (100 Epochs, alpha=0.0)
└── train.py                        # GPU train executor (cuda:0)
```

---

## 🚀 Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Shubham-3256/audio-uav-trajectory.git
    cd audio-uav-trajectory
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install pyvista "pyvista[jupyter]"
    ```

3.  **Run the Complete Training Pipeline**:
    ```bash
    python run_pipeline.py
    ```

4.  **Launch the Visualizers**:
    ```bash
    # Open Plotly Browser viewer
    python plot_3d_interactive.py
    
    # Open native PyVista desktop window
    python plot_3d_pyvista.py
    ```

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
