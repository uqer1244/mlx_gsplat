# MLX 3D Gaussian Splatting

An Apple Silicon-optimized, **UMA (Unified Memory Architecture) native 3D Gaussian Splatting (3DGS)** implementation. 
By combining the Apple MLX framework with custom Metal compute shaders, this project delivers extreme memory efficiency and fast training speeds on MacBooks and Mac Studios—**completely free of PyTorch and pycolmap dependencies**.

---

## Technical Highlights

This project is engineered to maximize Apple Silicon's Unified Memory Architecture while overcoming standard framework bottlenecks through key high-performance optimizations:

### 1. UMA-Optimized Pointer-Stable Allocator
* **Zero-Copy Memory Pooling**: Avoids the massive overhead of dynamic memory reallocation during Gaussian splitting and pruning (Densification & Pruning). Instead, it uses a fixed-size physical memory pool linked to active parameters through a stable indirection mapping table.
* **Host CPU Memory Preloading**: Large image datasets are cached in CPU host RAM as compact `uint8` NumPy arrays rather than preloaded as `float32` arrays in VRAM. Images are converted to `float32 mx.array` on-the-fly during batch loading, preventing VRAM cache fragmentation.
* **Proactive Memory Cache Flushing**: Automatically clears unused intermediate tensors from the GPU cache at the end of each training loop iteration (`mx.clear_cache()`), eliminating memory accumulation and preventing VRAM swap bottlenecks or OOM errors.

### 2. High-Efficiency Differentiable MLX Rasterizer
* **Dynamic Pixel Chunking**: Automatically adjusts the pixel chunk size (`pixel_chunk_size`) per step based on the number of active Gaussians. When $N_{\text{active}}$ is low, it scales up chunk sizes to maximize GPU thread parallelism. When $N_{\text{active}}$ is high, it automatically shrinks chunk sizes to fit safely within physical VRAM, preventing SSD swap bottlenecks.
* **Mathematical Simplification (`mx.cumprod`)**: Replaces the expensive chain of `log` -> `cumsum` -> `exp` for transmittance calculation with a single cumulative product (`mx.cumprod`). This reduces redundant shader dispatches and intermediate tensor footprints across millions of elements.
* **Early Opacity Culling**: Checks logit opacities during the culling step to filter out near-transparent Gaussians (logit < `-5.3` or raw opacity < `0.005`) before they enter the rasterizer, significantly reducing the downstream autograd workload.
* **Optimized Autograd Graph**: Bypasses redundant resizing of predictions and performs target image resizing outside the differentiable loss closure to avoid tracking non-differentiable operations in the backward pass.

### 3. Fused C++ / Metal Adam Optimizer Extension
* **Direct Virtual Memory Mapping**: Wraps the raw physical memory addresses of MLX/NumPy arrays directly into Metal buffers (`MTL::Buffer`) using page-aligned virtual memory wrapping, bypassing python-to-C++ array copying.
* **In-place GPU Updates**: Executes all parameter and Adam momentum updates in-place directly on the GPU using a fused Metal compute shader. This eliminates the CPU-GPU synchronization and the 18+ individual scatter updates previously required in Python every step.

---

## System Requirements

* **OS**: macOS 14.0 (Sonoma) or newer
* **Hardware**: Apple Silicon Mac (M1, M2, M3, M4 and Pro/Max/Ultra variants)
* **Software**:
  * **Xcode Command Line Tools** (Required for C++ and Metal kernel compilation)
  * **Python**: 3.10 or newer (3.11 or 3.12 recommended)

---

## Setup Instructions

### 1. Install Xcode Command Line Tools
Verify you have the required compiler tools installed:
```bash
xcode-select --install
```

### 2. Clone and Navigate to the Repository
Move to the directory containing the source files:
```bash
cd mlx_gsplat_new
```

### 3. Create and Activate a Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Compile the Custom Metal C++ Extension
Compile the fused GPU Adam optimizer extension:
```bash
python setup_mlx.py build_ext --inplace
```
* **Verification**: A successful compilation creates a library file named `mlx_gsplat_ext.cpython-xxx-darwin.so` in the project root directory.

---

## 📂 Dataset Structure

This project natively reads standard COLMAP SfM (Structure-from-Motion) binary exports. Place your dataset under a `data/` subdirectory at the project root.

### Expected Directory Layout
```text
mlx_gsplat_new/
└── data/
    └── garden/                 <-- Scene directory (e.g., garden)
        ├── images/             <-- Original high-resolution images (.jpg, etc.)
        └── sparse/
            └── 0/              <-- COLMAP sparse output path
                ├── cameras.bin
                ├── images.bin
                └── points3D.bin
```

> [!NOTE]
> Ensure the image filenames match the filenames registered in the COLMAP binary databases (including character casing).

---

## Running Instructions

The primary entrypoint for training is `train.py` in the root directory.

### 1. Validate Installation (Synthetic Mode)
Test the graphics pipeline and Metal compilation using synthetic data (3,000 points, no disk reads):
```bash
python train.py --max_steps 50
```

### 2. Train on Real COLMAP Datasets (Real Data Mode)
To run training and export the final PLY point cloud on any dataset placed inside the `data/` directory (e.g. `data/garden` or other custom directories):
```bash
python train.py --data_dir data/garden --max_steps 2500 --data_factor 4 --render_width 120 --render_height 90
```

### Command-Line Arguments
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--data_dir` | `data/garden` | Path to COLMAP scene directory (e.g. `data/garden` or any other dataset). |
| `--max_steps` | `2500` | Total number of training iterations. |
| `--data_factor` | `4` | Image downsampling factor (e.g., 4 divides width/height by 4. Essential for VRAM management). |
| `--render_width` | `200` | Intermediate rasterization width resolution used during training. |
| `--render_height` | `150` | Intermediate rasterization height resolution used during training. |
| `--sh_degree_interval` | `500` | Number of steps between increments of the Spherical Harmonics degree. |
| `--init_gaussians` | `None` | Optional cap for initial SfM points. Capping to a lower count (e.g., `30000`) speeds up training up to 3x. |
| `--lazy_images` | `False` | Enable lazy image loading to load images from disk on-the-fly, reducing host memory footprint. |

### Exported PLY File
Upon completing training, a standard ply file named after the dataset (e.g. **`<dataset_name>_mlx_custom_3dgs.ply`**) is dynamically exported to the project root.
* You can visualize the exported PLY file in standard 3DGS web viewers such as [SuperSplat](https://playcanvas.com/supersplat) or [Luma Web Virtualizer](https://lumalabs.ai/).

---

## Troubleshooting

### Q1. Compilation error during `setup_mlx.py` execution
* **Cause**: Xcode Command Line Tools are either not installed or the SDK path linkage is broken.
* **Fix**: Reinstall the command line tools via `xcode-select --install` and try compiling again.

### Q2. Training crashes due to out-of-memory (OOM) / Exit Code 137
* **Cause**: High-resolution training images are taking too much unified memory space, or intermediate autograd activations are exceeding available RAM.
* **Fix**:
  * Increase the `--data_factor` argument (e.g., from `4` to `8` or `16`) to decrease image resolution.
  * Reduce `--render_width` and `--render_height` resolutions to scale down the rasterization workload.

---

## Roadmap / TODO

- [x] **Generic CLI & Dataset Support**: Expose `--data_dir` to load custom datasets dynamically, and export dynamically-named PLY files matching the scene directory.
- [x] **UMA Memory Footprint Optimization**: Store dataset images as compact `uint8` arrays on host memory, utilize dynamic pixel chunking, and proactively flush the Metal cache.
- [ ] **Custom Metal (MSL) Rasterizer Kernel**: Port the forward and backward projection/alpha-blending logic to custom Metal compute shaders (MSL) to achieve sub-10ms iteration speeds.
- [ ] **Interactive Real-Time Viewer**: Build a lightweight, MacBook-native 3D Gaussian visualizer utilizing MLX and Metal APIs for real-time training monitoring.
- [ ] **Adaptive Chunk Auto-Tuning**: Implement automated chunk sizing algorithms that dynamically adjust based on active parameters and real-time GPU VRAM headroom.
