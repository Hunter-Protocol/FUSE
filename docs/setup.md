# Setup

## Hardware

- **GPU:** NVIDIA RTX 3070 (8GB VRAM)
- **Laptop:** Razer Blade 15
- **Camera:** ZED Mini

## Prerequisites

- ZED SDK v5.2.1 (installed)
- Python 3.x
- CUDA (required by ZED SDK and PyTorch)

## Python Dependencies

```bash
# TODO: create requirements.txt as modules are added
pip install torch torchvision          # PyTorch (inference)
pip install ultralytics                # YOLO World
pip install pyzed                      # ZED SDK Python bindings
pip install opencv-python              # 2D visualization
pip install open3d                     # 3D point cloud visualization
pip install numpy                      # Array operations
```

## ZED Camera Setup

1. Connect ZED Mini via USB 3.0
2. Verify camera is detected: `python -c "import pyzed.sl as sl; cam = sl.Camera(); print(cam.open())"`
3. Record SVO files for offline dev: use ZED Explorer or SDK recording API

## Running Experiments

### Hunyuan3D-2 (Image-to-3D shape generation)

```bash
cd src/

# Full model (Hunyuan3D-2, ~151s per object)
python test_hunyuan3d.py

# Mini model (0.6B params, faster)
python test_hunyuan3d.py --mini

# Mini turbo (fastest variant)
python test_hunyuan3d.py --turbo
```

### TripoSR (Image-to-3D)

```bash
cd src/
python test_triposr.py
```

## Offline Development

Record `.svo` files from the ZED camera for development without the camera plugged in. This allows testing the full pipeline on recorded data.
