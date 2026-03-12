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

## Cloud Inference Setup (Modal + Hunyuan3D Full)

### Prerequisites

```bash
pip install modal
modal token set          # authenticate with Modal account
```

### Deploy the Hunyuan3D Full endpoint

```bash
modal deploy src/cloud_hunyuan3d.py
```

This deploys a serverless A100 80GB endpoint. It scales to zero when idle (~$0.02/inference when used).

### Run cloud integration test

```bash
python src/test_hunyuan3d_cloud.py
```

This sends a mug crop to the cloud endpoint, receives the generated mesh, samples points, aligns to the partial cloud, and visualizes the result. First run will be slow (~100-170s cold start), subsequent runs ~30-35s.

## Offline Development

Record `.svo` files from the ZED camera for development without the camera plugged in. This allows testing the full pipeline on recorded data.
