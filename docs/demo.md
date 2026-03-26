# Demo Commands

All commands run from `~/Desktop/FUSE/src`.

```bash
cd ~/Desktop/FUSE/src
```

## Real Time SSI (VC Showcase)

```bash
# Live mode (ZED camera required)
python -m demos.real_time_ssi

# Offline mode (pre-computed data only, no camera)
python -m demos.real_time_ssi --offline

# SVO recording playback
python -m demos.real_time_ssi --svo path/to/recording.svo2
```

**Flow:** Live video + point cloud -> type object name in terminal -> hacker inference status -> mesh + info panel appear alongside live feed.

**Pre-computed data required in `data/<label>/`:** `partial.npy`, `mesh_aligned.obj`, `physics.json`, `crop.png`

---

## Pre-compute Object Data

Captures ZED frames, generates mesh via Hunyuan3D, aligns, computes physics.

```bash
# Single object (full pipeline: detect + mesh + align + physics)
python -m demos.precompute_demo_data --object mug

# All objects
python -m demos.precompute_demo_data --all

# Detect only (skip mesh generation — no internet needed)
python -m demos.precompute_demo_data --object cup --skip-mesh

# Physics only (recompute from existing aligned mesh)
python -m demos.precompute_demo_data --object mug --physics-only
```

---

## Run Hunyuan3D on Custom Image

```bash
# Generate mesh from any image (v2.0, default)
python -m demos.run_hunyuan3d path/to/image.png

# Use v2.1 (3.3B model, higher detail)
python -m demos.run_hunyuan3d path/to/image.png --model v21

# Generate + align + save into data/<label>/
python -m demos.run_hunyuan3d ../data/mug/crop.png --label mug --save

# Skip visualization
python -m demos.run_hunyuan3d image.png --no-vis

# Custom output path
python -m demos.run_hunyuan3d image.png --output my_mesh.obj
```

---

## Pipeline Phase Demos

Individual pipeline stages for debugging/testing.

```bash
# Phase 1: ZED camera feed only
python -m demos.phase1_camera

# Phase 2: 2D detection overlay
python -m demos.phase2_detection

# Phase 3: 3D point cloud extraction
python -m demos.phase3_extraction

# Phase 4: Full pipeline (detection + 3D)
python -m demos.phase4_pipeline

# All phases accept optional SVO file
python -m demos.phase4_pipeline path/to/recording.svo2
```

---

## Deploy Cloud Endpoints

```bash
# Hunyuan3D v2.0 (1.1B)
modal deploy cloud/hunyuan3d.py

# Hunyuan3D v2.1 (3.3B)
modal deploy cloud/hunyuan3d_v21.py
```
