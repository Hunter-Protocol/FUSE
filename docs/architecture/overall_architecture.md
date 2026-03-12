# Overall Architecture

End-to-end pipeline: ZED stereo camera → detection + segmentation → 3D extraction → denoising → fused output per object.

## Full Pipeline

```
ZED Mini (720p, SDK v5.2.1, NEURAL depth)
  │
  ├── Left RGB Image ──────────────────────────────────────────────────────────────┐
  │                                                                                │
  │   ┌──────────────────────────────────────────────┐                             │
  │   │  YOLOE Seg (open-vocab segmentation)         │                             │
  │   │  Input: 720p BGR frame                       │                             │
  │   │  Output: per-object label + box + pixel mask │                             │
  │   │  Speed: ~14ms inference (~25-30 FPS)         │                             │
  │   └──────────────┬───────────────────────────────┘                             │
  │                  │                                                             │
  │         label, box_2d, confidence, mask (H,W bool)                             │
  │                  │                                                             │
  ├── Depth / Point Cloud ─────────────────────────────────────────────────┐       │
  │   retrieve_measure(XYZRGBA) → organized (H,W,4) cloud, ~3ms           │       │
  │                                                                        │       │
  │   ┌────────────────────────────────────────────────────────────────┐   │       │
  │   │  Mask-Based 3D Extraction                                      │   │       │
  │   │                                                                │   │       │
  │   │  For each detected object:                                     │   │       │
  │   │    points_3d = point_cloud[mask]   (array indexing, ~2ms)      │   │       │
  │   │    Filter NaN/inf values                                       │   │       │
  │   │                                                                │   │       │
  │   │  Works because ZED point cloud is organized: pixel (row,col)   │   │       │
  │   │  in left RGB maps directly to point_cloud[row,col,:3].         │   │       │
  │   │  YOLOE mask on left RGB → 3D points via array indexing.        │   │       │
  │   └──────────────┬─────────────────────────────────────────────────┘   │       │
  │                  │                                                     │       │
  │          raw per-object 3D points (~13K pts, noisy at edges)           │       │
  │                  │                                                     │       │
  │   ┌──────────────▼─────────────────────────────────────────────────┐   │       │
  │   │  Two-Stage Outlier Removal (Denoising)                         │   │       │
  │   │                                                                │   │       │
  │   │  Stage 1: Depth (Z) MAD Filter                                │   │       │
  │   │    → Removes depth-bleeding artifacts at mask edges            │   │       │
  │   │    → See "Denoising" section below for details                 │   │       │
  │   │                                                                │   │       │
  │   │  Stage 2: Statistical Outlier Removal (Open3D)                 │   │       │
  │   │    → Removes remaining scattered noise                         │   │       │
  │   └──────────────┬─────────────────────────────────────────────────┘   │       │
  │                  │                                                     │       │
  │          clean per-object 3D cluster                                   │       │
  │                  │                                                     │       │
  │   ┌──────────────▼─────────────────────────────────────────────────┐   │       │
  │   │  Cloud 3D Generation — Hunyuan3D Full (Modal A100)             │   │       │
  │   │                                                                │   │       │
  │   │  YOLOE mask → crop RGBA (512x512)                              │◄──┘───────┘
  │   │  Upload to Modal A100 endpoint                                 │
  │   │  Hunyuan3D Full: DINOv2 → DiT (50 steps) → VAE → mesh         │
  │   │  Download mesh → sample points → align to partial cloud        │
  │   │  ~28s/object on A100 (see docs/architecture/modal_architecture)│
  │   └──────────────┬─────────────────────────────────────────────────┘
  │                  │
  │                  ▼
  │   ┌────────────────────────────────────────────────────────────────┐
  │   │  FusedObject Output                                            │
  │   │                                                                │
  │   │  label: str               — "mug"                              │
  │   │  confidence: float        — 0.94                               │
  │   │  source: str              — "fused" | "2d_only"                │
  │   │  box_2d: tuple            — (x_min, y_min, x_max, y_max)      │
  │   │  mask: np.ndarray         — (H,W) bool pixel mask              │
  │   │  points_3d: np.ndarray    — N x 3 (denoised partial cluster)   │
  │   │  centroid: tuple          — (x, y, z) meters                   │
  │   │  color: tuple             — (R, G, B) per-label visualization  │
  │   └────────────────────────────────────────────────────────────────┘
```

## How ZED Produces the Point Cloud

The ZED Mini is a stereo camera (left + right lenses). `zed.retrieve_measure(XYZRGBA)` returns a pre-computed point cloud via:

1. **Stereo matching** — for each left-image pixel, find the corresponding pixel in the right image
2. **Disparity** — the horizontal pixel offset (closer objects = larger disparity)
3. **Triangulation** — calculate 3D position using disparity, baseline, and focal length:
   ```
   Z = (focal_length × baseline) / disparity
   X = (pixel_x - cx) × Z / fx
   Y = (pixel_y - cy) × Z / fy
   ```

The `NEURAL` depth mode runs a neural network to improve stereo matching (filling holes, sharpening edges) before triangulating.

### Why Mask-Based 3D Extraction Works

The ZED point cloud is **organized** — each pixel `(row, col)` in the left RGB image has a corresponding 3D point at `point_cloud[row, col, :3]`. Since YOLOE runs on the same left RGB image, a segmentation mask `mask[row, col] == True` maps directly to a 3D point. No projection math needed — just array indexing: `xyz = point_cloud[mask]`.

### Two Point Cloud Views

- **Full scene:** `cam.get_point_cloud()` — flattens the full (H, W, 4) cloud, removes NaN/inf, returns all valid points with camera RGB colors
- **Detected objects only:** `extract_3d_points(mask, pc_data)` — indexes into the raw (H, W, 4) cloud using YOLOE masks, returns only object points colored by class label

## Denoising: Two-Stage Outlier Removal

Implementation: `remove_outliers()` in `src/core/pipeline.py`

### The Problem: Depth Bleeding at Mask Edges

When YOLOE generates a segmentation mask, boundary pixels are ambiguous — they partially overlap the object and partially overlap the background. The ZED's stereo matching for these boundary pixels often latches onto the background depth instead of the object surface. The result is a cluster of stray points projected 20-50cm behind the actual object, at the depth of the wall or table behind it.

This is called **depth bleeding**: the mask says "this pixel belongs to the mug," but the depth map says "this pixel is 1.2m away" (the wall), not "0.5m away" (the mug). The 3D point ends up on the wall, corrupting the object's point cluster and shifting the centroid backward.

### Stage 1: Depth (Z) MAD Filter

**Purpose:** Remove the primary depth-bleeding artifact — background-depth points at mask boundaries.

**Method:** Median Absolute Deviation (MAD) on the Z (depth) axis:

```
median_Z = median of all points' Z values
For each point: deviation = |Z - median_Z|
MAD = median(all deviations)
Remove points where |Z - median_Z| > 3 × MAD
```

**Why MAD instead of standard deviation?**

Standard deviation gets inflated by the very outliers it's trying to remove. If 20% of a mug's points are depth-bled onto the wall 50cm behind, std grows large enough that those outlier points survive the filter. MAD uses medians, which are inherently resistant to outliers — even if half the points are wrong, the median stays anchored to the true object surface. This makes MAD much more aggressive at removing the long-tailed depth bleed while preserving the core cluster.

**What it catches:** Points where stereo matching assigned background depth to object boundary pixels. These form a secondary cluster at the wall/table depth behind the object.

### Stage 2: Statistical Outlier Removal (Open3D)

**Purpose:** Remove remaining scattered noise that passed the depth filter.

**Method:** Open3D's `remove_statistical_outlier` with `nb_neighbors=20, std_ratio=1.5`:

```
For each point:
  Compute mean distance to its 20 nearest neighbors
Global mean = average of all per-point mean distances
Global std = standard deviation of all per-point mean distances
Remove points where mean_distance > global_mean + 1.5 × global_std
```

**What it catches:** Isolated noise points that aren't part of a depth-bleed cluster — random stereo matching errors, edge artifacts from the neural depth mode, or points from thin structures (like a mug handle edge) that ended up spatially isolated.

### Why Two Stages?

Stage 1 (MAD) targets the **dominant failure mode** — depth bleeding produces a bimodal depth distribution (object surface + background surface), and MAD cleanly separates them by operating on the depth axis alone.

Stage 2 (statistical) targets **spatial noise** — points that are at the correct depth but spatially scattered away from the main cluster. These don't show up as depth outliers because they're at similar Z values, but they're isolated in 3D space.

Neither stage alone is sufficient. MAD misses spatially scattered points at the correct depth. Statistical removal misses tightly clustered depth-bleed points (they have many neighbors in the bleed cluster, so they look "normal" to the neighbor-distance metric). Together, they handle both failure modes.

### Performance

Outlier removal is fast enough to run every frame without impacting FPS:
- MAD filter: pure numpy vectorized operations, sub-millisecond
- Statistical removal: Open3D KD-tree, ~1-2ms for ~13K points

## Why PoinTr Was Removed

PoinTr (ShapeNet55) was initially integrated for local shape completion but **removed due to the synthetic-to-real domain gap** making it ineffective on live ZED sensor data:

1. **Orientation mismatch:** PoinTr was trained on ShapeNet objects in canonical pose (mug upright, handle in consistent direction). ZED captures are in camera frame — the mug is rotated/tilted arbitrarily. PoinTr has no category input, so it guesses the shape purely from geometry. A mug in camera frame looks nothing like ShapeNet's canonical partials.

2. **Partial view pattern:** ShapeNet training generates partials by removing clean viewpoint slices from complete synthetic meshes. Real ZED partials only capture the front-facing surface with uneven density (denser near camera center, sparser at edges). The shape of the missing region differs fundamentally from training data.

3. **Point distribution:** ShapeNet points are uniformly sampled from clean meshes. ZED points are noisy, have variable density, and include stereo artifacts even after outlier removal.

4. **No category conditioning:** PoinTr is category-agnostic — it doesn't know it's looking at a mug. It hallucinates geometry based on whatever the partial cloud vaguely resembles in its training distribution.

5. **Scale/proportion:** Real objects have different proportions than ShapeNet's idealized CAD models.

PoinTr was designed for and evaluated on synthetic benchmarks (ShapeNet, PCN), not real-world noisy partial scans. Shape completion for the FUSE pipeline is now handled by Hunyuan3D Full on cloud A100, which uses image conditioning to understand object semantics (see `modal_architecture.md`).

## Execution Model

- **Models loaded:** YOLOE Seg (detection/segmentation)
- **3D extraction:** Direct point cloud lookup via mask indexing — no inference, microseconds
- **Denoising:** Two-stage outlier removal — sub-millisecond MAD + ~1-2ms statistical
- **Cloud 3D generation:** Async, ~28s on Modal A100 (separate from real-time loop)
- **Target:** ~16 FPS
- **Frame strategy:** Grab latest from ZED, drop stale frames

## Performance (RTX 3070, 720p)

### Per-Frame Timing Breakdown

| Step | Time | Notes |
|------|------|-------|
| ZED grab + BGR | ~9ms | `retrieve_image` + numpy copy |
| YOLOE detect | ~14ms | Open-vocab seg inference (after warmup) |
| Point cloud retrieve | ~3ms | `retrieve_measure(XYZRGBA)` |
| Mask → 3D extraction | ~2ms | Array indexing into organized point cloud |
| Outlier removal | ~1-2ms | MAD filter + statistical removal per object |
| Scene point cloud | ~30ms | Flatten, filter NaN, unpack RGBA (skipped 2/3 frames) |
| **Pipeline total** | **~32-64ms** | ~58ms avg with scene skip |

### Rendering Overhead

| Step | Time | Notes |
|------|------|-------|
| Build Open3D data | 1-20ms | `Vector3dVector` conversion, ~1ms when scene skipped |
| Open3D render | ~4-15ms | Two windows (objects + scene) |
| OpenCV render | ~7ms | 2D overlay with masks, boxes, labels |

### FPS

| Metric | Value |
|--------|-------|
| Pipeline only | ~16 FPS |
| Full loop (pipeline + render) | ~13-16 FPS |
| First frame (model warmup) | ~1.4 FPS (one-time) |

### Optimizations Applied

- **Scene point cloud updated every 3rd frame** — scene changes slowly, saves ~30ms pipeline + ~20ms render on skip frames
- **Reuse point cloud data** — `retrieve_measure` called once, shared between object extraction and scene cloud
- **Downsample scene to 100K points** — reduced from 200K, cuts Open3D conversion time
- **Contiguous float64 arrays** — faster `Vector3dVector` construction

### VRAM Budget (8GB RTX 3070)

| Component | Est. VRAM |
|-----------|-----------|
| YOLOE-11s-seg | ~2-3 GB |
| ZED NEURAL depth | ~1-2 GB |
| **Total** | **~3-5 GB** |

## Coordinate Frame

Camera frame (ZED default) for v1. Robot base / world frame deferred to v2.

## Source Files

| File | Role |
|------|------|
| `src/core/camera.py` | ZED Mini interface — RGB, depth, point cloud |
| `src/core/detector.py` | YOLOE Seg — open-vocab detection + segmentation |
| `src/core/pipeline.py` | FUSEPipeline — orchestrates all steps, includes `remove_outliers()` |
| `src/core/fused_object.py` | FusedObject dataclass + label color mapping |
| `src/cloud/hunyuan3d.py` | Modal endpoint for Hunyuan3D Full (see modal_architecture.md) |
| `src/demos/phase{1-4}_*.py` | Progressive demo scripts for each pipeline phase |
| `src/tests/test_hunyuan3d_cloud.py` | Cloud integration test |
