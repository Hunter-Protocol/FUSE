# Architecture

## Pipeline Overview

```
ZED Mini (720p, SDK v5.2.1)
  ├── RGB ──> YOLOE Seg (open-vocab, pretrained) ──> label + 2D box + pixel mask
  ├── Depth/Point Cloud ──> mask projection ──> per-object 3D point cluster
  ├── Shape Completion ──> PoinTr (ShapeNet55) ──> completed 3D shape (8192 pts)
  ├── Cloud 3D Generation ──> YOLOE crop ──> Modal A100 ──> Hunyuan3D Full ──> mesh + points
  └── FusedObject per object (label + box + 3D cluster + centroid + completed shape)
```

## Pipeline Steps

### Step 1: Detection + Segmentation (RGB Stream)
- **Model:** YOLOE Seg (open-vocabulary segmentation)
- **Input:** Left RGB image (720p)
- **Output:** Per-object label, 2D bounding box, confidence score, **pixel-level mask**
- **Speed:** ~25-30 FPS on RTX 3070

### Step 2: 3D Extraction (Mask → Point Cloud)
- **Method:** Use YOLOE pixel masks to index into ZED point cloud
- **Logic:** For each detected object, extract the 3D points at the masked pixel locations
- **Result:** Clean per-object 3D point cluster (no background table/wall points)
- **No separate 3D model needed** — the 2D mask defines which 3D points belong to each object

### How ZED Produces the Point Cloud

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

## Output Per Object

```python
@dataclass
class FusedObject:
    label: str                              # "cup"
    confidence: float                       # 0.94
    source: str                             # "fused" | "2d_only"
    box_2d: tuple                           # (x_min, y_min, x_max, y_max)
    mask: np.ndarray                        # (H, W) bool — pixel-level segmentation mask
    points_3d: np.ndarray                   # N x 3 (partial cluster)
    centroid: tuple                         # (x, y, z) meters
    color: tuple                            # (R, G, B) for point cloud visualization
    completed_points_3d: np.ndarray | None  # (v2) 8192 x 3 completed shape, or None
    completed_centroid: tuple | None        # (v2) centroid of completed shape
```

## Shape Completion (V2)

### Step 3: Shape Completion (PoinTr)

- **Model:** PoinTr (ICCV 2021), trained on ShapeNet55 (55 object categories)
- **Checkpoint:** `PoinTr_ShapeNet55.pth`
- **Input:** 2,048 points (FPS-downsampled from ~13K partial cloud, normalized to unit sphere)
- **Output:** 8,192 completed points (denormalized back to camera frame)
- **Speed:** ~29ms per object on RTX 3070

### Preprocessing

1. **FPS downsample** ~13K partial points to 2,048 (using `pointnet2_ops`)
2. **Normalize to unit sphere**: subtract centroid, divide by max radius
3. **Inference**: PoinTr forward pass
4. **Denormalize**: reverse normalization back to camera frame (meters)

### Caching

Completion is cached per object label. Cache hit if centroid moved < 3cm since last completion. Static scenes have zero completion overhead after the first frame.

### ShapeNet Category Coverage

| FUSE label | ShapeNet category | Synset ID | Coverage |
|------------|-------------------|-----------|----------|
| mug | mug | 03797390 | Direct match |
| bottle | bottle | 02876657 | Direct match |
| phone | cellphone | 02992529 | Direct match |
| cup | mug | 03797390 | Close match |
| fork | knife | 03624134 | Approximate |

Objects without a ShapeNet mapping skip completion gracefully (`completed_points_3d = None`).

## Execution Model

- Two models: YOLOE Seg (detection) + PoinTr (shape completion, optional)
- 3D extraction is a direct point cloud lookup (no inference, microseconds)
- Shape completion runs conditionally (cached for static objects)
- Target: 14-16 FPS with completion enabled
- Frame strategy: grab latest from ZED, drop stale frames

## Performance (RTX 3070, 720p)

### Per-Frame Timing Breakdown

| Step | Time | Notes |
|------|------|-------|
| ZED grab + BGR | ~9ms | `retrieve_image` + numpy copy |
| YOLOE detect | ~14ms | Open-vocab seg inference (after warmup) |
| Point cloud retrieve | ~3ms | `retrieve_measure(XYZRGBA)` |
| Mask → 3D fusion | ~2ms | Array indexing into organized point cloud |
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
| PoinTr | ~2-3 GB |
| **Total** | **~5-8 GB** |

### Completion FPS Impact

| Scenario | Completion overhead | Total FPS |
|----------|-------------------|-----------|
| All cached (static scene) | ~0ms | ~16 (unchanged) |
| 1-2 objects move | ~29-58ms | ~12-14 |
| 5 new objects (worst case, one-time) | ~145ms | ~7 |

## Cloud 3D Generation (Hunyuan3D Full)

### Architecture

The cloud inference path runs Hunyuan3D Full on Modal serverless (A100 80GB) for high-quality image-to-3D mesh generation:

```
Local (RTX 3070)                          Cloud (Modal A100 80GB)
─────────────────                         ──────────────────────────
ZED RGB frame
  → YOLOE Seg detect
  → Crop object (RGBA, 512x512)
  → Upload crop via Modal API ──────────→ Hunyuan3D Full inference
                                            → DINOv2 encode image
                                            → DiT denoise (50 steps, ~9s)
                                            → VAE decode volume (~17s)
                                            → Marching cubes → mesh
  ← Download .glb mesh ←─────────────── Return mesh (~600K verts)
  → Sample points from mesh
  → Align to partial cloud (ICP)
  → Visualize
```

### Timing Breakdown (A100 80GB)

| Stage | Time | Notes |
|-------|------|-------|
| Diffusion sampling (50 steps) | ~9s | ~5.6 it/s on A100 |
| Volume decoding (7134 chunks) | ~17s | ~420 it/s on A100 |
| **Total GPU generation** | **~28s** | 5.5x faster than RTX 3070 |
| Cold start (container spin-up) | ~100-170s | One-time after idle |
| Warm end-to-end | ~30-35s | Including network overhead |

### Files

- `src/cloud_hunyuan3d.py` — Modal endpoint (container def + inference function)
- `src/test_hunyuan3d_cloud.py` — Integration test (ZED crop → cloud → align → visualize)

## Coordinate Frame

Camera frame (ZED default) for v1. Robot base / world frame deferred to v2.
