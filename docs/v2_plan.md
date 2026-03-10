# FUSE V2 Plan -- Shape Completion (PoinTr ShapeNet55)

## Context

FUSE v1 is complete: ZED Mini -> YOLOE Seg -> mask-based 3D extraction -> FusedObject output, running at ~16 FPS on RTX 3070. The pipeline produces partial 3D point clusters per object (~13K points from single viewpoint). Shape completion fills in the unseen geometry (back faces, occluded regions) -- critical for downstream grasp planning.

**Model choice: PoinTr** (from github.com/yuxumin/PoinTr). Transformer-based point cloud completion, ICCV 2021 Oral. Using `PoinTr_ShapeNet55.pth` checkpoint trained on all 55 ShapeNet categories -- covers mug, bottle, telephone, cellphone, bowl, knife. 4/5 of our targets are covered (fork has no exact match but knife is close). Uncovered objects skip completion gracefully. Can upgrade to AdaPoinTr later when those weights become available.

## Pipeline Flow with Shape Completion

```
Frame arrives from ZED
       |
       +-- 1. cam.get_bgr()              -> RGB image
       +-- 2. detector.detect(bgr)        -> YOLOE masks + labels
       +-- 3. retrieve_measure(XYZRGBA)   -> raw point cloud
       +-- 4. pc_data[mask] for each det  -> partial 3D points per object (~13K pts)
       |
       v
  +---------------------------------------------+
  |  5. OUTLIER REMOVAL                         |
  |                                             |
  |  Two-stage filter per object:               |
  |    a. Depth (Z) MAD filter: remove points   |
  |       > 3 MAD from median depth             |
  |    b. Statistical outlier removal (Open3D)  |
  |       nb_neighbors=20, std_ratio=1.5        |
  +---------------------------------------------+
       |
       v
  +---------------------------------------------+
  |  6. SHAPE COMPLETION                        |  <-- PoinTr inference
  |                                             |
  |  For each detected object:                  |
  |    a. Downsample 13K -> 2,048 pts (FPS)     |
  |    b. Normalize to unit sphere              |
  |    c. Feed into PoinTr model                |
  |    d. Get 8,192 completed points out        |
  |    e. Denormalize back to camera frame      |
  |                                             |
  |  Caching: only runs when object first       |
  |  appears or centroid shifts > 3cm.          |
  |  Static scenes = zero overhead after        |
  |  first completion.                          |
  +---------------------------------------------+
       |
       +-- 7. Build FusedObject (with both partial + completed points)
       +-- 8. Compute centroid from completed points
       +-- 9. Return to demo for visualization
```

## Phase 2a: Standalone PoinTr Setup

### 2a.1 -- Clone PoinTr repo and build CUDA extensions

- Clone `github.com/yuxumin/PoinTr` outside the FUSE directory (`~/Desktop/PoinTr`)
- Build three CUDA extensions: `chamfer_dist`, `pointnet2_ops`, `knn_cuda`
  - `chamfer_dist`: built via `pip install . --no-build-isolation` in `extensions/chamfer_dist/`
  - `pointnet2_ops`: cloned from erikwijmans/Pointnet2_PyTorch, patched `setup.py` to set `TORCH_CUDA_ARCH_LIST="8.6"` (RTX 3070)
  - `knn_cuda`: installed from pre-built wheel
  - All require `CUDA_HOME=/usr/local/cuda-12.8` to match PyTorch's CUDA 12.8
- Dependencies: `timm`, `easydict`, `ninja`
- Download pretrained checkpoint: `pointr_shapenet55.pth`

**Status: DONE**

### 2a.2 -- `src/shape_completer.py`

```python
class ShapeCompleter:
    def __init__(self, checkpoint_path, device="cuda")
    def can_complete(self, label: str) -> bool       # check ShapeNet category mapping
    def preprocess(self, points: np.ndarray) -> tuple # FPS downsample 13K->2048, normalize to unit sphere
    def postprocess(self, completed, centroid, scale) # denormalize back to camera frame (meters)
    def complete(self, points: np.ndarray) -> np.ndarray | None  # full pipeline
```

Key details:
- Uses `importlib` to load PoinTr modules individually (avoids `models/__init__.py` which imports GRNet and other unneeded models with missing CUDA extensions)
- Checkpoint keys prefixed with `module.` from DDP training -- stripped on load
- FPS downsample via `pointnet2_ops.pointnet2_utils.furthest_point_sample` on GPU
- Input: `(N, 3) float32` partial cloud in camera frame -> Output: `(8192, 3) float32` completed cloud in camera frame

**Status: DONE**

### 2a.3 -- Standalone test

- Synthetic half-sphere test: 5000 pts in, 8192 pts out, ~29ms inference
- Real mug from ZED: 11,390 pts in, 8192 pts out, ~43ms inference
- Visualizes original (red) vs completed (cyan) side-by-side in Open3D

**Status: DONE**

## Phase 2b: Pipeline Integration

### 2b.1 -- Extend FusedObject (`src/fused_object.py`)

Added fields (with defaults for backward compatibility):
- `completed_points_3d: np.ndarray | None = None` -- completed shape (8192 x 3)
- `completed_centroid: tuple | None = None` -- centroid of completed shape
- Properties: `num_completed_points`, `has_completion`

**Status: DONE**

### 2b.2 -- Integrate into FUSEPipeline (`src/pipeline.py`)

- `enable_completion=False` parameter on `__init__`
- Lazy-load `ShapeCompleter` only when enabled (via `@property`)
- Completion cache keyed by object label:
  - Cache hit: if object centroid moved < 3cm since last completion, reuse cached result
  - Cache miss: run `completer.complete()`, store result
- Completion step runs after building FusedObjects, before scene cloud extraction
- Objects with `can_complete() == False` skip completion (`completed_points_3d = None`)

**Status: DONE**

### 2b.3 -- Outlier removal (`remove_outliers` in `src/pipeline.py`)

Two-stage filter applied to each object's raw point cloud before centroid computation:

1. **Depth (Z) MAD filter**: removes points whose depth deviates > 3 MAD from median depth. Catches depth-bleeding at mask edges where stereo matching assigns background depth to object boundary pixels.

   **MAD (Median Absolute Deviation)** is a robust measure of spread, like standard deviation but resistant to outliers:
   - Compute `median_Z` of all points
   - For each point, compute `|Z - median_Z|`
   - `MAD = median(|Z - median_Z|)`
   - Remove points where `|Z - median_Z| > 3 * MAD`

   Why MAD over standard deviation? Std gets inflated by outliers -- if 20% of points are on the wall 50cm behind the mug, std grows and those outliers survive the filter. MAD uses medians so it ignores outliers entirely, making it much better at catching depth bleed.

2. **Statistical outlier removal** (Open3D): `nb_neighbors=20, std_ratio=1.5`. For each point, computes its mean distance to its 20 nearest neighbors. Points whose mean distance exceeds `(global_mean + 1.5 * global_std)` are removed. Catches remaining scattered noise that passed the depth filter.

**Status: DONE**

### FPS Impact

| Scenario | Completion overhead | Total FPS |
|----------|-------------------|-----------|
| All cached (static scene) | ~0ms | ~16 (unchanged) |
| 1-2 objects move | ~29-58ms | ~12-14 |
| 5 new objects (worst case, one-time) | ~145ms | ~7 |

### VRAM Budget (8GB RTX 3070)

| Component | Est. VRAM |
|-----------|-----------|
| YOLOE-11s-seg | ~2-3 GB |
| ZED NEURAL depth | ~1-2 GB |
| PoinTr | ~2-3 GB |
| **Total** | **~5-8 GB** |

If tight: run PoinTr in float16, or switch ZED depth to ULTRA mode.

## Phase 2c: Visualization + Testing

### 2c.1 -- `src/phase5_demo.py`

Based on `phase4_demo.py`, adds:
- **Third Open3D window**: "FUSE - Completed Shapes"
- **Visual distinction**: original points in solid label color, completed points in lighter shade (50% blend toward white)
- **Toggle**: press `c` in OpenCV window to enable/disable completion at runtime
- **HUD**: per-object completion status: `[completed: 8192 pts]` or `[no completion]`

**Status: DONE**

### 2c.2 -- `src/capture_and_complete.py`

Live demo with parallel threads:
- Main thread: ZED capture + YOLOE detection + 2D/3D visualization
- Worker thread: PoinTr completion runs asynchronously, results appear when ready
- Two Open3D windows: "Partial (ZED)" and "Completed (PoinTr)"
- Press `s` to save `.npy` files, `q` to quit

**Status: DONE**

### 2c.3 -- Category coverage

| FUSE label | ShapeNet synset | Category | Coverage |
|------------|----------------|----------|----------|
| mug | 03797390 | mug | Direct match |
| bottle | 02876657 | bottle | Direct match |
| phone | 02992529 | cellphone | Direct match |
| cup | 03797390 | mug | Close match |
| fork | 03624134 | knife | Approximate |

### 2c.4 -- Docs updated

- `docs/architecture.md` -- v2 pipeline diagram, shape completion section, VRAM budget, FPS impact
- `docs/roadmap.md` -- v2 phases (5-8)
- `CLAUDE.md` -- architecture updated, shape completion marked done in V2 backlog

**Status: DONE**

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `src/shape_completer.py` | Created | PoinTr wrapper with preprocessing/postprocessing |
| `src/capture_and_complete.py` | Created | Live parallel capture + completion demo |
| `src/phase5_demo.py` | Created | Full demo with completion visualization |
| `src/fused_object.py` | Modified | Added completed_points_3d, completed_centroid fields |
| `src/pipeline.py` | Modified | Integrated ShapeCompleter with caching + outlier removal |
| `docs/architecture.md` | Modified | Added v2 pipeline + shape completion section |
| `docs/roadmap.md` | Modified | Added v2 phases |
| `CLAUDE.md` | Modified | Updated architecture and V2 backlog |
| `.gitignore` | Modified | Added data/, models/, *.pth, *.npy |

## Future Work (V2 continued)

- Evaluate AdaPoinTr when weights become available (checkpoint already downloaded: `AdaPoinTr_ps55.pth`)
- Fine-tune PoinTr on target environment objects
- TensorRT optimization for both YOLOE and PoinTr
- ROS2 integration
- Robot base / world coordinate frame transform
