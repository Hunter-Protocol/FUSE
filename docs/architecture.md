# Architecture

## Pipeline Overview

```
ZED Mini (720p, SDK v5.2.1)
  ├── RGB ──> YOLOE Seg (open-vocab, pretrained) ──> label + 2D box + pixel mask
  ├── Depth/Point Cloud ──> mask projection ──> per-object 3D point cluster
  └── FusedObject per object (label + box + 3D cluster + centroid)
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
    label: str               # "cup"
    confidence: float        # 0.94
    source: str              # "fused" | "2d_only" | "3d_only"
    box_2d: tuple            # (x_min, y_min, x_max, y_max)
    points_3d: np.ndarray    # N x 3 (partial cluster)
    centroid: tuple           # (x, y, z) meters
    color: tuple              # (R, G, B) for point cloud visualization
```

## Execution Model

- Single model (YOLOE Seg) runs detection + segmentation
- 3D extraction is a direct point cloud lookup (no inference, microseconds)
- Target: 15+ FPS fused output
- Frame strategy: grab latest from ZED, drop stale frames

## Coordinate Frame

Camera frame (ZED default) for v1. Robot base / world frame deferred to v2.
