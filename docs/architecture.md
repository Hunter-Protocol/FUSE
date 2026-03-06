# Architecture

## Pipeline Overview

```
ZED Mini (720p, SDK v5.2.1)
  ├── RGB ──> YOLO World Seg (open-vocab, pretrained) ──> label + 2D box + pixel mask
  ├── Depth/Point Cloud ──> mask projection ──> per-object 3D point cluster
  └── FusedObject per object (label + box + 3D cluster + centroid)
```

## Pipeline Steps

### Step 1: Detection + Segmentation (RGB Stream)
- **Model:** YOLO World Seg (open-vocabulary segmentation)
- **Input:** Left RGB image (720p)
- **Output:** Per-object label, 2D bounding box, confidence score, **pixel-level mask**
- **Speed:** ~25-30 FPS on RTX 3070

### Step 2: 3D Extraction (Mask → Point Cloud)
- **Method:** Use YOLO pixel masks to index into ZED point cloud
- **Logic:** For each detected object, extract the 3D points at the masked pixel locations
- **Result:** Clean per-object 3D point cluster (no background table/wall points)
- **No separate 3D model needed** — the 2D mask defines which 3D points belong to each object

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

- Single model (YOLO World Seg) runs detection + segmentation
- 3D extraction is a direct point cloud lookup (no inference, microseconds)
- Target: 15+ FPS fused output
- Frame strategy: grab latest from ZED, drop stale frames

## Coordinate Frame

Camera frame (ZED default) for v1. Robot base / world frame deferred to v2.
