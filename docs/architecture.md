# Architecture

## Pipeline Overview

```
ZED Mini (720p, SDK v5.2.1)
  ├── RGB ──> YOLO World (open-vocab, pretrained) ──> label + 2D box
  ├── Point Cloud ──> PointNet++ (semantic, ScanNet-pretrained) ──> 3D segments
  └── Fusion (projection method) ──> unified FusedObject per object
```

## Branches

### 2D Detection (RGB Stream)
- **Model:** YOLO World (open-vocabulary)
- **Input:** Left RGB image (720p)
- **Output:** Per-object label, 2D bounding box, confidence score
- **Speed:** ~30 FPS on RTX 3070

### 3D Segmentation (Point Cloud)
- **Model:** PointNet++ (semantic segmentation, ScanNet-pretrained)
- **Input:** Raw point cloud from ZED (N x 3)
- **Output:** Per-point semantic label
- **Speed:** ~15-20 FPS on RTX 3070

### Fusion Layer
- **Method:** Project 3D points into 2D image space using ZED calibration matrix
- **Logic:** Check if projected point (u, v) falls inside a YOLO bounding box -- if yes, assign that label
- **Conflict resolution:** 2D (YOLO) label wins over 3D (PointNet++) when they disagree
- **Unmatched detections:** Kept from both sides, tagged as `"fused"`, `"2d_only"`, or `"3d_only"`

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

- 2D and 3D branches run **async in parallel threads**
- Fusion runs when both branches complete for a given frame
- Target: 15 FPS fused output, ~100ms end-to-end latency
- Frame strategy: grab latest from ZED, drop stale frames

## Coordinate Frame

Camera frame (ZED default) for v1. Robot base / world frame deferred to v2.
