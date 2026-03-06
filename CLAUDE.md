# Project FUSE

Real-time 2D/3D perception pipeline: ZED stereo camera -> unified per-object output (label + 2D box + 3D point cluster + centroid).

## Architecture

```
ZED Mini (720p, SDK v5.2.1)
  ├── RGB ──> YOLO World Seg (open-vocab, pretrained) ──> label + 2D box + pixel mask
  ├── Depth/Point Cloud ──> mask projection ──> per-object 3D point cluster
  └── FusedObject per object (label + box + 3D cluster + centroid)
```

## Key Decisions (v1)

- **Hardware:** RTX 3070 (8GB VRAM), Razer Blade 15
- **Language:** Python only, standalone (no ROS2)
- **Inference:** PyTorch, no TensorRT yet
- **3D extraction:** YOLO World Seg pixel masks → project masked pixels into 3D via ZED point cloud → per-object 3D cluster
- **No separate 3D model for v1:** PointNet++ dropped — pretrained 3D models don't cover small household objects. Single YOLO World Seg model handles both 2D detection and 3D extraction.
- **Coordinate frame:** Camera frame
- **Shape completion:** Deferred to v2
- **Branches run async** (parallel threads), target 15 FPS, ~100ms latency
- **Frame strategy:** Grab latest from ZED, drop stale

## V1 Output Per Object

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

**Point cloud coloring:** Each object label gets a unique color. All points belonging to that object are rendered in that color in Open3D (e.g., mug=red, phone=yellow, cup=green).

## V1 Success Criteria

Detect and locate 5 household objects on a table (mug, phone, cup, fork, bottle). Centroid error < 5cm.

## Workflow Rules

- Update `docs/` folder after major milestones and major additions to the project

## Dev Workflow

- Record SVO files from ZED for offline development
- OpenCV for 2D overlay, Open3D for 3D visualization (real-time)
- Metrics: IoU (2D boxes), centroid error in cm (3D)

## V2 Backlog

- Dedicated 3D segmentation model (Mask3D or PointNet++)
- Shape completion (AdaPoinTr)
- TensorRT optimization
- ROS2 integration
- Robot base / world coordinate frame
- Fine-tuning on target environment
