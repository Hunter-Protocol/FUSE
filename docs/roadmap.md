# Roadmap

## V1 Phases

### Phase 1: Camera Interface
- Connect to ZED Mini, grab RGB + point cloud
- Display RGB with OpenCV, render point cloud with Open3D
- Record an SVO file for offline dev

### Phase 2: 2D Detection (standalone)
- Load YOLO World, run on ZED RGB frames
- Draw bounding boxes + labels on OpenCV window

### Phase 3: 3D Extraction via Segmentation Masks
- Upgrade YOLO World to YOLO World Seg (pixel-level masks)
- Project masked pixels into ZED point cloud → per-object 3D clusters
- Color each object's points by label in Open3D

### Phase 4: Fused Output
- Wire detection + 3D extraction into single pipeline
- Output FusedObject dataclass (label + box + 3D cluster + centroid)

### Phase 5: Visualization + Validation
- Side-by-side: OpenCV (2D boxes) + Open3D (colored point clouds)
- Run 5-object tabletop test (mug, phone, cup, fork, bottle)
- Measure centroid error (target < 5cm)

## V2
TBD
