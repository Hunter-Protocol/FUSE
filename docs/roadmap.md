# Roadmap

## V1 Phases

### Phase 1: Camera Interface
- Connect to ZED Mini, grab RGB + point cloud
- Display RGB with OpenCV, render point cloud with Open3D
- Record an SVO file for offline dev

### Phase 2: 2D Detection (standalone)
- Load YOLO World, run on ZED RGB frames
- Draw bounding boxes + labels on OpenCV window

### Phase 3: 3D Segmentation (standalone)
- Load PointNet++ (ScanNet-pretrained), run on ZED point cloud
- Color points by semantic label in Open3D

### Phase 4: Fusion
- Wire both branches with async threads
- Project 3D points into 2D, assign YOLO labels
- Output FusedObject dataclass

### Phase 5: Visualization + Validation
- Side-by-side: OpenCV (2D boxes) + Open3D (colored point clouds)
- Run 5-object tabletop test (mug, phone, cup, fork, bottle)
- Measure centroid error (target < 5cm)

## V2
TBD
