# Roadmap

## V1 Phases

### Phase 1: Camera Interface
- Connect to ZED Mini, grab RGB + point cloud
- Display RGB with OpenCV, render point cloud with Open3D
- Record an SVO file for offline dev

### Phase 2: 2D Detection (standalone)
- Load YOLOE, run on ZED RGB frames
- Draw bounding boxes + labels on OpenCV window

### Phase 3: 3D Extraction via Segmentation Masks
- Upgrade YOLOE to YOLOE Seg (pixel-level masks)
- Project masked pixels into ZED point cloud → per-object 3D clusters
- Color each object's points by label in Open3D

### Phase 4: Fused Output
- Wire detection + 3D extraction into single pipeline
- Output FusedObject dataclass (label + box + 3D cluster + centroid)

### Phase 5: Visualization + Validation
- Side-by-side: OpenCV (2D boxes) + Open3D (colored point clouds)
- Run 5-object tabletop test (mug, phone, cup, fork, bottle)
- Measure centroid error (target < 5cm)

## V2 Phases

### Phase 5: Shape Completion (PoinTr)
- Integrate PoinTr (ShapeNet55) for completing partial point clouds
- FPS downsample + normalize → PoinTr inference → denormalize back to camera frame
- Per-label caching (3cm threshold) to avoid redundant computation on static scenes
- Third Open3D window showing completed shapes (lighter color = completed points)
- Toggle completion on/off with 'c' key at runtime

### Phase 6: Cloud 3D Generation (Hunyuan3D Full) ✓
- Evaluated 6 image-to-3D models (PoinTr, TripoSR, TRELLIS, InstantMesh, Hunyuan3D Mini/Turbo, Hunyuan3D Full)
- Only Hunyuan3D Full produces semantically correct shapes (hollow interior, through-hole handle)
- Deployed on Modal serverless (A100 80GB): ~28s/object, 5.5x faster than local RTX 3070
- Integration: YOLOE crop → Modal API → Hunyuan3D Full → mesh → sample points → align to partial cloud
- Files: `src/cloud_hunyuan3d.py`, `src/test_hunyuan3d_cloud.py`

### Phase 7: TensorRT Optimization
- Convert YOLOE and/or PoinTr to TensorRT for faster inference
- Target: 20+ FPS with completion enabled

### Phase 8: ROS2 Integration
- Publish FusedObject as ROS2 messages
- Robot base / world coordinate frame transform

### Phase 9: Fine-tuning
- Fine-tune PoinTr on target environment objects
- Evaluate AdaPoinTr when weights become available
