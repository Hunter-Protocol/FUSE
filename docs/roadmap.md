# Roadmap

## V1 — Real-Time 2D/3D Perception

### Phase 1: Camera Interface — 2026-03-05 ✓
- Connect to ZED Mini, grab RGB + point cloud
- Display RGB with OpenCV, render point cloud with Open3D
- Record an SVO file for offline dev

### Phase 2: 2D Detection (standalone) — 2026-03-05 ✓
- Load YOLOE, run on ZED RGB frames
- Draw bounding boxes + labels on OpenCV window

### Phase 3: 3D Extraction via Segmentation Masks — 2026-03-05 ✓
- Upgrade YOLOE to YOLOE Seg (pixel-level masks)
- Project masked pixels into ZED point cloud → per-object 3D clusters
- Color each object's points by label in Open3D

### Phase 4: Fused Output — 2026-03-05 ✓
- Wire detection + 3D extraction into single pipeline
- Output FusedObject dataclass (label + box + 3D cluster + centroid)
- FPS optimization (~16 FPS)

### Phase 5: Visualization + Validation — 2026-03-10 ✓
- Side-by-side: OpenCV (2D boxes) + Open3D (colored point clouds)
- Two-stage outlier removal for depth-bleeding at mask edges (MAD + statistical)
- Run 5-object tabletop test (mug, phone, cup, fork, bottle)

## V2 — Shape Completion & 3D Generation

### Phase 6: Shape Completion — PoinTr (Attempted & Dropped) — 2026-03-10 ✗
- Integrated PoinTr (ShapeNet55) for completing partial point clouds
- FPS downsample + normalize → PoinTr inference → denormalize back to camera frame
- Per-label caching (3cm threshold), third Open3D window, toggle with 'c' key
- **Dropped:** synthetic-to-real domain gap made completions unusable on live ZED data (orientation mismatch, partial view pattern, no category conditioning). See `docs/architecture/overall_architecture.md`.

### Phase 7: Model Evaluation — 2026-03-10 to 2026-03-11 ✓
- Evaluated 6 image-to-3D models across local RTX 3070 and cloud A100:
  - PoinTr (2026-03-10) — domain gap, no image conditioning
  - TripoSR (2026-03-10) — solid blobs, no topology understanding
  - Hunyuan3D Full/Mini/Turbo locally (2026-03-10) — Full best quality but 151s, Mini/Turbo degraded handles
  - TRELLIS on cloud A100 (2026-03-11) — inconsistent quality, sparse octree limits resolution
  - InstantMesh on cloud A100 (2026-03-11) — wrong shapes, intermediate view bottleneck
  - Hunyuan3D Full on cloud A100 (2026-03-11) — **best quality, 28s, chosen model**
- Identified three architectural requirements for semantic understanding: direct image conditioning + DiT global attention + sufficient capacity
- Full analysis in `docs/experiments.md`

### Phase 8: Cloud 3D Generation — Hunyuan3D Full on Modal A100 — 2026-03-11 ✓
- Deployed Hunyuan3D Full on Modal serverless (A100 80GB)
- ~28s/object (5.5x faster than local RTX 3070)
- Integration test: YOLOE crop → Modal API → Hunyuan3D Full → mesh → sample points → align to partial cloud
- Files: `src/cloud_hunyuan3d.py`, `src/test_hunyuan3d_cloud.py`

### Phase 9: Cleanup & Documentation — 2026-03-12 ✓
- Removed source files for all 5 rejected models
- Removed PoinTr from pipeline (domain gap)
- Split architecture docs: `overall_architecture.md` + `modal_architecture.md`
- Comprehensive `decisions.md` with rationale for every major choice
- Updated all docs to reflect current state

## Future — Fine-Tuning & Production

### Phase 10: Hunyuan3D Decoder Fine-Tuning
- Replace VAE occupancy decoder + marching cubes with direct point cloud decoder
- Freeze DINOv2 encoder + DiT diffusion, train only transformer point decoder head
- Generate training data: extract 3D latents from existing pipeline on ~1,000-5,000 Objaverse objects
- Target: ~10-14s total generation (down from 28s) by eliminating 17s volume decoding bottleneck
- See `docs/architecture/modal_architecture.md` for full plan

### Phase 11: Alignment
- Solve partial-to-complete cloud registration (current FPFH + RANSAC + ICP is unreliable)
- Stretch: viewpoint-aware decoder that outputs points already in camera frame

### Phase 12: TensorRT Optimization
- Convert YOLOE to TensorRT for faster local inference
- Target: 20+ FPS on RTX 3070

### Phase 13: ROS2 Integration
- Publish FusedObject as ROS2 messages
- Robot base / world coordinate frame transform
