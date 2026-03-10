# Experiments Log

## Experiment 1: PoinTr (ShapeNet55) for Shape Completion

**Date:** 2026-03-10
**Goal:** Complete partial point clouds (front-facing only from ZED) into full 3D shapes for grasp planning.

### Setup

- **Model:** PoinTr (ICCV 2021 Oral), transformer-based point cloud completion
- **Checkpoint:** `PoinTr_ShapeNet55.pth` -- trained on all 55 ShapeNet categories
- **Input:** 2,048 points (FPS-downsampled from ~11K partial cloud, normalized to unit sphere)
- **Output:** 8,192 completed points (denormalized back to camera frame)
- **Inference time:** ~30-50ms per object on RTX 3070
- **Repo:** github.com/yuxumin/PoinTr
- **CUDA extensions built:** chamfer_dist, pointnet2_ops, knn_cuda (CUDA 12.8, SM 8.6)

### Results

**Test object:** Red mug on desk, captured live from ZED Mini

#### Before outlier removal
- Significant stray points behind the mug caused by depth bleeding at mask edges
- Stereo matching assigns background depth to object boundary pixels
- The scattered points extended 20-50cm behind the actual object surface

#### After outlier removal (two-stage MAD + statistical)
- Partial point cloud is much cleaner
- Mug shape clearly visible in the "3D Objects" window

#### Shape completion output
- PoinTr produces a completed shape that roughly resembles a mug
- The completed points (lighter shade in "Completed Shapes" window) fill in behind the partial cloud
- **However, the completion quality is disappointing:**
  - The completed geometry is blobby and doesn't match the actual mug shape
  - Hallucinated points don't align well with the real object
  - The handle region is poorly reconstructed
  - Overall shape looks like a rough approximation, not a clean mug

### Screenshots

#### Run 1: Before outlier removal
![Before outlier removal](screenshots/pointr_before_outlier_remover.png)
- "3D Objects" (top right): partial cloud with stray depth-bleed points trailing behind the mug
- "Completed Shapes" (bottom left): PoinTr output -- blobby, with noise from the stray input points
- "Full Point Cloud" (bottom right): full scene for reference

#### Run 2: After outlier removal, completion enabled
![After outlier removal with completion](screenshots/pointr_after_outlier_remover.png)
- "3D Objects" (center): clean partial cloud of the mug, no stray points
- "Completed Shapes" (left): PoinTr completion -- fills in back of mug but shape is approximate
- FPS: 12.8 with completion ON
- The mug is recognizable but the completed geometry is rough and blobby

### Analysis: Why PoinTr Underperforms on Real Data

**Root cause: synthetic-to-real domain gap**

1. **Orientation mismatch:** PoinTr was trained on ShapeNet objects in canonical pose (mug upright, handle in consistent direction). Our ZED captures are in camera frame -- the mug is rotated/tilted arbitrarily. PoinTr has no category input, so it must guess the shape purely from geometry. A mug in camera frame looks nothing like ShapeNet's canonical partials.

2. **Partial view pattern:** ShapeNet training generates partial views by removing clean viewpoint slices from complete synthetic meshes. Real ZED partials only capture the front-facing surface with uneven density (denser near camera center, sparser at edges). The "shape" of the missing region is very different from what the model was trained on.

3. **Point distribution:** ShapeNet points are uniformly sampled from clean meshes. ZED points are noisy, have variable density, and include stereo artifacts even after outlier removal.

4. **No category conditioning:** PoinTr is category-agnostic -- it doesn't know it's looking at a mug. It hallucinates geometry based on whatever the partial cloud vaguely resembles in its training distribution.

5. **Scale/proportion:** Real objects have different proportions than ShapeNet's idealized CAD models.

### Conclusion

PoinTr with ShapeNet55 weights is **not suitable** for real-time shape completion on live ZED sensor data. The synthetic-to-real domain gap is too large. The model was designed for and evaluated on synthetic benchmarks (ShapeNet, PCN), not real-world noisy partial scans.

### Next Steps

Exploring **image-to-3D models** instead of point-cloud-only completion. These models take an RGB image crop as input and generate a full 3D mesh, trained on massive real/rendered image datasets (Objaverse, 800K+ objects). No synthetic-to-real gap since they understand real images natively.

**Candidates:**

| Model | Input | Output | Speed | Why it might work |
|-------|-------|--------|-------|-------------------|
| TripoSR (Stability AI) | Single RGB crop | 3D mesh | ~0.5s | Fast, MIT license, trained on Objaverse |
| InstantMesh | Single RGB crop | 3D mesh | ~10s | Higher quality, slower |
| Trellis (Microsoft) | Image + text | 3D mesh / Gaussian | ~5s | State-of-art, multimodal |
| Hunyuan3D 2.0 (Tencent) | Text + image | 3D mesh | ~10s | Uses both text label and RGB crop |

**Advantage over PoinTr:** We already have the RGB crop from YOLOE's bounding box -- it's free input. RGB carries much richer shape information than a noisy partial point cloud. These models also handle arbitrary viewpoints since they're trained on diverse camera angles.

---

## Experiment 2: TBD

(Next shape completion approach)
