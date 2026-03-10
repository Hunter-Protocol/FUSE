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

## Experiment 2: TripoSR (Single-Image-to-3D) for Shape Completion

**Date:** 2026-03-10
**Goal:** Replace PoinTr with an image-based 3D reconstruction model. Use the RGB crop from YOLOE's bounding box to generate a full 3D mesh, then align it to the ZED partial point cloud.

### Setup

- **Model:** TripoSR (Stability AI + Tripo, March 2024), single-image-to-3D mesh
- **Backbone:** DINOv2 image encoder + transformer + NeRF-style triplane decoder
- **Training data:** Objaverse (~800K 3D objects rendered from multiple viewpoints)
- **Input:** Single RGB image crop (512x512), white background, object isolated via YOLOE mask
- **Output:** Textured 3D mesh (extracted via marching cubes at resolution 192)
- **Inference time:** ~0.7s inference + ~2.6s mesh extraction = ~3.3s total on RTX 3070
- **Repo:** stabilityai/TripoSR (HuggingFace)
- **Marching cubes:** Patched to use scikit-image (torchmcubes failed to build with CUDA)

### Pipeline

1. YOLOE detects mug → bounding box + segmentation mask
2. Crop RGB using bbox, apply mask with erosion (5x5 kernel) to remove edge bleed
3. Composite masked object onto white background, pad to square, resize to 512x512
4. TripoSR inference → scene code → mesh extraction (resolution=192, chunk_size=4096)
5. Clean mesh: remove disconnected components, keep largest
6. Sample 8,192 points from mesh surface
7. Align to ZED partial cloud: scale matching (bbox diagonal ratio) → FPFH feature extraction → RANSAC global registration → ICP refinement

### Results

**Test object:** Dark/gray mug on desk, captured live from ZED Mini

#### Attempt 1: Bbox crop + gray background (145x145 upscaled)
- **Input:** Tight bbox crop at native resolution (~145px), composited onto 50% gray background
- **Mesh output:** Boxy wedge shape, not recognizable as a mug
- **Alignment:** Brute-force rotation + ICP, fitness=0.000 (zero correspondences)
- **Cause:** Crop too small (145px upscaled = very blurry), gray background wrong for model, background elements visible in crop

#### Attempt 2: Mask-isolated crop + white background (512x512)
- **Input:** YOLOE mask with erosion, white background, resized to 512x512
- **Mesh output:** Still a boxy/wedge shape, not recognizable as a mug
- **Alignment:** FPFH + RANSAC (fitness=0.282, 886 correspondences) → ICP (fitness=0.204, RMSE=0.0018m)
- **Improvement:** Alignment now finds correspondences, but the mesh shape is still wrong

### Screenshots

#### Attempt 2 (latest)
![TripoSR attempt 2](screenshots/triposr_attempt2.png)
- **Bottom-left (Input):** Mug crop with white background — mask isolation working, but pixelated due to upscaling from ~145px native resolution
- **Top-left (Mesh):** TripoSR canonical mesh — boxy wedge, no mug-like features, no handle visible
- **Top-right (Alignment):** RED = ZED partial cloud, CYAN = TripoSR sampled points — partial overlap but shapes don't match

### Analysis: Why TripoSR Fails on This Input

1. **No semantic understanding:** TripoSR is purely geometry-from-appearance — it has no concept of what a "mug" is. It doesn't know a mug should have a hollow interior (for holding liquid), a hole under the handle (for gripping), or cylindrical symmetry. It just reconstructs whatever 3D shape matches the image silhouette, producing a solid blob instead of a functional object. This is a fundamental architectural limitation — the model lacks category/text conditioning.

2. **Low effective resolution:** The mug bbox in the 720p frame is only ~145 pixels across. Upscaling to 512x512 produces a blurry, pixelated input. TripoSR was trained on sharp, high-quality renders — blurry inputs confuse the DINOv2 encoder.

3. **Dark/reflective surface:** The mug is dark gray/metallic with specular reflections. TripoSR struggles with reflective objects because the appearance is view-dependent and doesn't match Objaverse's mostly diffuse training objects.

4. **Mask edge artifacts:** Even with erosion, the mask edges have staircase artifacts (jagged pixels). At low resolution, these artifacts take up a significant portion of the object boundary.

5. **Objaverse training bias:** TripoSR was trained on Objaverse renders with clean studio lighting and centered objects. Real ZED crops have uneven lighting, motion blur, and sensor noise.

6. **Handle occlusion:** The mug handle is partially self-occluded and dark, making it hard for the model to infer handle geometry from the silhouette alone.

### Potential Improvements

- **Higher resolution capture:** Use ZED at 1080p or 2K mode so the mug bbox is 300+ pixels natively
- **Better background removal:** Use rembg (U2-Net) for cleaner segmentation instead of YOLOE mask
- **Lighting normalization:** Apply histogram equalization or white-balance correction before feeding to TripoSR
- **Text-conditioned models:** Use models that accept text + image (e.g., Hunyuan3D 2.0, Trellis) so the model knows it's reconstructing a "mug" and can infer semantic features like hollow interior and handle hole
- **Multi-view:** Capture from 2-3 viewpoints and use a multi-view reconstruction model

### Conclusion

TripoSR produces poor mesh quality on real ZED crops at 720p resolution. The primary bottleneck is input quality — the mug occupies too few pixels in the 720p frame, and upscaling introduces blur that the model can't recover from. The mesh doesn't resemble the target object, making downstream alignment meaningless regardless of the registration algorithm used.

**Status:** Not viable in current form. Need higher resolution input or a different approach.

---

## Experiment 3: TBD

(Next approach — higher resolution capture, alternative model, or multi-view)
