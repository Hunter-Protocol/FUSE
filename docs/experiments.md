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

## Experiment 3: Hunyuan3D-2 (Image-to-3D) for Shape Completion

**Date:** 2026-03-10
**Goal:** Test Hunyuan3D-2 (Tencent) as a higher-quality replacement for TripoSR. Despite being image-only (no text conditioning), evaluate whether its larger model and training data produce semantically correct shapes. Compare three variants: Full, Mini, and Turbo.

### Setup

- **Model family:** Hunyuan3D-2 (Tencent, 2024), DiT-based flow matching for 3D generation
- **Architecture:** DINOv2 image encoder + DiT diffusion transformer + VAE shape decoder + marching cubes mesh extraction
- **Training data:** Objaverse (800K+ 3D objects)
- **Input:** Single RGBA image crop (512x512), object isolated via YOLOE mask, **no text prompt**
- **Pipeline:** `Hunyuan3DDiTFlowMatchingPipeline` — shape generation only (no texture stage)
- **Repo:** github.com/Tencent-Hunyuan/Hunyuan3D-2

**Important note on input:** Hunyuan3D-2's shape generation pipeline **only accepts an image** — there is no text/prompt parameter in the API. We passed only the RGB crop with no text conditioning whatsoever. Despite this, the model correctly inferred that the object is a mug and generated functionally correct geometry (hollow interior, graspable handle). This means **semantic inference is already happening purely from visual features** — the DINOv2 encoder + diffusion model learned enough about object categories from Objaverse training to reconstruct semantically meaningful shapes from appearance alone.

### Variants Tested

| Variant | Model ID | Subfolder | Weights | Params |
|---------|----------|-----------|---------|--------|
| **Full** | `tencent/Hunyuan3D-2` | `hunyuan3d-dit-v2-0` | ~9.2GB | Full size |
| **Mini** | `tencent/Hunyuan3D-2mini` | `hunyuan3d-dit-v2-mini` | ~6.5GB | 0.6B |
| **Turbo** | `tencent/Hunyuan3D-2mini` | `hunyuan3d-dit-v2-mini-turbo` | ~6.5GB | 0.6B (distilled) |

```bash
python test_hunyuan3d.py          # Full
python test_hunyuan3d.py --mini   # Mini
python test_hunyuan3d.py --turbo  # Turbo
```

### Comparison: Inference Time (RTX 3070, 8GB VRAM)

**Test object:** Dark/gray mug on desk, captured live from ZED Mini

| Stage | Full | Mini | Turbo |
|-------|------|------|-------|
| Model loading | 24.4s | 18.5s | 17.7s |
| Diffusion sampling | 46.8s (50 steps, ~0.94s/step) | 9.0s (50 steps, ~0.18s/step) | 5.3s (8 steps, ~0.66s/step) |
| Volume decoding / mesh extraction | 101s (7134 chunks) | ~75s (7134 chunks) | ~104s (7134 chunks) |
| **Total shape generation** | **~151s** | **~87s** | **~109s** |
| Output mesh | 812K verts, 1.6M faces | 791K verts, 1.58M faces | 703–862K verts, 1.4–1.7M faces |

### Comparison: Mesh Quality

| Aspect | Full | Mini | Turbo |
|--------|------|------|-------|
| Recognizable as mug | Yes | Yes (roughly) | Barely — more like a cup/bucket |
| Hollow interior | Yes | Yes (attempted) | Yes |
| Hollow handle (graspable) | Yes | No — malformed | No — handle gap closed/fused |
| Surface smoothness | Excellent | Moderate | Poor — rough, bumpy surface |
| Denoising quality | Excellent | Moderate | Poor — noisy mesh artifacts |

### Comparison: Alignment to ZED Partial Cloud

| Metric | Full | Mini | Turbo |
|--------|------|------|-------|
| Scale factor | 0.0459 | 0.044–0.046 | 0.047–0.048 |
| RANSAC fitness | 0.134 | 0.255–0.259 | 0.231–0.241 |
| ICP fitness | 0.062 | 0.126–0.192 | 0.152–0.173 |
| Visual overlap | Poor | Moderate | Moderate |

### Results: Full Model (Hunyuan3D-2)

#### Mesh Quality: Excellent

Hunyuan3D-2 produced a **semantically correct mug** from a single front-facing image:
- **Hollow interior** — the inside of the mug is open, as expected for holding liquid
- **Hollow handle** — the loop under the handle has a hole for gripping
- **Cylindrical body** — correct proportions and shape
- **Smooth surface** — the model also does a great job denoising; the mesh is clean and artifact-free despite the noisy/low-res input crop

This is a massive improvement over TripoSR, which produced a solid boxy wedge.

#### Alignment: Poor

- Attempt 1: Clouds completely separated, wrong orientation
- Attempt 2: Better overlap but still misaligned — cyan (Hunyuan3D) forms a ring, red (ZED partial) clustered in center
- The alignment algorithm (FPFH + RANSAC + ICP) struggles with the large shape difference between full mesh (all sides) and partial cloud (front face only)

### Results: Mini Model (Hunyuan3D-2mini)

#### Inference Time: ~1.8x faster than Full

Mini's diffusion sampling is **5.3x faster** (9s vs 47s), but volume decoding remains the bottleneck (~75s vs ~101s), limiting the overall speedup to ~1.8x.

#### Mesh Quality: Recognizable but degraded

Across two runs, the Mini model produces a shape that is **recognizably a mug** — the cylindrical body and hollow interior are present. However, the handle is **malformed in both attempts**:
- **Run 1:** Handle is a thick, blocky protrusion fused to the body — no hole for gripping, more like a fin than a handle
- **Run 2:** Handle is an angular wedge attached to the side — wrong proportions, no through-hole, not graspable

The body geometry is rougher than the Full model — less smooth surfaces, more angular artifacts. The model clearly "knows" it's generating a mug (hollow interior, handle region) but lacks the capacity to produce fine geometric details like a proper handle loop.

#### Alignment: Better than Full

Surprisingly, Mini's alignment scores are consistently better than Full's (RANSAC 0.255–0.259 vs 0.134, ICP 0.126–0.192 vs 0.062). This may be because Mini's simpler geometry has fewer outlier points that confuse the registration algorithm.

#### Screenshots

##### Mini Run 1
![Hunyuan3D Mini run 1](screenshots/HunYuan3D_mini_attempt_1_with_error.png)
- **Left (Mesh):** Mug body visible but handle is a thick blocky protrusion — no hole, not graspable
- **Center (Alignment):** RED = ZED partial, CYAN = Hunyuan3D Mini — moderate overlap on the body, but handle points scattered
- **Right (Input):** RGBA crop of mug, clean mask isolation

##### Mini Run 2
![Hunyuan3D Mini run 2](screenshots/HunYuan3D_mini_attempt_2_with_error.png)
- **Left (Mesh):** Mug body with angular wedge handle — wrong shape, no through-hole
- **Center (Alignment):** Better overlap on the cylindrical body, handle region still misaligned
- **Right (Input):** Same mug from slightly different capture

### Results: Turbo Model (Hunyuan3D-2mini-turbo)

#### Inference Time: Faster diffusion, slower volume decoding

Turbo uses consistency distillation with only **8 diffusion steps** (vs 50 for Mini/Full), bringing diffusion time down to **5.3s** — the fastest of all three variants. However, volume decoding is **slower than Mini** (~104s vs ~75s), resulting in a total of **~109s** — faster than Full (151s) but slower than Mini (87s). The turbo latent may produce a more complex occupancy field that takes longer to decode.

#### Mesh Quality: Worst of the three variants

The Turbo model produces the **lowest quality meshes** of all Hunyuan3D variants tested:

- **Run 1:** The mesh is recognizable as a mug-like shape with a hollow interior, but the handle is **completely fused** — the gap between the handle and the body is closed, creating a solid loop rather than a graspable handle. The surface has visible bumps and artifacts. The model failed to recognize the handle as a separate graspable feature.
- **Run 2:** Quality is even worse — the mesh is a rough, bumpy bucket-like shape with heavy surface noise. The handle region is a solid wedge fused to the body. The surface artifacts suggest the 8-step consistency distillation produces noisier latents that the VAE decoder struggles to clean up.

Both runs show that Turbo **fails to recognize the object as a mug** in the way that Full does. While Full produces a semantically correct mug (hollow interior, through-hole handle, smooth cylinder), Turbo produces a generic cup/bucket shape with a solid protrusion where the handle should be. The consistency distillation trades too much quality for speed.

#### Alignment: Comparable to Mini

Alignment scores (RANSAC 0.231–0.241, ICP 0.152–0.173) are similar to Mini's and significantly better than Full's. As with Mini, the simpler geometry may make registration easier, but the aligned shape doesn't match the actual object well.

#### Screenshots

##### Turbo Run 1
![Hunyuan3D Turbo run 1](screenshots/HunYuan3D_turbo_attempt_1_with_error.png)
- **Left (Mesh):** Mug-like shape but handle gap is completely closed/fused — no through-hole for gripping
- **Center (Alignment):** RED = ZED partial, CYAN = Hunyuan3D Turbo — moderate overlap on the body
- **Right (Input):** RGBA crop of mug, clear handle visible in input image

##### Turbo Run 2
![Hunyuan3D Turbo run 2](screenshots/HunYuan3D_turbo_attempt_2_with_error.png)
- **Left (Mesh):** Rough, bumpy bucket-like shape — heavy surface noise, handle is a solid wedge
- **Center (Alignment):** Partial overlap, but Turbo shape is clearly wrong proportions
- **Right (Input):** Same mug from slightly different capture

### Screenshots

#### Full Model — Attempt 1
![Hunyuan3D attempt 1](screenshots/HunYuan3D_attempt_1_with_error.png)
- **Left (Mesh):** Top-down view of mug mesh — hollow interior clearly visible, handle with hole
- **Center (Alignment):** RED = ZED partial, CYAN = Hunyuan3D — completely misaligned, different orientations
- **Right (Input):** RGBA crop of mug with mask-isolated background

#### Full Model — Attempt 2
![Hunyuan3D attempt 2](screenshots/HunYuan3D_attempt_2_with_no_error.png)
- **Left (Mesh):** Angled view — cylindrical body, handle, smooth surface
- **Center (Alignment):** Better overlap but still poor fit — cyan ring (full mug) doesn't align with red cluster (front face)
- **Right (Input):** Cleaner crop with better mask isolation

### Inference Time Comparison (All Models Tested)

| Model | Diffusion / Inference | Mesh Extraction | Total | Speedup vs Full |
|-------|----------------------|-----------------|-------|-----------------|
| PoinTr (ShapeNet55) | ~30-50ms | N/A (point cloud) | **~40ms** | ~3800x |
| TripoSR | ~0.7s | ~2.6s | **~3.3s** | ~46x |
| Hunyuan3D-2 Mini | ~9s (50 steps) | ~75s (volume decode) | **~87s** | ~1.7x |
| Hunyuan3D-2 Full | ~47s (50 steps) | ~101s (volume decode) | **~151s** | 1x (baseline) |
| Hunyuan3D-2 Turbo | ~5.3s (8 steps) | ~104s (volume decode) | **~109s** | ~1.4x |

**Key insight:** Volume decoding (marching cubes over 7134 chunks) is the dominant cost for Hunyuan3D variants, accounting for 67-86% of total time. Diffusion speedups from Mini (5.3x) are diluted by this fixed bottleneck. TripoSR is ~26x faster than Mini but produces much worse geometry. PoinTr is fastest but fails on real data due to domain gap.

### Analysis

#### Strengths
1. **Implicit semantic understanding:** Despite being image-only (no text prompt), the model correctly infers that a mug should be hollow, have a graspable handle, and be cylindrical. This knowledge comes from training on 800K+ Objaverse objects.
2. **Excellent denoising:** The output mesh is smooth and clean even from a noisy, low-resolution ZED crop. The diffusion process effectively denoises the input.
3. **High-fidelity geometry:** 812K vertices, 1.6M faces — extremely detailed mesh with correct topology.

#### Problems

1. **Latency:** **~151 seconds per object** (full model) is far too slow for any real-time or near-real-time pipeline. Even as an async background task, this is impractical for a perception system that needs to handle multiple objects. For comparison: PoinTr was ~30ms, TripoSR was ~3.3s. Mini/Turbo variants may improve this.

2. **Output format mismatch:** Hunyuan3D outputs high-poly meshes with textures — beautiful for rendering but **not directly useful for robot manipulation**. The robotics pipeline needs point clouds or simple geometric primitives for grasp planning, collision checking, and motion planning. Converting 1.6M-face meshes to point clouds is wasteful — the mesh representation carries overhead (topology, UV maps, normals) that the robot doesn't need.

3. **Alignment still unsolved:** FPFH + RANSAC + ICP doesn't reliably align the canonical-frame mesh to the camera-frame partial cloud. The full-vs-partial asymmetry (complete mesh vs. front-face-only scan) makes feature matching difficult.

4. **VRAM pressure:** The model uses most of the 8GB VRAM during inference, requiring sequential execution (ZED capture → release GPU → Hunyuan3D → release GPU → visualization). Cannot run alongside real-time perception.

### Future Direction: Finetuning

If we decide to finetune using Hunyuan3D's model architecture, the ideal approach would be to **skip the polygon/mesh extraction layer entirely** and generate point clouds directly from the latent space. This would:
- Eliminate the expensive volume decoding stage (~101s, 67% of total time for full model)
- Output a format directly usable by the robotics pipeline
- Reduce VRAM usage (no marching cubes, no mesh storage)
- Potentially allow much faster inference by targeting fewer output points (e.g., 8192 points vs. 812K vertices)

### Conclusion

Hunyuan3D-2 (full) produces the **best mesh quality** of all models tested — semantically correct, smooth, and detailed. However, **151s latency** makes it completely impractical for the FUSE pipeline, even as a cached background task. The output format (high-poly mesh) is also mismatched with robotics needs (point clouds).

**Mini** (87s) is the fastest variant overall but produces degraded handle geometry — malformed, no through-hole. **Turbo** (109s) is paradoxically slower than Mini despite fewer diffusion steps, because its volume decoding takes ~40% longer. Turbo also has the worst mesh quality — rough surfaces, heavy artifacts, and the model fails to recognize the handle as a graspable feature, closing the gap entirely.

**Ranking by quality:** Full >> Mini > Turbo
**Ranking by speed:** Mini (87s) > Turbo (109s) > Full (151s)

None of the variants are practical for real-time use. Volume decoding (marching cubes over 7134 chunks) dominates all variants at 75–104s, making diffusion speedups largely irrelevant.

**Status:** Best quality (Full), but too slow and wrong output format. Mini is the best speed/quality tradeoff but still impractical at 87s. Promising as a finetuning base if the polygon layer is bypassed.

---

## Experiment 4: TBD

(Next approach — retrieval-based, Shap-E, or Hunyuan3D finetuning)
