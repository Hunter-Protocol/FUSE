# Modal Architecture — Hunyuan3D Full

Cloud 3D generation: single image → semantically correct 3D mesh via Hunyuan3D Full on Modal serverless A100.

## Current Architecture

### High-Level Pipeline

```
Local (RTX 3070)                              Cloud (Modal A100 80GB)
────────────────                              ──────────────────────────
ZED RGB frame
  → YOLOE Seg detect + mask
  → Crop object region (RGBA, 512x512)
  → Encode as PNG bytes
  → Upload via Modal RPC ───────────────────→ Hunyuan3DModel.generate()
                                                │
                                                ▼
                                              ┌─────────────────────────────────┐
                                              │  DINOv2 Encoder                 │
                                              │  ViT-Giant (1.1B params)        │
                                              │  Input: RGBA image              │
                                              │  Output: visual feature vectors │
                                              │  (structural + spatial features │
                                              │   learned via self-supervised   │
                                              │   masked image modeling on      │
                                              │   142M images)                  │
                                              └──────────────┬──────────────────┘
                                                             │
                                                      feature vectors
                                                             │
                                              ┌──────────────▼──────────────────┐
                                              │  DiT Denoiser                   │
                                              │  (Diffusion Transformer)        │
                                              │                                 │
                                              │  50 flow-matching steps         │
                                              │  Self-attention over entire     │
                                              │  3D latent space                │
                                              │  Conditioned on DINOv2 features │
                                              │                                 │
                                              │  Iteratively denoises random    │
                                              │  noise → meaningful 3D latent   │
                                              │  ~9s on A100 (~5.6 it/s)        │
                                              └──────────────┬──────────────────┘
                                                             │
                                                        3D latent
                                                  (compressed shape repr,
                                                   ~3072 × 64 floats)
                                                             │
                                              ┌──────────────▼──────────────────┐
                                              │  VAE Decoder                    │
                                              │                                 │
                                              │  Evaluates latent on dense 3D   │
                                              │  grid → occupancy field         │
                                              │  (solid or empty at each point) │
                                              │                                 │
                                              │  7,134 chunks processed         │
                                              │  sequentially                   │
                                              │  ~17s on A100 (~420 it/s)       │
                                              └──────────────┬──────────────────┘
                                                             │
                                                      occupancy field
                                                             │
                                              ┌──────────────▼──────────────────┐
                                              │  Marching Cubes                 │
                                              │                                 │
                                              │  Traces surface boundary from   │
                                              │  occupancy field → triangle     │
                                              │  mesh (~600K verts, ~1.2M faces)│
                                              └──────────────┬──────────────────┘
                                                             │
                                                    mesh (vertices + faces)
                                                             │
                                              ┌──────────────▼──────────────────┐
                                              │  Point Sampling                 │
                                              │  trimesh.sample_surface → 8192  │
                                              │  points from mesh surface       │
                                              └──────────────┬──────────────────┘
                                                             │
  ← Download result dict (JSON) ←────────────────────────────┘
     vertices, faces, points, timing

  → Align sampled points to ZED partial cloud (FPFH + RANSAC + ICP)
  → Visualize
```

### Component Breakdown

#### DINOv2 Encoder — "What do I see?"

A self-supervised Vision Transformer (ViT-Giant, 40 layers, 1.1B params) pretrained by Meta on 142M images. Unlike supervised classifiers that only learn category labels, DINOv2 learns rich visual features through self-distillation: it masks parts of an image and trains to predict the missing regions, forcing it to understand object structure, part relationships, and 3D geometry from 2D images.

When Hunyuan3D feeds a mug crop through DINOv2, the encoder doesn't just see "pixels" — it extracts features that encode "cylindrical object with a protruding loop on the side." These are **emergent properties**: the model was never told about object parts, but learned them because they're useful for the self-supervised reconstruction task. Research has shown DINOv2 features naturally cluster by object parts and can segment objects without any labels.

#### DiT Denoiser — "What 3D shape matches these features?"

A Diffusion Transformer that replaces the U-Net traditionally used in diffusion models. DiT uses self-attention over the entire 3D latent space, allowing it to reason about long-range spatial relationships (e.g., "the handle connects back to the body at two points, forming a loop").

U-Net architectures process features locally through convolutions, making them worse at capturing global topology like holes and hollow interiors. DiT's global attention is why Hunyuan3D Full correctly generates the handle through-hole while Mini/Turbo (same architecture but fewer parameters/steps) fail to resolve this fine detail.

During training on 800K+ Objaverse (image, 3D shape) pairs, DiT learned the mapping: "when DINOv2 features look like *this pattern*, the 3D shape should look like *that*." The self-attention lets it reason globally: "the features on the left and right sides of the object both look like handle attachment points — they should connect with a loop in between."

#### 3D Latent Space

Instead of working directly with 600K+ vertices, the model operates in a compressed latent space (~3072 × 64 floats). Every point in this space corresponds to a valid 3D shape. DiT denoises in this compressed space, not in full 3D — this is what makes diffusion tractable.

The latent was learned by a VAE trained in two phases:
1. **Train the VAE:** thousands of 3D meshes → encode to latent → decode back → measure reconstruction accuracy
2. **Train DiT in that latent space:** denoise random noise → meaningful latents, conditioned on DINOv2 features

#### VAE Decoder — "Convert latent to actual 3D"

Evaluates the latent at every point in a dense 3D grid to determine "solid or empty?" (occupancy field). Then marching cubes traces the surface boundary to extract a triangle mesh. The 7,134-chunk loop in the logs is the decoder checking sub-volumes of this grid one by one.

**This is the dominant bottleneck.** Volume decoding takes ~17s on A100 (61% of total generation time) and ~101s on RTX 3070 (67% of total time). The bottleneck is I/O-bound: sequentially evaluating thousands of dense sub-volumes through the decoder network.

### Why Only Hunyuan3D Full Produces Correct Shapes

Across 6 models tested, only Hunyuan3D Full consistently produces semantically correct objects (hollow mug interior, through-hole handle). Three requirements must all be met:

1. **Direct image-to-3D conditioning** — no intermediate bottleneck. InstantMesh fails because it routes through 6 synthesized views at 320x320, losing semantic detail.

2. **Global-attention denoiser (DiT)** — not U-Net or local convolutions. TripoSR pairs DINOv2 with a triplane NeRF decoder (local only), producing solid blobs. TRELLIS uses xformers attention but on a sparse octree (limited resolution).

3. **Sufficient model capacity + diffusion steps** — Mini/Turbo have the same architecture but fewer parameters (0.6B vs full) and fewer/distilled steps. They "know" it's a mug but can't resolve the handle through-hole — the most topologically demanding feature.

Remove any one requirement and the model fails. Full analysis: `docs/experiments.md`.

### Timing Breakdown

| Stage | RTX 3070 (8GB) | A100 (80GB) | Speedup | % of A100 total |
|-------|---------------|-------------|---------|-----------------|
| Diffusion (50 steps) | ~47s (~0.94s/step) | ~9s (~5.6 it/s) | 5.2x | 32% |
| Volume decoding (7134 chunks) | ~101s (~70 it/s) | ~17s (~420 it/s) | 5.9x | **61%** |
| Marching cubes + sampling | ~3s | ~2s | 1.5x | 7% |
| **Total generation** | **~151s** | **~28s** | **5.5x** | 100% |

### Infrastructure

- **Platform:** Modal serverless
- **GPU:** NVIDIA A100 80GB
- **Container:** `nvidia/cuda:12.1.0-devel-ubuntu22.04`, PyTorch 2.4.0
- **Model weights:** `tencent/Hunyuan3D-2` (HuggingFace), subfolder `hunyuan3d-dit-v2-0`
- **Scale-down:** 60s idle → container terminates (auto-scale to zero)
- **Cost:** ~$0.02/inference (A100 at ~$2.50/hr, ~28s generation)
- **Cold start:** ~100-170s (container spin-up + model download + CUDA init)
- **Warm latency:** ~30-35s (generation + network overhead)

### Files

- `src/cloud_hunyuan3d.py` — Modal app definition, container image, `Hunyuan3DModel` class with `generate()` method
- `src/test_hunyuan3d_cloud.py` — Integration test: ZED crop → cloud endpoint → align → visualize

---

## Fine-Tuning Plan

### Problem Statement

The current Hunyuan3D Full pipeline produces high-quality 3D meshes but has two fundamental issues for the FUSE robotics pipeline:

1. **Output format mismatch:** Hunyuan3D outputs high-poly meshes (~600K vertices, ~1.2M faces) with topology, UV maps, and normals. The robotics pipeline needs **point clouds** for grasp planning, collision checking, and motion planning. We currently sample 8,192 points from the mesh surface post-hoc — the mesh representation is pure overhead.

2. **Volume decoding bottleneck:** The VAE decoder evaluates 7,134 dense sub-volumes to produce an occupancy field, then marching cubes extracts a mesh. This takes **~17s on A100 (61% of total time)**. The DiT diffusion itself only takes ~9s. The latent already encodes the complete shape — we're spending most of our time converting it to a format (mesh) that we then immediately discard in favor of sampled points.

### Proposed Architecture: Direct Point Cloud Decoder

Replace the VAE occupancy decoder + marching cubes pipeline with a lightweight decoder that maps the 3D latent directly to a point cloud:

```
Current pipeline:
  3D latent → [VAE decoder: 7134 chunks → occupancy field] → [marching cubes → mesh] → [sample → 8192 pts]
              ~~~~~~~~~~~~~~~~~ 17s ~~~~~~~~~~~~~~~~~~          ~~~~~ 2s ~~~~~           ~~~ <1s ~~~

Proposed pipeline:
  3D latent → [Point Cloud Decoder → 8192 pts]
              ~~~~~~~~ target: 1-3s ~~~~~~~~~~
```

#### Architecture Options

**Option A: MLP Decoder**

Simple fully-connected layers that map the latent directly to point coordinates:

```
3D latent (3072 × 64) → flatten → FC layers → 8192 × 3 points
```

- Pros: Fast inference, simple to train, minimal VRAM
- Cons: May struggle with complex topology (handles, holes) — MLPs are poor at capturing discontinuous surfaces
- Training data: pairs of (latent, ground-truth point cloud) from existing Hunyuan3D runs

**Option B: Transformer Point Decoder**

Cross-attention from learnable point queries to the 3D latent features:

```
Learnable queries (8192 × D) + 3D latent (3072 × 64)
  → Cross-attention layers (queries attend to latent)
  → Linear projection → 8192 × 3 points
```

- Pros: Attention mechanism can capture global structure and topology. Architecturally similar to PoinTr's decoder — proven approach for point cloud generation. Can condition on different numbers of output points.
- Cons: Slower than MLP, more parameters to train
- This is the preferred approach — maintains the global reasoning capability that makes Hunyuan3D Full work

**Option C: Hybrid — Coarse MLP + Fine Transformer**

```
3D latent → MLP → 1024 coarse points → Transformer upsampler → 8192 fine points
```

- Pros: MLP handles rough shape quickly, transformer refines details
- Cons: Two-stage training, more complex

#### Training Strategy

1. **Generate training pairs:** Run existing Hunyuan3D Full pipeline on diverse objects (Objaverse subset). For each: save the 3D latent (output of DiT, before VAE decoder) + the final mesh point cloud (ground truth). This builds a dataset of (latent → point cloud) pairs.

2. **Freeze DiT + DINOv2:** Only train the new point cloud decoder. The encoder and diffusion model already produce excellent latents — we only need to replace the decoder head.

3. **Loss function:** Chamfer Distance between predicted and ground-truth point clouds. Optionally add Earth Mover's Distance for more uniform coverage.

4. **Training data scope:**
   - Start with FUSE target categories: mugs, bottles, phones, cups, forks (and close ShapeNet equivalents)
   - Expand to broader Objaverse categories if the decoder needs to generalize
   - ~1,000-5,000 objects should be sufficient for initial training given the frozen encoder

#### Expected Impact

| Metric | Current (VAE + marching cubes) | Target (point cloud decoder) |
|--------|-------------------------------|------------------------------|
| Decoder time | ~17s (A100) | ~1-3s (A100) |
| Total generation | ~28s | ~10-14s |
| Output format | 600K vertex mesh → sample 8192 pts | 8192 pts directly |
| VRAM (decoder) | High (dense 3D grid evaluation) | Low (single forward pass) |
| Mesh available? | Yes | No (point cloud only) |

#### Open Questions

1. **Latent extraction:** Can we hook into the Hunyuan3D pipeline to extract the 3D latent before the VAE decoder, or do we need to modify the source? The `hy3dgen` library's `Hunyuan3DDiTFlowMatchingPipeline` may expose intermediate outputs.

2. **Quality retention:** Will a point cloud decoder maintain the semantic quality (handle topology, hollow interiors) that makes Hunyuan3D Full valuable? The latent should encode this — the question is whether the decoder can faithfully extract it as points.

3. **Training compute:** Generating training pairs requires running the full pipeline (~28s/object on A100). For 5,000 objects: ~39 GPU-hours (~$97 on Modal). Training the decoder itself should be much cheaper.

4. **Local inference:** With the lighter decoder, total inference (~10-14s on A100) might become feasible on RTX 3070 if the DiT diffusion can run in float16 without quality loss. DiT alone took ~47s on 3070 — still too slow for real-time, but potentially acceptable for a cached background task.

5. **Alignment:** Current FPFH + RANSAC + ICP alignment between the generated shape and ZED partial cloud is poor. A point cloud decoder trained on partial→complete pairs (rather than canonical objects) could potentially learn to output shapes already roughly aligned to the input viewpoint, reducing or eliminating the alignment step.

#### Phased Approach

**Phase 1: Latent dataset generation**
- Modify `cloud_hunyuan3d.py` to save 3D latents alongside mesh outputs
- Run on ~1,000 diverse objects from Objaverse
- Validate latent quality by decoding a few with the existing VAE

**Phase 2: Decoder training**
- Implement transformer point decoder
- Train on (latent, point cloud) pairs with Chamfer Distance loss
- Evaluate: point cloud quality vs mesh-sampled ground truth

**Phase 3: Integration**
- Replace VAE decoder + marching cubes in the Modal endpoint
- Update `test_hunyuan3d_cloud.py` to consume point clouds directly
- Benchmark end-to-end latency improvement

**Phase 4: Viewpoint-aware training (stretch)**
- Train decoder to output points in the input camera's frame
- Eliminate the alignment step entirely
- Training data: partial ZED clouds paired with completed shapes from the current pipeline
