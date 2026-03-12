# Key Decisions

Strategic decisions and rationale behind major FUSE pipeline choices. Decisions are listed in roughly chronological order. See `docs/experiments.md` for full experimental evidence.

---

## Camera & SDK

| Decision | Choice | Why |
|----------|--------|-----|
| ZED model | ZED Mini | Available hardware |
| Resolution | 720p | Balances quality vs compute on 3070 |
| SDK version | v5.2.1 | Latest, best ZED Mini support |
| Depth mode | NEURAL | Neural stereo matching fills holes and sharpens edges vs ULTRA mode. Worth the ~1-2 GB VRAM cost for cleaner point clouds. |
| Built-in detection | Bypass | Custom models give full control for indoor objects |

## 2D Detection + Segmentation

| Decision | Choice | Why |
|----------|--------|-----|
| Model | YOLOE Seg (upgraded from YOLO World v2) | Open-vocab segmentation — provides both bounding boxes and pixel-level masks in a single pass. Masks enable direct 3D extraction from the organized point cloud. |
| Vocabulary | Open-vocab | Detect arbitrary objects via text prompts, no retraining |
| Training | Pretrained only | Generalizes well to household objects; fine-tune later if needed |

## 3D Extraction

| Decision | Choice | Why |
|----------|--------|-----|
| Method | YOLOE Seg masks → point cloud lookup | Pixel masks from YOLOE index directly into ZED's organized point cloud via array indexing (`xyz = point_cloud[mask]`). No projection math, no separate 3D model. ~2ms per frame. |
| Separate 3D model | Dropped | Evaluated PointNet++ and Mask3D — pretrained 3D models don't cover small household objects (mug, fork, phone). No open-vocab 3D seg model is real-time ready. YOLOE Seg handles detection + 3D extraction in a single pass. |
| Coordinate frame | Camera frame | ZED default, no extra transforms. Robot base / world frame deferred. |

## Point Cloud Denoising

| Decision | Choice | Why |
|----------|--------|-----|
| Strategy | Two-stage outlier removal | Single-stage filters miss either depth-bleed clusters (statistical misses them — they have many neighbors) or spatially scattered noise (MAD misses them — they're at the correct depth). Two stages cover both failure modes. |
| Stage 1 | Depth (Z) MAD filter, threshold 3× MAD | Targets dominant failure mode: depth bleeding at mask edges, where stereo matching assigns background depth to boundary pixels. MAD chosen over std because std gets inflated by the very outliers it's trying to remove. |
| Stage 2 | Open3D statistical outlier removal (k=20, std=1.5) | Catches spatially isolated noise that passed the depth filter — random stereo errors, edge artifacts from neural depth mode. |

## Shape Completion — PoinTr (Dropped)

| Decision | Choice | Why |
|----------|--------|-----|
| PoinTr integration | Removed from pipeline | Synthetic-to-real domain gap makes PoinTr ineffective on live ZED data. Five root causes: (1) orientation mismatch — ShapeNet canonical pose vs arbitrary camera frame, (2) partial view pattern — clean synthetic slices vs noisy front-facing scans, (3) point distribution — uniform mesh sampling vs variable-density stereo, (4) no category conditioning — model can't know it's looking at a mug, (5) scale/proportion mismatch vs idealized CAD models. |
| Shape completion strategy | Replaced by Hunyuan3D Full (image-conditioned) | Image-to-3D models use the RGB image to understand object semantics, bypassing the domain gap entirely. The image tells the model "this is a mug" — something a category-agnostic point cloud model cannot infer from noisy partial geometry. |

## 3D Generation Model Selection

| Decision | Choice | Why |
|----------|--------|-----|
| Model | Hunyuan3D Full | Only model of 6 tested that produces semantically correct shapes — hollow mug interior, through-hole handle, smooth cylindrical body. All other models failed on semantic quality. See detailed rejection reasons below. |
| Commitment | Go deep on Hunyuan3D Full architecture | Rather than continuing to evaluate more models, we're investing in understanding and fine-tuning Hunyuan3D Full's architecture. The three requirements for semantic understanding (direct image conditioning + DiT global attention + sufficient capacity) are well understood, and Hunyuan3D Full is the only available model meeting all three. Fine-tuning the decoder is more promising than hoping a new model appears. |

### Why Hunyuan3D Full — The Three Requirements

Semantic understanding in image-to-3D requires three architectural properties simultaneously. Remove any one and the model fails:

1. **Direct image-to-3D conditioning** — The image features must flow directly to the 3D generator with no intermediate bottleneck. InstantMesh routes through 6 synthesized 320x320 views, losing semantic detail at this bottleneck.

2. **Global-attention denoiser (DiT)** — The denoiser must reason about long-range spatial relationships (e.g., "the handle connects back to the body, forming a loop"). TripoSR pairs DINOv2 with a triplane NeRF decoder (local convolutions only) — it produces solid blobs because it can't capture global topology. TRELLIS uses xformers attention but on a sparse octree (SLAT), limiting resolution.

3. **Sufficient model capacity + diffusion steps** — Mini (0.6B params) and Turbo (distilled steps) have the same architecture as Full but lack capacity to resolve fine topology. They "know" it's a mug but produce fused handles — the through-hole is the most topologically demanding feature and requires full capacity to resolve.

### Rejected Models — Specific Failure Modes

| Model | Failure Mode | Root Cause |
|-------|-------------|------------|
| PoinTr | Blobby hallucinations, wrong geometry | No image input — only sees noisy 3D points with no visual/semantic signal. Synthetic-to-real domain gap. |
| TripoSR | Solid blobs, no hollow interior | Uses DINOv2 but triplane NeRF decoder lacks global attention. Can't reason about topology (holes, handles). |
| TRELLIS | Inconsistent — sometimes box shapes, sometimes rough mugs | Sparse octree (SLAT) limits resolution. DINOv2 + xformers attention is architecturally sound but sparse representation loses detail. Results vary wildly between runs. |
| InstantMesh | Wrong shapes entirely (not recognizable as mugs) | Intermediate bottleneck: routes through 6 synthesized 320x320 views. Semantic information about handle, interior, and object identity is lost at this bottleneck. |
| Hunyuan3D Mini | Cylindrical body correct, handle malformed (fused, no through-hole) | Same architecture as Full but 0.6B params — insufficient capacity to resolve handle topology. Fastest variant at 87s. |
| Hunyuan3D Turbo | Rough surfaces, heavy artifacts, handle fully closed | Distilled fewer diffusion steps — not enough refinement iterations. Paradoxically slower than Mini (109s vs 87s) due to longer volume decoding. Worst quality of all Hunyuan3D variants. |

## GPU Strategy

| Decision | Choice | Why |
|----------|--------|-----|
| Inference hardware | Cloud (Modal A100 80GB) over local GPU | RTX 3070 8GB too slow (151s vs 28s on A100) and barely fits the model. New hardware ($2,200-$4,100+) not justified during research phase. Modal serverless scales to zero — ~$0.02/inference, ~28,000 inferences to match cost of a new GPU. |
| Platform | Modal serverless | Auto-scales to zero when idle (no idle cost). A100 80GB eliminates VRAM constraints. Simple Python SDK — `modal deploy` for endpoints, `modal run` for testing. |
| Cold start | Accepted (~100-170s) | First invocation after idle requires full container spin-up + model download + CUDA init. Acceptable for research/async workflow. Keep-alive or pre-warming planned for production. |
| Buy hardware later? | Deferred until model is validated + fine-tuned | If Hunyuan3D Full (or fine-tuned variant) proves viable for production use and real-time latency is needed, invest in dedicated hardware (RTX 4090/5090 or eGPU) at that point. Cloud-first validates the approach cheaply. |

## Fine-Tuning Strategy

| Decision | Choice | Why |
|----------|--------|-----|
| Target | Replace VAE decoder with direct point cloud decoder | The VAE occupancy decoder + marching cubes takes ~17s (61% of total time on A100) to produce a mesh we immediately discard in favor of sampled points. The 3D latent already encodes the complete shape — we just need a cheaper extraction path. |
| Freeze encoder + diffusion | Yes — only train the decoder head | DINOv2 and DiT already produce excellent latents with correct semantic understanding. The problem is purely in the output conversion stage. Freezing keeps training tractable and preserves the learned representations. |
| Decoder architecture | Transformer point decoder (preferred) | Cross-attention from learnable point queries to latent features. Maintains global reasoning capability. Architecturally similar to PoinTr's decoder (proven for point cloud generation). MLP alternative is faster but may struggle with discontinuous surfaces (handles, holes). |
| Training data | Generate (latent, point cloud) pairs from existing pipeline | Run Hunyuan3D Full on ~1,000-5,000 Objaverse objects, saving latents before VAE decoder + mesh-sampled ground truth point clouds. ~39 GPU-hours (~$97 on Modal) for 5,000 objects. |
| Expected speedup | ~28s → ~10-14s total generation | Decoder target: 1-3s (vs current 17s VAE). DiT diffusion (~9s) unchanged. Eliminates mesh overhead entirely. |
| Stretch goal | Viewpoint-aware decoder | Train decoder to output points already in the input camera's frame, eliminating the FPFH + RANSAC + ICP alignment step (currently unreliable for full-vs-partial cloud registration). |

## System & Infrastructure

| Decision | Choice | Why |
|----------|--------|-----|
| Language | Python only | Fastest to prototype, native Ultralytics/PyTorch/Modal support |
| Framework | Standalone (no ROS2) | Avoids node/topic complexity for research phase |
| Inference | PyTorch | All models run natively; TensorRT deferred |
| Branch execution | Async (parallel threads) | Real-time local loop (ZED + YOLOE + denoising at ~16 FPS) runs independently from cloud 3D generation (~28s async) |
| Frame strategy | Grab latest, drop stale | Ensures visualization always shows current state, even if processing takes multiple frame intervals |
