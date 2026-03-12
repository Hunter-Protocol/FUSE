# Key Decisions

Rationale behind major v1 decisions. See `brainstorm.md` for full exploration.

## Camera & SDK
| Decision | Choice | Why |
|----------|--------|-----|
| ZED model | ZED Mini | Available hardware |
| Resolution | 720p | Balances quality vs compute on 3070 |
| SDK version | v5.2.1 | Latest, best ZED Mini support |
| Built-in detection | Bypass | Custom models give full control for indoor objects |

## 2D Detection
| Decision | Choice | Why |
|----------|--------|-----|
| Model | YOLO World | Open-vocab, real-time (~30 FPS), low VRAM vs Grounding DINO |
| Vocabulary | Open-vocab | Detect arbitrary objects via text prompts, no retraining |
| Training | Pretrained only | Generalizes well; fine-tune later if needed |

## 3D Extraction (changed from PointNet++)
| Decision | Choice | Why |
|----------|--------|-----|
| Method | YOLOE Seg masks → point cloud lookup | Pretrained 3D models (PointNet++, Mask3D) don't cover small household objects (mug, fork, phone). YOLOE Seg pixel masks projected into 3D give clean per-object clusters with one model. |
| Separate 3D model | Dropped for v1 | No open-vocab 3D seg model is real-time ready. YOLOE Seg handles detection + 3D extraction in a single pass. |
| Coordinate frame | Camera frame | ZED default, no extra transforms |

## System
| Decision | Choice | Why |
|----------|--------|-----|
| Language | Python only | Fastest to prototype, native Ultralytics/PyTorch |
| Framework | Standalone (no ROS2) | Avoids node/topic complexity for v1 |
| Inference | PyTorch | Both models run natively; TensorRT in v2 |
| Branch execution | Async (parallel threads) | Cuts latency vs sequential |

## Cloud 3D Generation
| Decision | Choice | Why |
|----------|--------|-----|
| GPU strategy | Cloud (Modal A100) over local GPU | RTX 3070 too slow for Hunyuan3D Full (151s vs 28s on A100). New hardware not justified for research phase. Modal serverless scales to zero when idle (~$0.02/inference). |
| 3D generation model | Hunyuan3D Full | Only model of 6 tested that produces semantically correct shapes. Requires direct image conditioning + global-attention denoiser (DiT) + sufficient capacity/steps. See `docs/experiments.md`. |
| Rejected models | TripoSR, TRELLIS, InstantMesh, Hunyuan3D Mini/Turbo | TripoSR: solid blobs. TRELLIS: inconsistent quality. InstantMesh: wrong shapes (intermediate bottleneck). Mini/Turbo: insufficient capacity (fused handles). |
