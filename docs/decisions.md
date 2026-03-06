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

## 3D Segmentation
| Decision | Choice | Why |
|----------|--------|-----|
| Model | PointNet++ (semantic) | Lighter VRAM (~1-2GB), simpler. Mask3D in v2 |
| Training data | ScanNet-pretrained | Already knows indoor categories |
| Labeling | Deferred | Not fine-tuning for v1 |

## Fusion
| Decision | Choice | Why |
|----------|--------|-----|
| Method | Projection (3D->2D) | Simpler, deterministic, no instance seg needed |
| Conflicts | 2D wins | YOLO more reliable than pretrained PointNet++ |
| Unmatched | Keep both sides | Tag with source field, let downstream decide |
| Coordinate frame | Camera frame | ZED default, no extra transforms |

## System
| Decision | Choice | Why |
|----------|--------|-----|
| Language | Python only | Fastest to prototype, native Ultralytics/PyTorch |
| Framework | Standalone (no ROS2) | Avoids node/topic complexity for v1 |
| Inference | PyTorch | Both models run natively; TensorRT in v2 |
| Branch execution | Async (parallel threads) | Cuts latency vs sequential |
| Shape completion | Deferred to v2 | VRAM pressure, upstream must work first |
