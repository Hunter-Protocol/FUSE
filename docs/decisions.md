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
| Method | YOLO World Seg masks → point cloud lookup | Pretrained 3D models (PointNet++, Mask3D) don't cover small household objects (mug, fork, phone). YOLO Seg pixel masks projected into 3D give clean per-object clusters with one model. |
| Separate 3D model | Dropped for v1 | No open-vocab 3D seg model is real-time ready. YOLO Seg handles detection + 3D extraction in a single pass. |
| Coordinate frame | Camera frame | ZED default, no extra transforms |

## System
| Decision | Choice | Why |
|----------|--------|-----|
| Language | Python only | Fastest to prototype, native Ultralytics/PyTorch |
| Framework | Standalone (no ROS2) | Avoids node/topic complexity for v1 |
| Inference | PyTorch | Both models run natively; TensorRT in v2 |
| Branch execution | Async (parallel threads) | Cuts latency vs sequential |
| Shape completion | Deferred to v2 | VRAM pressure, upstream must work first |
