# Project FUSE - Brainstorm

## Core Concept

Real-time 2D/3D perception pipeline for humanoid robotics. Takes synchronized RGB and point cloud streams from a ZED stereo camera and produces unified scene understanding per object: **text label + 2D bounding box + 3D point cluster**.

---

## Pipeline Overview

```
ZED Camera --> 2D Branch (YOLO World)   --> label + bounding box
           --> 3D Branch (PointNet++)   --> point cloud segments
                       --> Fusion Layer --> unified per-object output
                       --> Completion   --> complete 3D geometry (optional)
```

---

## ZED Camera Outputs

The ZED stereo camera provides simultaneously:

| Output          | Format                              | Used For              |
|-----------------|-------------------------------------|-----------------------|
| Left RGB image  | 2D color                            | 2D detection (YOLO)   |
| Right RGB image | 2D color                            | Internal stereo math  |
| Depth map       | 2D, per-pixel distance              | Optional bridge       |
| Point cloud     | 3D, unordered set of (x, y, z)     | 3D segmentation       |

Depth map and point cloud are two representations of the same 3D information. Pipeline uses RGB for 2D detection and point cloud for 3D segmentation directly.

### Decisions
- **ZED model:** ZED Mini
- **Resolution:** 720p
- **SDK:** v5.2.1 (latest)
- **Built-in detection:** Bypass -- use custom models for full control over indoor object detection

---

## Branch 1: 2D Detection (Left RGB Stream)

```
Left RGB image (640x640x3)
       |
YOLO World / Grounding DINO
       |
Per detected object:
  - Class label: "cup"
  - 2D bounding box: (x_min, y_min, x_max, y_max)
  - Confidence score: 0.94
```

**Model choice:** YOLO World for open-vocabulary detection -- can query with arbitrary text prompts rather than a fixed category list. Critical for home robotics where objects are unpredictable.

**Speed:** 30+ FPS, keeps up with video stream in real time.

### Decisions
- **Model:** YOLO World -- open-vocab like Grounding DINO but fast enough for real-time (~30 FPS on 3070), lower VRAM than GDINO
- **Vocabulary:** Open-vocab from day one -- pass text prompts at inference, no retraining needed for new objects
- **Training:** Pretrained only for v1 -- generalizes well to household objects out of the box, fine-tune later if needed

---

## Branch 2: 3D Segmentation (Point Cloud)

```
Raw point cloud from ZED (N x 3 matrix)
       |
PointNet++ / Mask3D
       |
Per point:
  - Semantic label: "cup", "table", "background"
  - Instance ID: which specific cup
```

### Segmentation Levels

- **Semantic segmentation:** Labels every point with a category. All cup points get "cup" regardless of count.
- **Instance segmentation:** Assigns unique ID per object. Cup 1 = ID 001, Cup 2 = ID 002. Required for manipulation tasks.

**Model choice:** Start with PointNet++ for semantic segmentation. Upgrade to Mask3D (transformer-based, strongest for indoor 3D instance segmentation) once pipeline is working.

### Decisions
- **Segmentation level:** Semantic (PointNet++) for v1 -- lighter on VRAM (~1-2GB), simpler setup. YOLO World provides instance-level info via fusion. Upgrade to Mask3D in v2.
- **Training data:** Pretrained only (ScanNet-pretrained) -- already knows indoor categories. Custom data collection deferred.
- **Labeling:** Deferred -- not fine-tuning for v1. Future path: use own pipeline's 2D detections projected into 3D as auto-labels, then manually correct.

---

## Fusion Layer: Linking 2D and 3D

### Method 1: Project 3D Points into 2D Image Space

```
For each 3D point (x, y, z):
  Apply camera projection matrix (from ZED calibration)
  --> pixel location (u, v) in left image

Check if (u, v) falls inside YOLO bounding box for "cup"
  --> if yes, this 3D point belongs to the cup
```

- Fast, deterministic, no additional model needed
- Uses 2D bounding box as a 3D region-of-interest filter
- ZED SDK provides projection utilities out of the box

### Method 2: Independent Matching by Instance ID

- Run 3D instance segmentation to get point clusters with unique IDs
- Run 2D detection to get labeled bounding boxes
- Match by projecting each 3D cluster centroid into 2D and checking bounding box containment
- More robust when objects are partially occluded in one modality but visible in the other

### Decisions
- **Method:** Method 1 (projection) first -- simpler, deterministic, no instance segmentation required. Method 2 becomes relevant with Mask3D in v2.
- **Conflict resolution:** 2D (YOLO World) wins -- more mature and reliable than PointNet++ pretrained. Rule: `final_label = yolo_label if confidence > threshold else pointnet_label`.
- **Unmatched detections:** Keep from both sides, tag with `source` field: `"fused"`, `"2d_only"`, or `"3d_only"`. Lets downstream decide trust level.
- **Coordinate frame:** Camera frame for v1 -- ZED default, no extra transforms. Robot base/world frame requires extrinsic calibration (v2).

---

## Full Pipeline End to End

```
ZED Camera frame
       |
       |--- Left RGB --> YOLO / Grounding DINO
       |                       |
       |              "cup" at pixel (320, 240)
       |              bounding box (280,200,360,280)
       |                       |
       +--- Point Cloud --> Mask3D / PointNet++
                               |
                    Cluster A: 1,240 points (table)
                    Cluster B: 387 points   (cup)
                    Cluster C: 2,891 points (background)
                               |
                    +----------+----------+
                    |    Fusion Layer      |
                    |    (projection)      |
                    +----------+----------+
                               |
                    Output:
                    label         = "cup"
                    2D box        = (280, 200, 360, 280)
                    3D points     = Cluster B (387 points)
                    3D centroid   = (0.42, 0.85, 1.23) meters
                    3D bounding   = 8cm x 8cm x 12cm
```

---

## Model Responsibilities

| Model              | Input        | Output             | Speed                |
|--------------------|--------------|--------------------|----------------------|
| YOLO / GDino       | Left RGB     | Text + 2D box      | Real-time (30+ FPS)  |
| Mask3D / PointNet++| Point cloud  | 3D point clusters   | Near real-time       |
| Fusion layer       | Both above   | Linked 2D+3D       | Milliseconds         |
| PCN (optional)     | Partial cluster | Completed cluster | Real-time on GPU     |

---

## Shape Completion (Optional Stage)

Once you have the 3D point cluster for an object, it is likely incomplete -- occluded faces, bottom on table, etc.

```
387 partial cup points
       |
PCN / AdaPoinTr
       |
16,384 complete cup points
(back face, bottom, handle if present)
       |
Grasp point estimation
```

This completed representation gives a robot arm the spatial understanding needed for reliable grasping -- core value proposition of the system.

### Decisions
- **Scope:** Deferred to v2 -- adds VRAM pressure on 3070, upstream pipeline must work reliably first.
- **Minimum viable output for v1:** Label + 2D box + 3D point cluster + centroid + 3D bounding box. Full completed geometry deferred.

**V1 output spec per object:**
```
label:      "cup"
confidence: 0.94
source:     "fused" | "2d_only" | "3d_only"
box_2d:     (x_min, y_min, x_max, y_max)
points_3d:  N x 3 array (partial cluster)
centroid:   (x, y, z) meters
color:      (R, G, B) for point cloud visualization
```

Each object label gets a unique color. All points belonging to that object are rendered in that color in the 3D viewer (e.g., mug=red, phone=yellow, cup=green).

---

## Practical Starting Stack

| Component   | Model/Tool                          | Rationale                                    |
|-------------|-------------------------------------|----------------------------------------------|
| 2D branch   | YOLO World                          | Open vocabulary, no predefined categories    |
| 3D branch   | PointNet++ -> Mask3D                | Start simple, upgrade to instance seg later  |
| Fusion      | ZED SDK projection utilities        | Calibration and sync already handled         |
| Completion  | AdaPoinTr                           | Shape completion on isolated object clusters |

**Key advantage of ZED SDK:** Provides synchronized RGB + point cloud frames with timestamps and calibration already handled. Main engineering work is model integration and the fusion layer.

---

## Decisions: System-Wide

### Hardware & Deployment
- **Dev GPU:** NVIDIA RTX 3070 (8GB VRAM) on Razer Blade 15
- **Deployment:** Same laptop for v1 -- no embedded target yet

### Performance
- **Target FPS:** 15 FPS fused output (bottleneck is 3D branch)
- **Latency budget:** ~100ms end-to-end (~30ms YOLO, ~50ms PointNet++, ~20ms fusion + overhead)
- **Branch execution:** Async (parallel) -- run 2D and 3D in separate threads, fuse when both complete. Python `threading` / `concurrent.futures` works since inference releases the GIL.

### Software Architecture
- **Language:** Python only for v1 -- fastest to prototype, Ultralytics and PyTorch are Python-native
- **Framework:** Standalone, no ROS2 -- single Python process, avoids node/topic complexity. Add ROS2 in v2 if needed.
- **Inference:** PyTorch -- both models run natively. TensorRT optimization deferred to v2.
- **Inter-module:** Function calls -- pass numpy arrays between functions in a single process. No message passing needed.

### ZED SDK
- **Version:** v5.2.1 (confirmed installed)
- **Detection:** Bypass built-in, use custom YOLO World
- **Frame strategy:** Grab latest, drop stale -- always process most recent frame, no queue backlog

### Testing & Validation
- **Metrics:** IoU for 2D boxes, centroid error (cm) for 3D localization
- **Ground truth:** Manual measurement -- place known objects at measured positions, compare predicted centroids
- **Offline dev:** Record SVO files from ZED, replay during development without camera plugged in
- **Demo scenario:** Detect and locate 5 household objects on a table (mug, phone, cup, fork, bottle). Centroid error < 5cm = v1 success.

### Visualization
- **Tools:** OpenCV for 2D overlay (bounding boxes + labels on RGB), Open3D for 3D point cloud rendering
- **Mode:** Real-time during inference -- essential for debugging, show 2D and 3D side by side

### Downstream Integration
- **Consumer:** TBD -- pipeline is the product for v1, no grasp planner yet
- **Output format:** Python dataclass (`FusedObject`) for v1 -- clean interface, serialize later when a consumer exists
- **Update speed:** N/A until downstream is defined
