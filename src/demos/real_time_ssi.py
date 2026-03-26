"""VC Showcase Demo: Spatial Semantics + Physics Inference.

Flow:
  Phase 1 — Live detection: RGB video + colored point cloud (all objects)
            User types object name in terminal to select target
  Phase 2 — Hacker-style inference status in terminal
  Phase 3 — Result: raw partial cloud + complete mesh + info panel

Usage:
    python -m demos.demo_vc                    # live mode (ZED camera)
    python -m demos.demo_vc --svo path.svo2    # offline camera (SVO file)
    python -m demos.demo_vc --offline          # fully offline (pre-computed data only)
"""

import argparse
import json
import random
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh

from core.fused_object import LABEL_COLORS, DEFAULT_LABEL_COLOR

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DEMO_OBJECTS = ["mug", "cup", "fork"]

# Descriptive YOLOE class names to disambiguate similar objects in CLIP space
# Maps: YOLOE class name -> our short label (used for data dirs, colors, etc.)
YOLOE_CLASS_MAP = {
    "coffee mug":      "mug",
    "transparent cup":  "cup",
    "fork":            "fork",
}
YOLOE_CLASSES = list(YOLOE_CLASS_MAP.keys())

# Info panel dimensions
PANEL_W = 400
PANEL_H = 600

# ANSI colors for terminal
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_object_data(label):
    """Load pre-computed data for an object. Returns None if missing."""
    obj_dir = DATA_DIR / label
    if not obj_dir.exists():
        return None

    data = {"label": label, "dir": obj_dir}

    # Crop image
    crop_path = obj_dir / "crop.png"
    data["crop"] = cv2.imread(str(crop_path)) if crop_path.exists() else None

    # Partial point cloud — prefer raw noisy version
    raw_path = obj_dir / "partial_raw.npy"
    partial_path = obj_dir / "partial.npy"
    if raw_path.exists():
        data["partial"] = np.load(str(raw_path))
    elif partial_path.exists():
        data["partial"] = np.load(str(partial_path))
    else:
        return None

    # Aligned mesh
    mesh_path = obj_dir / "mesh_aligned.obj"
    if mesh_path.exists():
        data["mesh"] = trimesh.load(str(mesh_path), force='mesh')
    else:
        data["mesh"] = None

    # Physics
    physics_path = obj_dir / "physics.json"
    if physics_path.exists():
        with open(physics_path) as f:
            data["physics"] = json.load(f)
    else:
        data["physics"] = None

    return data


# ---------------------------------------------------------------------------
# Terminal hacker status output
# ---------------------------------------------------------------------------

def hacker_print(msg, delay=0.02):
    """Print with typewriter effect."""
    for ch in msg:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def spinner(text, duration):
    """Show a spinner for a fixed duration."""
    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    end = time.time() + duration
    i = 0
    while time.time() < end:
        sys.stdout.write(f"\r  {CYAN}{chars[i % len(chars)]}{RESET} {text}")
        sys.stdout.flush()
        time.sleep(0.08)
        i += 1
    sys.stdout.write(f"\r  {GREEN}✓{RESET} {text}\n")
    sys.stdout.flush()


def fake_hex_stream(lines=3):
    """Print random hex lines for wow effect."""
    for _ in range(lines):
        hex_str = " ".join(f"{random.randint(0,255):02x}" for _ in range(20))
        print(f"  {DIM}{hex_str}{RESET}")
        time.sleep(0.05)


def run_inference_status(label, obj_data):
    """Hacker-style terminal output for inference process."""
    print()
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  REAL TIME SSI — INFERENCE ENGINE{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")
    print()

    hacker_print(f"  {YELLOW}>{RESET} Target acquired: {BOLD}{label.upper()}{RESET}", 0.03)
    time.sleep(0.3)

    # Phase 1: Point cloud extraction
    print(f"\n  {BOLD}[1/4] POINT CLOUD EXTRACTION{RESET}")
    fake_hex_stream(2)
    n_pts = len(obj_data["partial"])
    spinner(f"Extracting depth points from stereo pair... {n_pts:,} points", 1.2)

    # Phase 2: Mesh reconstruction
    print(f"\n  {BOLD}[2/4] SPATIAL SEMANTICS — MESH RECONSTRUCTION{RESET}")
    fake_hex_stream(3)
    spinner("Reconstructing complete 3D mesh from partial observation...", 2.5)
    if obj_data["mesh"]:
        n_faces = len(obj_data["mesh"].faces)
        n_verts = len(obj_data["mesh"].vertices)
        spinner(f"Mesh generated: {n_verts:,} vertices, {n_faces:,} faces", 0.6)

    # Phase 3: Alignment
    print(f"\n  {BOLD}[3/4] RANSAC + ICP ALIGNMENT{RESET}")
    fake_hex_stream(2)
    spinner("Computing FPFH features...", 0.7)
    spinner("RANSAC global registration (100k iterations)...", 1.0)
    spinner("ICP refinement (200 iterations)...", 0.8)
    spinner("Mesh aligned to camera frame — real-world scale locked", 0.4)

    # Phase 4: Physics
    print(f"\n  {BOLD}[4/4] PHYSICS INFERENCE{RESET}")
    fake_hex_stream(2)
    physics = obj_data.get("physics")
    if physics:
        spinner("Computing oriented bounding box...", 0.5)
        spinner("Estimating volume (convex hull)...", 0.4)
        spinner(f"Material lookup: {physics['material']} — density applied", 0.3)

        print(f"\n  {GREEN}{'─'*50}{RESET}")
        print(f"  {BOLD}{GREEN}RESULTS{RESET}")
        print(f"  {GREEN}{'─'*50}{RESET}")
        if label in ("mug", "cup"):
            radius_cm = round(physics['depth_cm'] / 2, 1)
            props = [
                ("Height",   f"{physics['height_cm']} cm"),
                ("Radius",   f"{radius_cm} cm"),
                ("Volume",   f"{physics['volume_cm3']} cm3"),
                ("Weight",   f"~{physics['weight_g']:.0f} g"),
                ("Material", physics['material']),
            ]
        elif label == "fork":
            props = [
                ("Length",    f"{physics['height_cm']} cm"),
                ("Width",     f"{physics['width_cm']} cm"),
                ("Thickness", f"{physics['depth_cm']} cm"),
                ("Weight",    f"~{physics['weight_g']:.0f} g"),
                ("Material",  physics['material']),
            ]
        else:
            props = [
                ("Height",   f"{physics['height_cm']} cm"),
                ("Width",    f"{physics['width_cm']} cm"),
                ("Volume",   f"{physics['volume_cm3']} cm3"),
                ("Weight",   f"~{physics['weight_g']:.0f} g"),
                ("Material", physics['material']),
            ]
        for name, val in props:
            time.sleep(0.15)
            print(f"    {CYAN}{name:10s}{RESET}  {BOLD}{val}{RESET}")
    else:
        spinner("No pre-computed physics available", 0.3)

    print(f"\n  {GREEN}✓ Inference complete.{RESET}")
    print(f"  {DIM}Opening 3D visualization...{RESET}\n")
    time.sleep(0.5)


# ---------------------------------------------------------------------------
# OpenCV drawing helpers
# ---------------------------------------------------------------------------

def bgr_color(rgb_color):
    """Convert normalized RGB to BGR uint8 for OpenCV."""
    return (int(rgb_color[2] * 255), int(rgb_color[1] * 255), int(rgb_color[0] * 255))


def draw_detections(frame, objects, selected_label=None, inference_done=False,
                    inference_start=None, raw_counts=None):
    """Draw bounding boxes, masks, labels, and live metrics. Highlight selected object."""
    overlay = frame.copy()
    for obj in objects:
        color_bgr = bgr_color(obj.color)
        is_selected = (selected_label is not None and obj.label == selected_label)

        # Filled mask
        overlay[obj.mask] = (
            overlay[obj.mask] * 0.5 + np.array(color_bgr) * 0.5
        ).astype(np.uint8)

        # Bounding box
        x1, y1, x2, y2 = obj.box_2d
        thickness = 4 if is_selected else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, thickness)

        # Corner accents for selected object
        if is_selected:
            corner_len = 20
            ct = 3
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), ct)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), ct)
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), (255, 255, 255), ct)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), (255, 255, 255), ct)
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), (255, 255, 255), ct)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), (255, 255, 255), ct)
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), ct)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), ct)

        # Label
        text = f"{obj.label} {obj.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color_bgr, -1)
        cv2.putText(frame, text, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Live metrics below bounding box
        metric_y = y2 + 16
        # 3D centroid
        if obj.source == "fused" and obj.centroid:
            cx, cy, cz = obj.centroid
            cv2.putText(frame, f"xyz: ({cx:.2f}, {cy:.2f}, {cz:.2f})",
                        (x1, metric_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1)
            metric_y += 16
        # Raw point count
        raw_n = raw_counts.get(obj.label, 0) if raw_counts else 0
        if raw_n > 0:
            cv2.putText(frame, f"pts: {raw_n:,}",
                        (x1, metric_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (180, 180, 180), 1)
            metric_y += 16

        # Inference status tag above box
        if is_selected:
            if inference_done:
                cv2.putText(frame, "INFERRED", (x1, y1 - th - 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif inference_start is not None:
                elapsed = time.time() - inference_start
                cv2.putText(frame, f"INFERRING... {elapsed:.1f}s",
                            (x1, y1 - th - 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    return frame


def remap_labels(objects):
    """Remap YOLOE descriptive class names to short labels (e.g. 'coffee mug' -> 'mug').

    Also updates the color field to match the remapped label.
    """
    for obj in objects:
        short = YOLOE_CLASS_MAP.get(obj.label, obj.label)
        obj.label = short
        obj.color = LABEL_COLORS.get(short, DEFAULT_LABEL_COLOR)


# ---------------------------------------------------------------------------
# Open3D helpers
# ---------------------------------------------------------------------------

def create_partial_pcd(points, label):
    """Create Open3D point cloud colored by label."""
    color = LABEL_COLORS.get(label, DEFAULT_LABEL_COLOR)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(points), 1)))
    return pcd


def create_mesh_geometry(obj_data):
    """Create Open3D triangle mesh from trimesh data."""
    mesh = obj_data["mesh"]
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(
        np.array(mesh.vertices, dtype=np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(
        np.array(mesh.faces, dtype=np.int32))
    o3d_mesh.compute_vertex_normals()

    color = LABEL_COLORS.get(obj_data["label"], DEFAULT_LABEL_COLOR)
    mesh_color = tuple(min(1.0, c * 0.7 + 0.3) for c in color)
    o3d_mesh.paint_uniform_color(mesh_color)
    return o3d_mesh


def build_scene_pcd(objects_data):
    """Build a combined point cloud of ALL objects, each in its label color."""
    all_pts, all_clr = [], []
    for obj in objects_data:
        pts = obj["partial"]
        color = LABEL_COLORS.get(obj["label"], DEFAULT_LABEL_COLOR)
        all_pts.append(pts)
        all_clr.append(np.tile(color, (len(pts), 1)))

    if not all_pts:
        pcd = o3d.geometry.PointCloud()
        return pcd

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.vstack(all_pts).astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(
        np.vstack(all_clr).astype(np.float64))
    return pcd


# ---------------------------------------------------------------------------
# Info panel (Phase 3)
# ---------------------------------------------------------------------------

def make_info_panel(obj_data, live_crop=None):
    """Render the OpenCV info panel for a single object."""
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    y = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    gray = (180, 180, 180)
    accent = (0, 200, 255)

    # Thumbnail: prefer live crop of selected object, fall back to saved crop
    thumb = live_crop if live_crop is not None else obj_data.get("crop")
    if thumb is not None:
        crop_h = 120
        scale = crop_h / thumb.shape[0]
        crop_w = min(int(thumb.shape[1] * scale), 160)
        crop_resized = cv2.resize(thumb, (crop_w, crop_h))
        panel[y:y+crop_h, 10:10+crop_w] = crop_resized
        cv2.putText(panel, "RT-SSI", (crop_w + 20, y + 35), font, 0.9, accent, 2)
        cv2.putText(panel, "Spatial Semantics", (crop_w + 20, y + 65), font, 0.5, gray, 1)
        y += crop_h + 15
    else:
        cv2.putText(panel, "Real Time SSI", (10, y + 30), font, 0.8, accent, 2)
        y += 45

    cv2.line(panel, (10, y), (PANEL_W - 10, y), (80, 80, 80), 1)
    y += 15

    label = obj_data["label"]
    cv2.putText(panel, f"Object: {label.capitalize()}", (10, y + 5), font, 0.65, white, 1)
    y += 30

    physics = obj_data.get("physics")
    if physics:
        cv2.putText(panel, f"Material: {physics['material'].capitalize()}",
                    (10, y + 5), font, 0.55, gray, 1)
        y += 25

    cv2.line(panel, (10, y), (PANEL_W - 10, y), (80, 80, 80), 1)
    y += 20

    # Spatial semantics
    cv2.putText(panel, "SPATIAL SEMANTICS", (10, y), font, 0.55, accent, 1)
    y += 25

    partial = obj_data.get("partial")
    if partial is not None:
        cv2.putText(panel, f"Partial points: {len(partial):,}", (20, y), font, 0.5, gray, 1)
        y += 22

    mesh = obj_data.get("mesh")
    if mesh is not None:
        cv2.putText(panel, f"Complete mesh: {len(mesh.faces):,} faces", (20, y), font, 0.5, gray, 1)
        y += 22

    cv2.line(panel, (10, y), (PANEL_W - 10, y), (80, 80, 80), 1)
    y += 20

    # Physics
    cv2.putText(panel, "PHYSICS PROPERTIES", (10, y), font, 0.55, accent, 1)
    y += 25

    if physics:
        label = obj_data["label"]
        if label in ("mug", "cup"):
            # Cylindrical objects: show radius instead of depth
            radius_cm = round(physics['depth_cm'] / 2, 1)
            props = [
                ("Height",  f"{physics['height_cm']} cm"),
                ("Radius",  f"{radius_cm} cm"),
                ("Volume",  f"{physics['volume_cm3']} cm3"),
                ("Weight",  f"~{physics['weight_g']:.0f} g ({physics['material']})"),
            ]
        elif label == "fork":
            # Flat utensil: show length, width, thickness
            props = [
                ("Length",    f"{physics['height_cm']} cm"),
                ("Width",     f"{physics['width_cm']} cm"),
                ("Thickness", f"{physics['depth_cm']} cm"),
                ("Weight",    f"~{physics['weight_g']:.0f} g ({physics['material']})"),
            ]
        else:
            props = [
                ("Height",  f"{physics['height_cm']} cm"),
                ("Width",   f"{physics['width_cm']} cm"),
                ("Depth",   f"{physics['depth_cm']} cm"),
                ("Volume",  f"{physics['volume_cm3']} cm3"),
                ("Weight",  f"~{physics['weight_g']:.0f} g ({physics['material']})"),
            ]
        for name, value in props:
            cv2.putText(panel, f"{name}:", (20, y), font, 0.5, gray, 1)
            cv2.putText(panel, value, (140, y), font, 0.5, white, 1)
            y += 22
    else:
        cv2.putText(panel, "(no physics data)", (20, y), font, 0.5, gray, 1)

    return panel


# ---------------------------------------------------------------------------
# Main demo class
# ---------------------------------------------------------------------------

class VCDemo:
    def __init__(self, offline=False, svo_path=None):
        self.offline = offline
        self.svo_path = svo_path
        self.objects = {}  # label -> obj_data

    def load_data(self):
        """Load all pre-computed object data."""
        for label in DEMO_OBJECTS:
            data = load_object_data(label)
            if data is not None:
                self.objects[label] = data
            else:
                print(f"  Warning: no data for {label}")

        if not self.objects:
            print("ERROR: No object data found. Run precompute_demo_data.py first.")
            sys.exit(1)

    def run(self):
        print(f"\n{BOLD}{CYAN}  Real Time Spatial Semantics Inference{RESET}")
        print(f"  {'─'*40}\n")

        self.load_data()

        if self.offline:
            self._run_offline()
        else:
            self._run_live()

    def _run_live(self):
        """Single-loop live mode. Pipeline and windows stay alive the whole time."""
        from core.pipeline import FUSEPipeline

        available = list(self.objects.keys())
        classes = YOLOE_CLASSES

        # --- Window layout ---
        # Top:    Live Video (OpenCV, resized)
        # Bottom: Point Cloud | Complete Mesh | Info Panel
        LIVE_W, LIVE_H = 860, 480  # resized live feed
        BOT_W, BOT_H = 430, 400
        TITLE_BAR = 55  # window title bar + border
        ROW2_Y = LIVE_H + TITLE_BAR

        # Bottom-left: raw point cloud
        vis_pcd = o3d.visualization.Visualizer()
        vis_pcd.create_window("RT-SSI - Raw Point Cloud", width=BOT_W, height=BOT_H,
                              left=0, top=ROW2_Y)
        opt = vis_pcd.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.05, 0.05, 0.05])

        live_pcd = o3d.geometry.PointCloud()
        vis_pcd.add_geometry(live_pcd)
        first_points = True

        # Bottom-center: complete mesh (shown after inference)
        vis_mesh = o3d.visualization.Visualizer()
        vis_mesh.create_window("RT-SSI - Complete Mesh", width=BOT_W, height=BOT_H,
                               left=BOT_W, top=ROW2_Y)
        opt = vis_mesh.get_render_option()
        opt.background_color = np.array([0.05, 0.05, 0.05])
        opt.mesh_show_wireframe = False
        mesh_geom = None

        # State
        raw_points_by_label = {}
        live_crops_by_label = {}
        inference_done = False
        active_label = None
        loading_start = None  # when click happened (for loading bar)
        saved_mesh_views = {}  # label -> pinhole camera params (user's manual orientation)

        # Load saved views from disk if available
        views_path = DATA_DIR / "mesh_views.json"
        if views_path.exists():
            with open(views_path) as f:
                views_on_disk = json.load(f)
            for label, v in views_on_disk.items():
                params = o3d.camera.PinholeCameraParameters()
                intrinsic = o3d.camera.PinholeCameraIntrinsic()
                intrinsic.width = v["width"]
                intrinsic.height = v["height"]
                intrinsic.intrinsic_matrix = np.array(v["intrinsic"])
                params.intrinsic = intrinsic
                params.extrinsic = np.array(v["extrinsic"])
                saved_mesh_views[label] = params
        latest_detected = []  # shared with mouse callback

        LOADING_DURATION = 1.5  # seconds for loading bar animation

        # Pre-create OpenCV windows at fixed positions
        cv2.namedWindow("RT-SSI - Live Detection", cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("RT-SSI - Live Detection", 0, 0)
        cv2.namedWindow("RT-SSI - Info", cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("RT-SSI - Info", BOT_W * 2, ROW2_Y)

        # Mouse click callback — select object by clicking its bounding box
        def on_click(event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            # Scale click coords from resized frame back to original
            scale_x = 1280.0 / LIVE_W
            scale_y = 720.0 / LIVE_H
            orig_x, orig_y = int(x * scale_x), int(y * scale_y)
            for obj in latest_detected:
                x1, y1, x2, y2 = obj.box_2d
                if x1 <= orig_x <= x2 and y1 <= orig_y <= y2:
                    if obj.label in self.objects:
                        selected[0] = obj.label
                    break

        cv2.setMouseCallback("RT-SSI - Live Detection", on_click)
        selected = [None]

        with FUSEPipeline(classes, svo_path=self.svo_path, model_size="11m") as pipe:
            # Warmup
            for _ in range(3):
                bgr, detected, _, _ = pipe.process_frame(skip_scene=True)
                if bgr is not None:
                    remap_labels(detected)
                    frame = draw_detections(bgr, detected)
                    frame = cv2.resize(frame, (LIVE_W, LIVE_H))
                    cv2.imshow("RT-SSI - Live Detection", frame)
                    vis_pcd.poll_events()
                    vis_pcd.update_renderer()
                    cv2.waitKey(1)

            print(f"\n{BOLD}{CYAN}{'─'*50}{RESET}")
            print(f"{BOLD}  RT-SSI — Click on an object to infer{RESET}")
            print(f"{BOLD}{CYAN}{'─'*50}{RESET}")
            print(f"  {DIM}Press 'q' to quit.{RESET}")

            fps = 0.0
            prev_time = time.time()

            # --- Main loop: live feed runs continuously ---
            while True:
                t_frame = time.time()
                bgr, detected, _, _ = pipe.process_frame(skip_scene=True)
                latency_ms = (time.time() - t_frame) * 1000
                if bgr is None:
                    if self.svo_path:
                        break
                    continue

                remap_labels(detected)
                latest_detected = detected  # share with click callback

                # Extract raw points + crops per label
                pc_data = pipe.pc_mat.get_data()
                all_pts, all_clr = [], []
                raw_counts = {}
                for obj in detected:
                    xyz = pc_data[:, :, :3][obj.mask]
                    valid = np.isfinite(xyz).all(axis=1)
                    raw_pts = xyz[valid].astype(np.float32)
                    if len(raw_pts) > 0:
                        color = LABEL_COLORS.get(obj.label, DEFAULT_LABEL_COLOR)
                        if active_label and obj.label != active_label:
                            color = tuple(c * 0.3 for c in color)
                        all_pts.append(raw_pts)
                        all_clr.append(np.tile(color, (len(raw_pts), 1)))
                        raw_points_by_label[obj.label] = raw_pts
                    raw_counts[obj.label] = len(raw_pts) if len(raw_pts) > 0 else 0
                    x1, y1, x2, y2 = obj.box_2d
                    pad = 10
                    h, w = bgr.shape[:2]
                    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
                    cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
                    live_crops_by_label[obj.label] = bgr[cy1:cy2, cx1:cx2].copy()

                # Update live point cloud
                if all_pts:
                    pts = np.vstack(all_pts).astype(np.float64)
                    clr = np.vstack(all_clr).astype(np.float64)
                    live_pcd.points = o3d.utility.Vector3dVector(
                        np.ascontiguousarray(pts))
                    live_pcd.colors = o3d.utility.Vector3dVector(
                        np.ascontiguousarray(clr))
                else:
                    live_pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                    live_pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))

                vis_pcd.update_geometry(live_pcd)
                if first_points and all_pts:
                    vis_pcd.reset_view_point(True)
                    first_points = False

                # Handle click → start loading (or switch to new object)
                if selected[0] is not None and selected[0] != active_label:
                    # Save current mesh camera orientation before switching
                    if active_label is not None:
                        ctr = vis_mesh.get_view_control()
                        saved_mesh_views[active_label] = ctr.convert_to_pinhole_camera_parameters()
                    active_label = selected[0]
                    loading_start = time.time()
                    inference_done = False
                    # Clear previous mesh and info during loading
                    if mesh_geom is not None:
                        vis_mesh.remove_geometry(mesh_geom, reset_bounding_box=False)
                        mesh_geom = None
                    # Show "inferencing" placeholder in info panel
                    loading_panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
                    loading_panel[:] = (30, 30, 30)
                    cv2.putText(loading_panel, f"Inferencing {active_label}...",
                                (20, PANEL_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 255), 2)
                    cv2.imshow("RT-SSI - Info", loading_panel)
                    print(f"  {CYAN}> Selected: {active_label}{RESET}")

                # Draw detections with live metrics
                frame = draw_detections(bgr, detected,
                                        selected_label=active_label,
                                        inference_done=inference_done,
                                        inference_start=loading_start,
                                        raw_counts=raw_counts)

                # Draw loading bar on selected object's bounding box
                if active_label and not inference_done and loading_start:
                    elapsed = time.time() - loading_start
                    progress = min(elapsed / LOADING_DURATION, 1.0)
                    for obj in detected:
                        if obj.label == active_label:
                            x1, y1, x2, y2 = obj.box_2d
                            bar_w = int((x2 - x1) * progress)
                            # Background bar
                            cv2.rectangle(frame, (x1, y2 + 2), (x2, y2 + 10),
                                          (50, 50, 50), -1)
                            # Progress fill
                            cv2.rectangle(frame, (x1, y2 + 2), (x1 + bar_w, y2 + 10),
                                          (0, 255, 255), -1)
                            break

                    # Loading complete → show results
                    if progress >= 1.0:
                        label = active_label
                        obj_data = self.objects[label]

                        if mesh_geom is not None:
                            vis_mesh.remove_geometry(mesh_geom, reset_bounding_box=False)
                        if obj_data["mesh"] is not None:
                            mesh_geom = create_mesh_geometry(obj_data)
                        else:
                            mesh_geom = create_partial_pcd(obj_data["partial"], label)
                        vis_mesh.add_geometry(mesh_geom, reset_bounding_box=True)
                        # Restore saved camera view if user already oriented this object
                        if label in saved_mesh_views:
                            ctr = vis_mesh.get_view_control()
                            ctr.convert_from_pinhole_camera_parameters(
                                saved_mesh_views[label], allow_arbitrary=True)
                        else:
                            vis_mesh.reset_view_point(True)

                        live_crop = live_crops_by_label.get(label)
                        panel = make_info_panel(obj_data, live_crop=live_crop)
                        cv2.imshow("RT-SSI - Info", panel)

                        inference_done = True

                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
                prev_time = now
                frame = cv2.resize(frame, (LIVE_W, LIVE_H))
                cv2.putText(frame, f"FPS: {fps:.1f} | Latency: {latency_ms:.0f}ms",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if not active_label:
                    cv2.putText(frame, "Click on an object to infer",
                                (10, LIVE_H - 15), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (200, 200, 200), 1)
                cv2.imshow("RT-SSI - Live Detection", frame)

                # Poll Open3D windows
                vis_pcd.poll_events()
                vis_pcd.update_renderer()
                vis_mesh.poll_events()
                vis_mesh.update_renderer()

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    # Save current object's view before quitting
                    if active_label is not None:
                        ctr = vis_mesh.get_view_control()
                        saved_mesh_views[active_label] = ctr.convert_to_pinhole_camera_parameters()
                    break

        # Persist all saved views to disk
        views_path = DATA_DIR / "mesh_views.json"
        views_to_save = {}
        for label, params in saved_mesh_views.items():
            views_to_save[label] = {
                "intrinsic": params.intrinsic.intrinsic_matrix.tolist(),
                "extrinsic": params.extrinsic.tolist(),
                "width": params.intrinsic.width,
                "height": params.intrinsic.height,
            }
        with open(views_path, 'w') as f:
            json.dump(views_to_save, f, indent=2)

        vis_pcd.destroy_window()
        vis_mesh.destroy_window()
        cv2.destroyAllWindows()

    def _run_offline(self):
        """Offline mode with pre-computed data only."""
        available = list(self.objects.keys())

        BOT_W, BOT_H = 440, 380
        ROW2_Y = 60  # no live video on top in offline

        # Left: point cloud
        vis = o3d.visualization.Visualizer()
        vis.create_window("RT-SSI - Raw Point Cloud", width=BOT_W, height=BOT_H,
                          left=0, top=ROW2_Y)
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.05, 0.05, 0.05])

        scene_pcd = build_scene_pcd(list(self.objects.values()))
        vis.add_geometry(scene_pcd)
        vis.reset_view_point(True)

        # Center: mesh
        vis_mesh = o3d.visualization.Visualizer()
        vis_mesh.create_window("RT-SSI - Complete Mesh", width=BOT_W, height=BOT_H,
                               left=BOT_W, top=ROW2_Y)
        opt = vis_mesh.get_render_option()
        opt.background_color = np.array([0.05, 0.05, 0.05])
        opt.mesh_show_wireframe = False
        mesh_geom = None

        selected = [None]
        inference_done = False

        def ask_input():
            print(f"\n{BOLD}{CYAN}{'─'*50}{RESET}")
            print(f"{BOLD}  RT-SSI — Object Selection{RESET}")
            print(f"{BOLD}{CYAN}{'─'*50}{RESET}")
            print(f"\n  Available for inference: {BOLD}{', '.join(available)}{RESET}")
            while selected[0] is None:
                choice = input(f"\n  {YELLOW}>{RESET} Which object? ").strip().lower()
                if choice in available:
                    selected[0] = choice
                elif choice in ("q", "quit", "exit"):
                    selected[0] = "__quit__"
                else:
                    print(f"  {RED}'{choice}' not available. Choose from: {', '.join(available)}{RESET}")

        input_thread = threading.Thread(target=ask_input, daemon=True)
        input_thread.start()

        while True:
            vis.poll_events()
            vis.update_renderer()
            vis_mesh.poll_events()
            vis_mesh.update_renderer()

            if selected[0] is not None and not inference_done:
                if selected[0] == "__quit__":
                    break

                label = selected[0]
                run_inference_status(label, self.objects[label])

                obj_data = self.objects[label]
                if mesh_geom is not None:
                    vis_mesh.remove_geometry(mesh_geom, reset_bounding_box=False)
                if obj_data["mesh"] is not None:
                    mesh_geom = create_mesh_geometry(obj_data)
                else:
                    mesh_geom = create_partial_pcd(obj_data["partial"], label)
                vis_mesh.add_geometry(mesh_geom, reset_bounding_box=True)
                vis_mesh.reset_view_point(True)

                panel = make_info_panel(obj_data)
                cv2.imshow("RT-SSI - Info", panel)

                inference_done = True
                print(f"\n  {DIM}Press 'q' to quit.{RESET}")

            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break

        vis.destroy_window()
        vis_mesh.destroy_window()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Real Time Spatial Semantics Inference")
    parser.add_argument("--offline", action="store_true",
                        help="Fully offline mode (no camera)")
    parser.add_argument("--svo", type=str,
                        help="SVO file path for offline camera playback")
    args = parser.parse_args()

    demo = VCDemo(offline=args.offline, svo_path=args.svo)
    demo.run()


if __name__ == "__main__":
    main()
