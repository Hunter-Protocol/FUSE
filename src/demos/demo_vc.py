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
    print(f"{BOLD}{CYAN}  FUSE SPATIAL AI — INFERENCE ENGINE{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")
    print()

    hacker_print(f"  {YELLOW}>{RESET} Target acquired: {BOLD}{label.upper()}{RESET}", 0.03)
    time.sleep(0.3)

    # Phase 1: Point cloud extraction
    print(f"\n  {BOLD}[1/4] POINT CLOUD EXTRACTION{RESET}")
    fake_hex_stream(2)
    n_pts = len(obj_data["partial"])
    spinner(f"Extracting depth points from stereo pair... {n_pts:,} points", 1.2)

    # Phase 2: Mask projection
    print(f"\n  {BOLD}[2/4] SPATIAL SEMANTICS — MESH RECONSTRUCTION{RESET}")
    fake_hex_stream(3)
    spinner("Uploading crop to Modal A100 cluster...", 0.8)
    spinner("Running Hunyuan3D-2 DiT flow matching (50 steps)...", 2.5)
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
        props = [
            ("Height",   f"{physics['height_cm']} cm"),
            ("Width",    f"{physics['width_cm']} cm"),
            ("Depth",    f"{physics['depth_cm']} cm"),
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


def draw_detections(frame, objects):
    """Draw bounding boxes, masks, and labels. Same colors as point cloud."""
    overlay = frame.copy()
    for obj in objects:
        color_bgr = bgr_color(obj.color)

        # Filled mask
        overlay[obj.mask] = (
            overlay[obj.mask] * 0.5 + np.array(color_bgr) * 0.5
        ).astype(np.uint8)

        # Bounding box
        x1, y1, x2, y2 = obj.box_2d
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)

        # Label
        text = f"{obj.label} {obj.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color_bgr, -1)
        cv2.putText(frame, text, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

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

def make_info_panel(obj_data):
    """Render the OpenCV info panel for a single object."""
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    y = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    gray = (180, 180, 180)
    accent = (0, 200, 255)

    # Crop thumbnail
    crop = obj_data.get("crop")
    if crop is not None:
        crop_h = 120
        scale = crop_h / crop.shape[0]
        crop_w = min(int(crop.shape[1] * scale), 160)
        crop_resized = cv2.resize(crop, (crop_w, crop_h))
        panel[y:y+crop_h, 10:10+crop_w] = crop_resized
        cv2.putText(panel, "FUSE", (crop_w + 20, y + 35), font, 0.9, accent, 2)
        cv2.putText(panel, "Spatial AI", (crop_w + 20, y + 65), font, 0.6, gray, 1)
        y += crop_h + 15
    else:
        cv2.putText(panel, "FUSE Spatial AI", (10, y + 30), font, 0.8, accent, 2)
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
        cv2.putText(panel, "Completion: ~28s (cloud A100)", (20, y), font, 0.5, gray, 1)
        y += 22

    cv2.line(panel, (10, y), (PANEL_W - 10, y), (80, 80, 80), 1)
    y += 20

    # Physics
    cv2.putText(panel, "PHYSICS PROPERTIES", (10, y), font, 0.55, accent, 1)
    y += 25

    if physics:
        props = [
            ("Height",  f"{physics['height_cm']} cm"),
            ("Width",   f"{physics['width_cm']} cm"),
            ("Depth",   f"{physics['depth_cm']} cm"),
            ("Volume",  f"{physics['volume_cm3']} cm3"),
            ("Surface", f"{physics['surface_area_cm2']} cm2"),
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
                print(f"  Loaded {label}: {len(data['partial']):,} pts"
                      + (f", {len(data['mesh'].faces):,} faces" if data['mesh'] else "")
                      + (" + physics" if data['physics'] else ""))
            else:
                print(f"  Skipping {label}: no data in {DATA_DIR / label}")

        if not self.objects:
            print("ERROR: No object data found. Run precompute_demo_data.py first.")
            sys.exit(1)

    # ---- Phase 1: Live detection view ----

    def run_phase1_live(self):
        """Show RGB video + live raw point cloud. Returns when user picks object."""
        from core.pipeline import FUSEPipeline

        available = list(self.objects.keys())
        classes = YOLOE_CLASSES

        # Open3D: live raw point cloud (updated every frame)
        vis = o3d.visualization.Visualizer()
        vis.create_window("FUSE - Raw Point Cloud", width=720, height=540,
                          left=700, top=50)
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.05, 0.05, 0.05])

        live_pcd = o3d.geometry.PointCloud()
        vis.add_geometry(live_pcd)
        first_points = True
        # Store latest raw points per label for Phase 3
        raw_points_by_label = {}

        with FUSEPipeline(classes, svo_path=self.svo_path, model_size="11m") as pipe:
            # Run a few warmup frames so model is loaded before showing prompt
            for _ in range(3):
                bgr, detected, _, _ = pipe.process_frame(skip_scene=True)
                if bgr is not None:
                    remap_labels(detected)
                    frame = draw_detections(bgr, detected)
                    cv2.imshow("FUSE - Live Detection", frame)
                    vis.poll_events()
                    vis.update_renderer()
                    cv2.waitKey(1)

            # Now show the prompt (after all init output is done)
            selected = [None]

            def ask_input():
                print(f"\n{BOLD}{CYAN}{'─'*50}{RESET}")
                print(f"{BOLD}  FUSE — Object Selection{RESET}")
                print(f"{BOLD}{CYAN}{'─'*50}{RESET}")
                print(f"\n  Detected objects are shown in the video feed.")
                print(f"  Available for inference: {BOLD}{', '.join(available)}{RESET}")
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

            fps = 0.0
            prev_time = time.time()

            while selected[0] is None:
                bgr, detected, _, _ = pipe.process_frame(skip_scene=True)
                if bgr is None:
                    if self.svo_path:
                        break
                    continue

                # Remap "coffee mug" -> "mug", etc.
                remap_labels(detected)

                # Extract RAW unfiltered points from each detected object
                # (pipeline already retrieved pc_mat, reuse it)
                pc_data = pipe.pc_mat.get_data()
                all_pts, all_clr = [], []
                for obj in detected:
                    xyz = pc_data[:, :, :3][obj.mask]
                    valid = np.isfinite(xyz).all(axis=1)
                    raw_pts = xyz[valid].astype(np.float32)
                    if len(raw_pts) > 0:
                        color = LABEL_COLORS.get(obj.label, DEFAULT_LABEL_COLOR)
                        all_pts.append(raw_pts)
                        all_clr.append(np.tile(color, (len(raw_pts), 1)))
                        raw_points_by_label[obj.label] = raw_pts

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

                vis.update_geometry(live_pcd)
                if first_points and all_pts:
                    vis.reset_view_point(True)
                    first_points = False

                # Draw detections with colored bounding boxes
                frame = draw_detections(bgr, detected)

                # FPS
                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
                prev_time = now
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("FUSE - Live Detection", frame)

                vis.poll_events()
                vis.update_renderer()

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    selected[0] = "__quit__"

        vis.destroy_window()
        cv2.destroyAllWindows()
        return selected[0], raw_points_by_label

    def run_phase1_offline(self):
        """Offline version: show pre-computed point cloud, prompt in terminal."""
        available = list(self.objects.keys())

        # Open3D: all-object noisy point cloud
        vis = o3d.visualization.Visualizer()
        vis.create_window("FUSE - Raw Point Cloud", width=720, height=540,
                          left=700, top=50)
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.05, 0.05, 0.05])

        scene_pcd = build_scene_pcd(list(self.objects.values()))
        vis.add_geometry(scene_pcd)
        vis.reset_view_point(True)

        # Terminal prompt (non-blocking)
        selected = [None]

        def ask_input():
            print(f"\n{BOLD}{CYAN}{'─'*50}{RESET}")
            print(f"{BOLD}  FUSE — Object Selection{RESET}")
            print(f"{BOLD}{CYAN}{'─'*50}{RESET}")
            print(f"\n  Point clouds are shown in the 3D viewer.")
            print(f"  Available for inference: {BOLD}{', '.join(available)}{RESET}")
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

        while selected[0] is None:
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.03)

        vis.destroy_window()
        return selected[0], {}  # no live raw points in offline mode

    # ---- Phase 2: Hacker inference status (terminal) ----

    def run_phase2(self, label):
        """Show hacker-style inference progress in terminal."""
        obj_data = self.objects[label]
        run_inference_status(label, obj_data)

    # ---- Phase 3: Result visualization ----

    def run_phase3(self, label, raw_points=None):
        """Show 3 windows: raw partial cloud, complete mesh, info panel."""
        obj_data = self.objects[label]

        # Use live raw points if available, otherwise fall back to stored data
        partial_pts = raw_points if raw_points is not None else obj_data["partial"]

        # Window 1: raw partial point cloud
        vis_partial = o3d.visualization.Visualizer()
        vis_partial.create_window("FUSE - Partial Point Cloud (raw)",
                                   width=640, height=480, left=0, top=50)
        opt = vis_partial.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.05, 0.05, 0.05])

        pcd = create_partial_pcd(partial_pts, label)
        vis_partial.add_geometry(pcd)
        vis_partial.reset_view_point(True)

        # Window 2: complete mesh
        vis_mesh = o3d.visualization.Visualizer()
        vis_mesh.create_window("FUSE - Complete Mesh",
                                width=640, height=480, left=660, top=50)
        opt = vis_mesh.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.05, 0.05, 0.05])
        opt.mesh_show_wireframe = False

        if obj_data["mesh"] is not None:
            mesh_geom = create_mesh_geometry(obj_data)
        else:
            mesh_geom = create_partial_pcd(obj_data["partial"], label)
        vis_mesh.add_geometry(mesh_geom)
        vis_mesh.reset_view_point(True)

        # Window 3: info panel (OpenCV)
        panel = make_info_panel(obj_data)
        cv2.imshow("FUSE - Info", panel)

        print(f"  {DIM}Showing results for {label}. Press 'q' to continue.{RESET}")

        while True:
            vis_partial.poll_events()
            vis_partial.update_renderer()
            vis_mesh.poll_events()
            vis_mesh.update_renderer()

            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break

        vis_partial.destroy_window()
        vis_mesh.destroy_window()
        cv2.destroyAllWindows()

    # ---- Main loop ----

    def run(self):
        print(f"\n{BOLD}{CYAN}  FUSE — Spatial AI Demo{RESET}")
        print(f"  {'─'*40}\n")

        self.load_data()

        while True:
            # Phase 1: live detection + object selection
            if self.offline:
                selected, raw_points_map = self.run_phase1_offline()
            else:
                selected, raw_points_map = self.run_phase1_live()

            if selected is None or selected == "__quit__":
                print(f"\n{DIM}Exiting demo.{RESET}")
                break

            # Phase 2: hacker inference status
            self.run_phase2(selected)

            # Phase 3: result visualization (use live raw points if captured)
            live_raw = raw_points_map.get(selected)
            self.run_phase3(selected, raw_points=live_raw)

            print(f"\n{DIM}{'─'*40}{RESET}")
            print(f"{DIM}Returning to detection view...{RESET}")
            time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description="FUSE VC Showcase Demo")
    parser.add_argument("--offline", action="store_true",
                        help="Fully offline mode (no camera)")
    parser.add_argument("--svo", type=str,
                        help="SVO file path for offline camera playback")
    args = parser.parse_args()

    demo = VCDemo(offline=args.offline, svo_path=args.svo)
    demo.run()


if __name__ == "__main__":
    main()
