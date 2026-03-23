"""VC Showcase Demo: Spatial Semantics + Physics Inference.

Shows two capabilities of FUSE:
1. Spatial Semantics — complete mesh from noisy, incomplete point cloud
2. Physics Inference — height, width, depth, volume, weight from mesh

Usage:
    python -m demos.demo_vc                    # live mode (ZED camera)
    python -m demos.demo_vc --svo path.svo2    # offline camera (SVO file)
    python -m demos.demo_vc --offline          # fully offline (pre-computed data only)

Controls:
    n / right arrow  = next object
    p / left arrow   = previous object
    space            = toggle partial cloud / complete mesh
    q / ESC          = quit
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DEMO_OBJECTS = ["mug", "cup", "fork"]

# Panel dimensions
PANEL_W = 400
PANEL_H = 600


def load_object_data(label):
    """Load pre-computed data for an object. Returns None if missing."""
    obj_dir = DATA_DIR / label
    if not obj_dir.exists():
        return None

    data = {"label": label, "dir": obj_dir}

    # Crop image
    crop_path = obj_dir / "crop.png"
    if crop_path.exists():
        data["crop"] = cv2.imread(str(crop_path))
    else:
        data["crop"] = None

    # Partial point cloud
    partial_path = obj_dir / "partial.npy"
    if partial_path.exists():
        data["partial"] = np.load(str(partial_path))
    else:
        return None  # partial cloud is required

    # Aligned mesh
    mesh_path = obj_dir / "mesh_aligned.obj"
    if mesh_path.exists():
        mesh = trimesh.load(str(mesh_path), force='mesh')
        data["mesh"] = mesh
        data["mesh_vertices"] = np.array(mesh.vertices, dtype=np.float64)
        data["mesh_faces"] = np.array(mesh.faces, dtype=np.int32)
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


def make_info_panel(obj_data, show_mesh=True):
    """Render the OpenCV info panel for an object."""
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)  # dark background

    y = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_s = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    gray = (180, 180, 180)
    accent = (0, 200, 255)  # orange-yellow
    green = (0, 220, 100)

    # Crop image (top-left)
    crop = obj_data.get("crop")
    if crop is not None:
        # Resize crop to fit panel
        crop_h = 120
        scale = crop_h / crop.shape[0]
        crop_w = int(crop.shape[1] * scale)
        crop_w = min(crop_w, 160)
        crop_resized = cv2.resize(crop, (crop_w, crop_h))
        panel[y:y+crop_h, 10:10+crop_w] = crop_resized

        # Title next to crop
        cv2.putText(panel, "FUSE", (crop_w + 20, y + 35),
                    font, 0.9, accent, 2)
        cv2.putText(panel, "Spatial AI", (crop_w + 20, y + 65),
                    font_s, 0.6, gray, 1)
        y += crop_h + 15
    else:
        cv2.putText(panel, "FUSE Spatial AI", (10, y + 30),
                    font, 0.8, accent, 2)
        y += 45

    # Divider
    cv2.line(panel, (10, y), (PANEL_W - 10, y), (80, 80, 80), 1)
    y += 15

    # Object info
    label = obj_data["label"]
    cv2.putText(panel, f"Object: {label.capitalize()}", (10, y + 5),
                font_s, 0.65, white, 1)
    y += 30

    physics = obj_data.get("physics")
    if physics:
        conf_text = f"Material: {physics['material'].capitalize()}"
        cv2.putText(panel, conf_text, (10, y + 5), font_s, 0.55, gray, 1)
        y += 25

    # Divider
    cv2.line(panel, (10, y), (PANEL_W - 10, y), (80, 80, 80), 1)
    y += 20

    # Spatial semantics section
    cv2.putText(panel, "SPATIAL SEMANTICS", (10, y), font_s, 0.55, accent, 1)
    y += 25

    partial = obj_data.get("partial")
    if partial is not None:
        cv2.putText(panel, f"Partial points: {len(partial):,}", (20, y),
                    font_s, 0.5, gray, 1)
        y += 22

    mesh = obj_data.get("mesh")
    if mesh is not None:
        cv2.putText(panel, f"Complete mesh: {len(mesh.faces):,} faces", (20, y),
                    font_s, 0.5, gray, 1)
        y += 22
        cv2.putText(panel, "Completion: ~28s (cloud A100)", (20, y),
                    font_s, 0.5, gray, 1)
        y += 22

    # Current view indicator
    view_text = "Showing: COMPLETE MESH" if show_mesh else "Showing: PARTIAL CLOUD"
    view_color = green if show_mesh else (100, 180, 255)
    cv2.putText(panel, view_text, (20, y), font_s, 0.5, view_color, 1)
    y += 30

    # Divider
    cv2.line(panel, (10, y), (PANEL_W - 10, y), (80, 80, 80), 1)
    y += 20

    # Physics section
    cv2.putText(panel, "PHYSICS PROPERTIES", (10, y), font_s, 0.55, accent, 1)
    y += 25

    if physics:
        props = [
            ("Height", f"{physics['height_cm']} cm"),
            ("Width", f"{physics['width_cm']} cm"),
            ("Depth", f"{physics['depth_cm']} cm"),
            ("Volume", f"{physics['volume_cm3']} cm3"),
            ("Surface", f"{physics['surface_area_cm2']} cm2"),
            ("Weight", f"~{physics['weight_g']:.0f} g ({physics['material']})"),
        ]
        for name, value in props:
            cv2.putText(panel, f"{name}:", (20, y), font_s, 0.5, gray, 1)
            cv2.putText(panel, value, (140, y), font_s, 0.5, white, 1)
            y += 22
    else:
        cv2.putText(panel, "(no physics data)", (20, y), font_s, 0.5, gray, 1)
        y += 22

    # Bottom controls hint
    y = PANEL_H - 30
    cv2.putText(panel, "[N] next  [P] prev  [SPACE] toggle  [Q] quit",
                (10, y), font_s, 0.4, (120, 120, 120), 1)

    return panel


def create_partial_pcd(points, label):
    """Create Open3D point cloud from partial points with label color."""
    from core.fused_object import LABEL_COLORS, DEFAULT_LABEL_COLOR
    color = LABEL_COLORS.get(label, DEFAULT_LABEL_COLOR)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    colors = np.tile(color, (len(points), 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def create_mesh_geometry(obj_data):
    """Create Open3D triangle mesh from loaded trimesh data."""
    mesh = obj_data["mesh"]
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int32)

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.compute_vertex_normals()

    # Color the mesh with a pleasant uniform color
    from core.fused_object import LABEL_COLORS, DEFAULT_LABEL_COLOR
    color = LABEL_COLORS.get(obj_data["label"], DEFAULT_LABEL_COLOR)
    # Slightly desaturated version for mesh
    mesh_color = tuple(min(1.0, c * 0.7 + 0.3) for c in color)
    o3d_mesh.paint_uniform_color(mesh_color)

    return o3d_mesh


class VCDemo:
    def __init__(self, offline=False, svo_path=None):
        self.offline = offline
        self.svo_path = svo_path
        self.objects = []
        self.current_idx = 0
        self.show_mesh = False  # start with partial cloud

        # Open3D visualizer
        self.vis = None
        self.current_geometry = None

    def load_data(self):
        """Load all available pre-computed object data."""
        for label in DEMO_OBJECTS:
            data = load_object_data(label)
            if data is not None:
                self.objects.append(data)
                print(f"Loaded {label}: {len(data['partial'])} partial pts"
                      + (f", {len(data['mesh'].faces)} mesh faces" if data['mesh'] else "")
                      + (" + physics" if data['physics'] else ""))
            else:
                print(f"Skipping {label}: missing data in {DATA_DIR / label}")

        if not self.objects:
            print("ERROR: No object data found. Run precompute_demo_data.py first.")
            sys.exit(1)

        print(f"\nLoaded {len(self.objects)} objects: "
              + ", ".join(o['label'] for o in self.objects))

    def setup_visualizer(self):
        """Create Open3D visualization window."""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("FUSE - 3D View", width=800, height=600,
                               left=420, top=50)
        opt = self.vis.get_render_option()
        opt.point_size = 3.0
        opt.background_color = np.array([0.1, 0.1, 0.1])
        opt.mesh_show_wireframe = False

    def update_3d_view(self):
        """Update the Open3D window with current object and view mode."""
        obj = self.objects[self.current_idx]

        # Remove old geometry
        if self.current_geometry is not None:
            self.vis.remove_geometry(self.current_geometry, reset_bounding_box=False)

        if self.show_mesh and obj["mesh"] is not None:
            self.current_geometry = create_mesh_geometry(obj)
        else:
            self.current_geometry = create_partial_pcd(obj["partial"], obj["label"])

        self.vis.add_geometry(self.current_geometry, reset_bounding_box=True)
        self.vis.reset_view_point(True)

    def run_live(self):
        """Run with live camera feed + pre-computed 3D overlays."""
        from core.pipeline import FUSEPipeline
        from demos.phase4_pipeline import draw_objects

        classes = [o["label"] for o in self.objects]

        with FUSEPipeline(classes, svo_path=self.svo_path) as pipe:
            print("Live mode — press 'q' to quit")
            prev_time = time.time()
            fps = 0.0

            while True:
                bgr, detected, _, _ = pipe.process_frame(skip_scene=True)
                if bgr is None:
                    if self.svo_path:
                        print("End of SVO file.")
                        break
                    continue

                # Draw detections on frame
                frame = draw_objects(bgr, detected)

                # FPS
                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
                prev_time = now
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Show camera feed
                cv2.imshow("FUSE - Camera", frame)

                # Info panel
                obj = self.objects[self.current_idx]
                panel = make_info_panel(obj, self.show_mesh)
                cv2.imshow("FUSE - Info", panel)

                # 3D view
                self.vis.poll_events()
                self.vis.update_renderer()

                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key):
                    break

    def run_offline(self):
        """Run fully offline with pre-computed data only."""
        print("Offline mode — press 'q' to quit")

        while True:
            obj = self.objects[self.current_idx]
            panel = make_info_panel(obj, self.show_mesh)
            cv2.imshow("FUSE - Info", panel)

            # Show crop as standalone window if available
            if obj["crop"] is not None:
                cv2.imshow("FUSE - Input Crop", obj["crop"])

            # 3D view
            self.vis.poll_events()
            self.vis.update_renderer()

            key = cv2.waitKey(30) & 0xFF
            if not self._handle_key(key):
                break

    def _handle_key(self, key):
        """Handle keyboard input. Returns False to quit."""
        if key in (ord('q'), 27):  # q or ESC
            return False
        elif key in (ord('n'), 83):  # n or right arrow
            self.current_idx = (self.current_idx + 1) % len(self.objects)
            self.show_mesh = False
            self.update_3d_view()
        elif key in (ord('p'), 81):  # p or left arrow
            self.current_idx = (self.current_idx - 1) % len(self.objects)
            self.show_mesh = False
            self.update_3d_view()
        elif key == ord(' '):  # space = toggle
            if self.objects[self.current_idx]["mesh"] is not None:
                self.show_mesh = not self.show_mesh
                self.update_3d_view()
        return True

    def run(self):
        self.load_data()
        self.setup_visualizer()
        self.update_3d_view()

        try:
            if self.offline:
                self.run_offline()
            else:
                self.run_live()
        finally:
            self.vis.destroy_window()
            cv2.destroyAllWindows()


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
