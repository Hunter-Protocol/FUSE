"""Send any image to Hunyuan3D on Modal and visualize the result.

Usage:
    python -m demos.run_hunyuan3d image.png                     # generate + view
    python -m demos.run_hunyuan3d image.png --output out.obj    # custom output path
    python -m demos.run_hunyuan3d image.png --label cup         # also align + physics
    python -m demos.run_hunyuan3d image.png --label cup --save  # save into data/cup/
    python -m demos.run_hunyuan3d image.png --no-vis            # skip visualization
"""

import argparse
import io
import time
from pathlib import Path

import cv2
import modal
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def main():
    parser = argparse.ArgumentParser(description="Run Hunyuan3D on a custom image")
    parser.add_argument("image", type=str, help="Input image path (PNG/JPEG)")
    parser.add_argument("--output", type=str, help="Output OBJ path (default: <image>_mesh.obj)")
    parser.add_argument("--label", type=str, help="Object label (for alignment + physics)")
    parser.add_argument("--save", action="store_true",
                        help="Save into data/<label>/ (requires --label)")
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip visualization")
    parser.add_argument("--model", type=str, default="v20", choices=["v20", "v21"],
                        help="Model version: v20 (1.1B) or v21 (3.3B, higher detail)")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return

    # Load and send image
    print(f"Input: {image_path} ({image_path.stat().st_size // 1024} KB)")
    image = Image.open(image_path).convert("RGBA")
    print(f"Size: {image.size}, mode: {image.mode}")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    # Call Modal endpoint
    if args.model == "v21":
        print(f"\nSending to Hunyuan3D v2.1 (3.3B) on Modal A100...")
        ModelCls = modal.Cls.from_name("fuse-hunyuan3d-v21", "Hunyuan3DV21Model")
    else:
        print(f"\nSending to Hunyuan3D v2.0 (1.1B) on Modal A100...")
        ModelCls = modal.Cls.from_name("fuse-hunyuan3d", "Hunyuan3DModel")
    model = ModelCls()

    t0 = time.time()
    result = model.generate.remote(image_bytes)
    dt = time.time() - t0

    vertices = np.array(result['vertices'], dtype=np.float32)
    faces = np.array(result['faces'], dtype=np.int32)
    print(f"\nGeneration: {result['generation_time']:.2f}s (total latency: {dt:.2f}s)")
    print(f"Mesh: {len(vertices)} verts, {len(faces)} faces")

    # Save canonical mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    aligned_mesh = None
    partial_pts = None

    if args.save and args.label:
        # Save into data/<label>/ structure
        obj_dir = DATA_DIR / args.label
        obj_dir.mkdir(parents=True, exist_ok=True)

        # Copy input image as crop (skip if same file)
        crop_dst = obj_dir / "crop.png"
        if image_path.resolve() != crop_dst.resolve():
            import shutil
            shutil.copy2(image_path, crop_dst)
            print(f"Saved crop: {crop_dst}")

        # Save canonical mesh
        np.savez(str(obj_dir / "mesh_canonical.npz"), vertices=vertices, faces=faces)
        print(f"Saved canonical: {obj_dir / 'mesh_canonical.npz'}")

        # Align if partial cloud exists
        partial_path = obj_dir / "partial.npy"
        if partial_path.exists():
            from sim.save_mesh import align_mesh_to_partial
            partial_pts = np.load(str(partial_path))
            print(f"\nAligning to partial cloud ({len(partial_pts)} pts)...")
            aligned_verts, info = align_mesh_to_partial(vertices, partial_pts)
            aligned_mesh = trimesh.Trimesh(vertices=aligned_verts, faces=faces)
            aligned_mesh.export(str(obj_dir / "mesh_aligned.obj"))
            print(f"Saved aligned: {obj_dir / 'mesh_aligned.obj'}")
            print(f"  ICP fitness: {info['icp_fitness']:.3f}, RMSE: {info['icp_rmse']:.4f}m")

            # Compute physics
            from core.physics import estimate_physics
            import json
            props = estimate_physics(args.label, str(obj_dir / "mesh_aligned.obj"), partial_pts)
            with open(obj_dir / "physics.json", 'w') as f:
                json.dump(props, f, indent=2)
            print(f"\nPhysics:")
            for k, v in props.items():
                print(f"  {k}: {v}")
        else:
            # Just save canonical OBJ
            mesh.export(str(obj_dir / "mesh_canonical.obj"))
            print(f"No partial cloud found — skipping alignment + physics")

    else:
        # Save to output path
        out_path = args.output or str(image_path.with_suffix('.obj'))
        mesh.export(out_path)
        print(f"\nSaved: {out_path}")

        # Align if label + partial cloud provided
        if args.label:
            partial_path = DATA_DIR / args.label / "partial.npy"
            if partial_path.exists():
                from sim.save_mesh import align_mesh_to_partial
                partial_pts = np.load(str(partial_path))
                print(f"\nAligning to partial cloud ({len(partial_pts)} pts)...")
                aligned_verts, info = align_mesh_to_partial(vertices, partial_pts)
                aligned_mesh = trimesh.Trimesh(vertices=aligned_verts, faces=faces)
                aligned_path = str(Path(out_path).with_name(
                    Path(out_path).stem + "_aligned.obj"))
                aligned_mesh.export(aligned_path)
                print(f"Saved aligned: {aligned_path}")
                print(f"  ICP fitness: {info['icp_fitness']:.3f}, RMSE: {info['icp_rmse']:.4f}m")

    # Visualize
    if not args.no_vis:
        visualize(image_path, mesh, aligned_mesh=aligned_mesh, partial_pts=partial_pts)


def visualize(image_path, mesh, aligned_mesh=None, partial_pts=None):
    """Show input image + generated mesh (+ optional aligned mesh vs partial cloud)."""

    # Window 1: input image (OpenCV)
    img = cv2.imread(str(image_path))
    if img is not None:
        # Resize for display if too large
        max_dim = 512
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        cv2.imshow("Input Image", img)

    # Window 2: generated mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(
        np.array(mesh.vertices, dtype=np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(
        np.array(mesh.faces, dtype=np.int32))
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.paint_uniform_color([0.7, 0.5, 0.3])

    vis_mesh = o3d.visualization.Visualizer()
    vis_mesh.create_window("Hunyuan3D - Generated Mesh",
                            width=720, height=540, left=550, top=50)
    opt = vis_mesh.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.05])
    opt.mesh_show_wireframe = False
    vis_mesh.add_geometry(o3d_mesh)
    vis_mesh.reset_view_point(True)

    # Window 3: aligned mesh + partial cloud (if available)
    vis_aligned = None
    if aligned_mesh is not None and partial_pts is not None:
        vis_aligned = o3d.visualization.Visualizer()
        vis_aligned.create_window("Aligned Mesh + Partial Cloud",
                                   width=720, height=540, left=550, top=620)
        opt = vis_aligned.get_render_option()
        opt.background_color = np.array([0.05, 0.05, 0.05])
        opt.point_size = 3.0

        # Partial cloud in blue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(partial_pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(
            np.tile([0.2, 0.6, 1.0], (len(partial_pts), 1)))
        vis_aligned.add_geometry(pcd)

        # Aligned mesh in orange (translucent via wireframe)
        o3d_aligned = o3d.geometry.TriangleMesh()
        o3d_aligned.vertices = o3d.utility.Vector3dVector(
            np.array(aligned_mesh.vertices, dtype=np.float64))
        o3d_aligned.triangles = o3d.utility.Vector3iVector(
            np.array(aligned_mesh.faces, dtype=np.int32))
        o3d_aligned.compute_vertex_normals()
        o3d_aligned.paint_uniform_color([1.0, 0.5, 0.2])
        vis_aligned.add_geometry(o3d_aligned)
        vis_aligned.reset_view_point(True)

    print("\nVisualization open. Press 'q' in the image window to close.")

    while True:
        vis_mesh.poll_events()
        vis_mesh.update_renderer()
        if vis_aligned:
            vis_aligned.poll_events()
            vis_aligned.update_renderer()

        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):
            break

    vis_mesh.destroy_window()
    if vis_aligned:
        vis_aligned.destroy_window()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
