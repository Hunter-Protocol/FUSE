"""Pre-compute demo data for VC showcase.

For each object: detect with FUSE pipeline, save crop + partial cloud,
generate mesh via Hunyuan3D Modal, align to camera frame, compute physics.

Usage:
    python -m demos.precompute_demo_data --object mug
    python -m demos.precompute_demo_data --all
    python -m demos.precompute_demo_data --object mug --svo recording.svo2
    python -m demos.precompute_demo_data --object mug --skip-mesh  # reuse existing mesh
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DEMO_OBJECTS = ["mug", "cup", "fork"]


def detect_and_save(object_label, svo_path=None):
    """Run pipeline, find the target object, save crop + partial clouds.

    Saves both the filtered partial cloud (used for alignment) and the raw
    noisy cloud (pre-outlier-removal, for VC demo visualization).
    """
    import pyzed.sl as sl
    from core.pipeline import FUSEPipeline

    classes = [object_label]
    obj_dir = DATA_DIR / object_label
    obj_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Detecting: {object_label}")
    print(f"{'='*50}")

    with FUSEPipeline(classes, svo_path=svo_path, model_size="11m") as pipe:
        best_obj = None
        best_bgr = None
        best_raw_points = None
        # Try multiple frames to get a good detection
        for attempt in range(30):
            bgr, objects, _, _ = pipe.process_frame(skip_scene=True)
            if bgr is None:
                continue
            for obj in objects:
                if obj.label == object_label and obj.source == "fused":
                    if best_obj is None or obj.num_points > best_obj.num_points:
                        best_obj = obj
                        best_bgr = bgr.copy()
                        # Extract raw (unfiltered) points from the mask
                        pc_data = pipe.pc_mat.get_data()
                        xyz = pc_data[:, :, :3][obj.mask]
                        valid = np.isfinite(xyz).all(axis=1)
                        best_raw_points = xyz[valid].astype(np.float32)
            if best_obj and best_obj.num_points > 500:
                break

    if best_obj is None:
        print(f"ERROR: Could not detect '{object_label}' — ensure it's visible to the camera.")
        return None

    print(f"Detected {object_label}: {best_obj.num_points} filtered pts, "
          f"{len(best_raw_points)} raw pts, conf={best_obj.confidence:.2f}")

    # Save crop (YOLOE bounding box region)
    x1, y1, x2, y2 = best_obj.box_2d
    pad = 10
    h, w = best_bgr.shape[:2]
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = best_bgr[y1:y2, x1:x2]
    crop_path = obj_dir / "crop.png"
    cv2.imwrite(str(crop_path), crop)
    print(f"Saved crop: {crop_path}")

    # Save filtered partial cloud (for alignment)
    partial_path = obj_dir / "partial.npy"
    np.save(str(partial_path), best_obj.points_3d)
    print(f"Saved filtered partial: {partial_path} ({best_obj.num_points} points)")

    # Save raw noisy cloud (for VC demo visualization)
    raw_path = obj_dir / "partial_raw.npy"
    np.save(str(raw_path), best_raw_points)
    print(f"Saved raw partial: {raw_path} ({len(best_raw_points)} points)")

    return obj_dir


def generate_mesh(object_label):
    """Call Hunyuan3D Modal endpoint and save mesh."""
    import modal
    from PIL import Image

    obj_dir = DATA_DIR / object_label
    crop_path = obj_dir / "crop.png"

    if not crop_path.exists():
        print(f"ERROR: No crop found at {crop_path}. Run detection first.")
        return False

    print(f"\nGenerating mesh for {object_label} via Hunyuan3D...")
    image = Image.open(crop_path).convert("RGBA")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    Hunyuan3DModel = modal.Cls.from_name("fuse-hunyuan3d", "Hunyuan3DModel")
    model = Hunyuan3DModel()

    t0 = time.time()
    result = model.generate.remote(image_bytes)
    dt = time.time() - t0
    print(f"Generation: {result['generation_time']:.2f}s (total latency: {dt:.2f}s)")

    vertices = np.array(result['vertices'], dtype=np.float32)
    faces = np.array(result['faces'], dtype=np.int32)
    print(f"Mesh: {len(vertices)} verts, {len(faces)} faces")

    # Save canonical mesh
    canonical_path = obj_dir / "mesh_canonical.npz"
    np.savez(str(canonical_path), vertices=vertices, faces=faces)
    print(f"Saved canonical mesh: {canonical_path}")

    return True


def align_mesh(object_label):
    """Align canonical mesh to partial cloud and save as OBJ."""
    import trimesh
    from sim.save_mesh import align_mesh_to_partial

    obj_dir = DATA_DIR / object_label
    canonical_path = obj_dir / "mesh_canonical.npz"
    partial_path = obj_dir / "partial.npy"

    if not canonical_path.exists() or not partial_path.exists():
        print(f"ERROR: Missing canonical mesh or partial cloud for {object_label}.")
        return False

    data = np.load(str(canonical_path))
    vertices, faces = data['vertices'], data['faces']
    partial_pts = np.load(str(partial_path))

    print(f"\nAligning {object_label} mesh to camera frame...")
    aligned_verts, align_info = align_mesh_to_partial(vertices, partial_pts)

    aligned_mesh = trimesh.Trimesh(vertices=aligned_verts, faces=faces)
    aligned_path = obj_dir / "mesh_aligned.obj"
    aligned_mesh.export(str(aligned_path))
    print(f"Saved aligned mesh: {aligned_path}")
    print(f"  ICP fitness: {align_info['icp_fitness']:.3f}, RMSE: {align_info['icp_rmse']:.4f}m")

    return True


def compute_physics(object_label):
    """Compute and cache physics properties."""
    from core.physics import estimate_physics

    obj_dir = DATA_DIR / object_label
    mesh_path = obj_dir / "mesh_aligned.obj"
    partial_path = obj_dir / "partial.npy"

    if not mesh_path.exists():
        print(f"ERROR: No aligned mesh found for {object_label}.")
        return False

    partial_pts = np.load(str(partial_path)) if partial_path.exists() else None

    print(f"\nComputing physics for {object_label}...")
    props = estimate_physics(object_label, str(mesh_path), partial_pts)

    physics_path = obj_dir / "physics.json"
    with open(physics_path, 'w') as f:
        json.dump(props, f, indent=2)

    print(f"Saved physics: {physics_path}")
    for k, v in props.items():
        print(f"  {k}: {v}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Pre-compute VC demo data")
    parser.add_argument("--object", choices=DEMO_OBJECTS, help="Single object to process")
    parser.add_argument("--all", action="store_true", help="Process all demo objects")
    parser.add_argument("--svo", type=str, help="SVO file for offline camera input")
    parser.add_argument("--skip-mesh", action="store_true",
                        help="Skip mesh generation (reuse existing)")
    parser.add_argument("--physics-only", action="store_true",
                        help="Only recompute physics from existing mesh")
    args = parser.parse_args()

    if not args.object and not args.all:
        parser.error("Specify --object <name> or --all")

    objects = DEMO_OBJECTS if args.all else [args.object]

    for obj_label in objects:
        if args.physics_only:
            compute_physics(obj_label)
            continue

        # Step 1: Detect and save crop + partial cloud
        obj_dir = DATA_DIR / obj_label
        if not (obj_dir / "partial.npy").exists():
            detect_and_save(obj_label, svo_path=args.svo)
        else:
            print(f"\nUsing existing detection data for {obj_label}")

        # Step 2: Generate mesh
        if not args.skip_mesh:
            generate_mesh(obj_label)
        else:
            print(f"\nSkipping mesh generation for {obj_label} (--skip-mesh)")

        # Step 3: Align mesh
        if (obj_dir / "mesh_canonical.npz").exists():
            align_mesh(obj_label)

        # Step 4: Compute physics
        if (obj_dir / "mesh_aligned.obj").exists():
            compute_physics(obj_label)


if __name__ == "__main__":
    main()
