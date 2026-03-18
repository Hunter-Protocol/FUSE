"""End-to-end demo: Hunyuan3D mesh → CoACD → MuJoCo → Contact-GraspNet → grasp sim.

Runs the full Path B pipeline on a single object and visualizes results.

Usage:
    python -m sim.run_demo                      # use existing mug data
    python -m sim.run_demo --visualize          # launch MuJoCo viewer for best grasp
"""

import argparse
import os
import time
import numpy as np

DATA_DIR = "/home/hunter/Desktop/FUSE/data"
MESH_DIR = f"{DATA_DIR}/meshes"
MJCF_DIR = f"{DATA_DIR}/mjcf"
RESULTS_DIR = "/home/hunter/Desktop/FUSE/results"


def run_pipeline(object_name="mug", target_height=0.10, visualize=False):
    """Run full grasp evaluation pipeline on a single object."""

    print("=" * 60)
    print(f"FUSE Path B Demo: {object_name}")
    print("=" * 60)

    # --- Step 1: Check for mesh ---
    mesh_path = f"{MESH_DIR}/{object_name}_complete.obj"
    if not os.path.exists(mesh_path):
        # Fall back to canonical mesh
        canonical_path = f"{MESH_DIR}/{object_name}_complete_canonical.obj"
        if os.path.exists(canonical_path):
            mesh_path = canonical_path
        else:
            print(f"ERROR: No mesh found at {mesh_path}")
            print("Run save_mesh.py first to get mesh from Modal Hunyuan3D.")
            return

    print(f"\n[1/4] Mesh: {mesh_path}")

    # --- Step 2: CoACD + MuJoCo XML ---
    mjcf_name = f"{object_name}_complete"
    mjcf_dir = f"{MJCF_DIR}/{mjcf_name}"
    xml_path = f"{mjcf_dir}/{mjcf_name}.xml"

    if not os.path.exists(xml_path):
        print(f"\n[2/4] Building MuJoCo scene (CoACD decomposition)...")
        from sim.mesh_to_mjcf import mesh_to_mjcf
        xml_path = mesh_to_mjcf(mesh_path, mjcf_name, target_height=target_height)
    else:
        print(f"\n[2/4] MuJoCo scene exists: {xml_path}")

    # Verify
    from sim.mesh_to_mjcf import verify_mjcf
    verify_mjcf(xml_path)

    # --- Step 3: Plan grasps ---
    print(f"\n[3/4] Planning grasps with Contact-GraspNet...")
    grasps_path = f"{RESULTS_DIR}/{object_name}_grasps.npz"

    # Load points — use canonical frame for grasp planning
    points_path = f"{DATA_DIR}/{object_name}_hunyuan3d_cloud.npy"
    if not os.path.exists(points_path):
        print(f"ERROR: No points found at {points_path}")
        return

    points = np.load(points_path).astype(np.float32)
    print(f"  Point cloud: {points.shape}")

    from sim.plan_grasps import plan_grasps, load_grasp_estimator
    estimator, config = load_grasp_estimator()
    grasps = plan_grasps(points, estimator, config, top_k=50)
    print(f"  Generated {len(grasps)} grasp candidates")

    if not grasps:
        print("ERROR: No grasps generated!")
        return

    # Transform grasps from canonical frame to MuJoCo frame
    # The mesh was recentered (centroid→origin, bottom→z=0) and scaled
    import trimesh
    original_mesh = trimesh.load(mesh_path)
    canonical_center = original_mesh.centroid.copy()
    canonical_bottom = original_mesh.bounds[0][2]
    canonical_height = original_mesh.extents[2]
    scale = target_height / canonical_height

    # Transform: subtract center, subtract bottom offset, scale
    for g in grasps:
        pose = g['pose'].copy()
        # Transform position
        pos = pose[:3, 3]
        pos -= canonical_center
        pos[2] -= (original_mesh.bounds[0][2] - canonical_center[2])
        pos *= scale
        pose[:3, 3] = pos
        g['pose_mujoco'] = pose

    # --- Step 4: Evaluate grasps in MuJoCo ---
    print(f"\n[4/4] Evaluating grasps in MuJoCo simulation...")
    from sim.grasp_eval import evaluate_grasp

    results = []
    n_success = 0
    n_eval = min(20, len(grasps))  # evaluate top 20

    for i in range(n_eval):
        g = grasps[i]
        try:
            result = evaluate_grasp(mjcf_dir, mjcf_name, g['pose_mujoco'])
            results.append(result)
            if result['success']:
                n_success += 1
            status = "PASS" if result['success'] else "FAIL"
            print(f"  Grasp {i:2d}: {status} (score={g['score']:.4f}, lift={result['lift_delta']:.3f}m)")
        except Exception as e:
            print(f"  Grasp {i:2d}: ERROR ({e})")
            results.append({'success': False, 'error': str(e)})

    rate = n_success / max(n_eval, 1)
    print(f"\n{'=' * 60}")
    print(f"Results: {n_success}/{n_eval} grasps succeeded ({rate:.0%})")
    print(f"{'=' * 60}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(
        f"{RESULTS_DIR}/{object_name}_eval.npz",
        success_rate=rate,
        n_success=n_success,
        n_total=n_eval,
        grasp_scores=np.array([g['score'] for g in grasps[:n_eval]]),
        grasp_results=np.array([r.get('success', False) for r in results]),
    )
    print(f"Saved results to {RESULTS_DIR}/{object_name}_eval.npz")

    # --- Visualize best grasp ---
    if visualize and n_success > 0:
        # Find first successful grasp
        for i, r in enumerate(results):
            if r.get('success', False):
                print(f"\nLaunching MuJoCo viewer with grasp {i}...")
                from sim.visualize import simulate_grasp_with_viewer
                simulate_grasp_with_viewer(mjcf_dir, mjcf_name, grasps[i]['pose_mujoco'])
                break


def main():
    parser = argparse.ArgumentParser(description="FUSE Path B end-to-end demo")
    parser.add_argument("--object", default="mug", help="Object name")
    parser.add_argument("--height", type=float, default=0.10, help="Target height in meters")
    parser.add_argument("--visualize", action="store_true", help="Launch MuJoCo viewer")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_pipeline(args.object, args.height, args.visualize)


if __name__ == "__main__":
    main()
