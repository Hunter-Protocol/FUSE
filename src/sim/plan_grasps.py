"""Step 4: Contact-GraspNet wrapper — predict 6-DOF grasps from point clouds.

Takes an Nx3 point cloud and returns ranked grasp poses (4x4 SE3) with scores.

Usage:
    python -m sim.plan_grasps --points data/mug_hunyuan3d_cloud.npy
"""

import argparse
import os
import sys
import numpy as np

# Add third_party to path for contact_graspnet_pytorch
THIRD_PARTY = os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'contact_graspnet_pytorch')
sys.path.insert(0, os.path.abspath(THIRD_PARTY))

CKPT_DIR = os.path.join(THIRD_PARTY, 'checkpoints', 'contact_graspnet')


def load_grasp_estimator():
    """Load Contact-GraspNet model with pretrained weights."""
    from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
    from contact_graspnet_pytorch import config_utils
    from contact_graspnet_pytorch.checkpoints import CheckpointIO

    global_config = config_utils.load_config(CKPT_DIR, batch_size=1)
    estimator = GraspEstimator(global_config)

    ckpt_io = CheckpointIO(
        checkpoint_dir=os.path.join(CKPT_DIR, 'checkpoints'),
        model=estimator.model,
    )
    ckpt_io.load('model.pt')
    estimator.model.eval()

    return estimator, global_config


def plan_grasps(points, estimator=None, global_config=None, top_k=50):
    """Predict grasp poses from a point cloud.

    Args:
        points: Nx3 float32 array (any coordinate frame)
        estimator: pre-loaded GraspEstimator (or None to load fresh)
        global_config: config dict (or None)
        top_k: return top-k grasps by score

    Returns:
        list of dicts with 'pose' (4x4), 'score' (float), 'contact_pt' (3,)
    """
    if estimator is None:
        estimator, global_config = load_grasp_estimator()

    # predict_scene_grasps expects full scene cloud
    # local_regions=False, filter_grasps=False for single-object cloud
    pred_grasps_cam, scores, contact_pts, gripper_openings = estimator.predict_scene_grasps(
        points.astype(np.float32),
        pc_segments={},
        local_regions=False,
        filter_grasps=False,
        forward_passes=1,
    )

    # Results are keyed by segment id; -1 = full cloud
    key = -1
    if key not in pred_grasps_cam or len(pred_grasps_cam[key]) == 0:
        print("No grasps predicted!")
        return []

    grasps = pred_grasps_cam[key]   # (N, 4, 4)
    confs = scores[key]              # (N,)
    contacts = contact_pts[key]      # (N, 3)

    # Sort by score descending
    order = np.argsort(-confs)
    top_n = min(top_k, len(order))

    results = []
    for i in range(top_n):
        idx = order[i]
        results.append({
            'pose': grasps[idx],         # 4x4 SE3
            'score': float(confs[idx]),
            'contact_pt': contacts[idx],  # 3D contact point
        })

    return results


def visualize_grasps_o3d(points, grasps, top_k=10):
    """Visualize point cloud with grasp poses in Open3D."""
    from contact_graspnet_pytorch.visualization_utils_o3d import visualize_grasps as viz

    poses = {-1: np.array([g['pose'] for g in grasps[:top_k]])}
    scores = {-1: np.array([g['score'] for g in grasps[:top_k]])}

    viz(points, poses, scores, plot_opencv_cam=True)


def main():
    parser = argparse.ArgumentParser(description="Plan grasps on a point cloud")
    parser.add_argument("--points", required=True, help="Path to Nx3 .npy point cloud")
    parser.add_argument("--top-k", type=int, default=50, help="Number of top grasps to return")
    parser.add_argument("--visualize", action="store_true", help="Show grasps in Open3D")
    parser.add_argument("--save", default=None, help="Save grasps to .npz file")
    args = parser.parse_args()

    points = np.load(args.points).astype(np.float32)
    print(f"Loaded point cloud: {points.shape}")

    print("Loading Contact-GraspNet...")
    estimator, config = load_grasp_estimator()

    print("Planning grasps...")
    grasps = plan_grasps(points, estimator, config, top_k=args.top_k)
    print(f"Got {len(grasps)} grasps")

    if grasps:
        print(f"\nTop 5 grasps:")
        for i, g in enumerate(grasps[:5]):
            pos = g['pose'][:3, 3]
            print(f"  {i}: score={g['score']:.4f}, pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    if args.save:
        poses = np.array([g['pose'] for g in grasps])
        scores = np.array([g['score'] for g in grasps])
        contacts = np.array([g['contact_pt'] for g in grasps])
        np.savez(args.save, poses=poses, scores=scores, contacts=contacts)
        print(f"Saved to {args.save}")

    if args.visualize and grasps:
        visualize_grasps_o3d(points, grasps, top_k=10)


if __name__ == "__main__":
    main()
