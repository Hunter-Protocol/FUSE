"""Step 0: Save Hunyuan3D mesh (vertices+faces) to disk as OBJ.

Calls the existing Modal endpoint and saves the full mesh, not just sampled points.
Also applies alignment transform to produce a camera-frame mesh.

Usage:
    modal run src/sim/save_mesh.py                    # use saved crop
    modal run src/sim/save_mesh.py --image path.png   # custom image
"""

import io
import sys
import time
import numpy as np
import modal
import trimesh


DATA_DIR = "/home/hunter/Desktop/FUSE/data"
MESH_DIR = f"{DATA_DIR}/meshes"


def align_mesh_to_partial(mesh_verts, partial_pts):
    """Align Hunyuan3D canonical-frame mesh vertices to ZED camera frame.

    Reuses the same RANSAC+ICP approach from test_hunyuan3d_cloud.py,
    but operates on mesh vertices instead of sampled points.
    Returns aligned vertices and the transformation parameters.
    """
    import open3d as o3d

    # Center both
    mesh_center = mesh_verts.mean(axis=0)
    partial_center = partial_pts.mean(axis=0)
    mesh_centered = mesh_verts - mesh_center
    partial_centered = partial_pts - partial_center

    # Scale: match bbox diagonal
    mesh_extent = np.linalg.norm(mesh_centered.max(axis=0) - mesh_centered.min(axis=0))
    partial_extent = np.linalg.norm(partial_centered.max(axis=0) - partial_centered.min(axis=0))
    scale = partial_extent / max(mesh_extent, 1e-6)
    mesh_scaled = mesh_centered * scale

    # Sample points from mesh for registration (use subset for speed)
    rng = np.random.default_rng(42)
    n_sample = min(len(mesh_scaled), 4096)
    sample_idx = rng.choice(len(mesh_scaled), n_sample, replace=False)
    mesh_sample = mesh_scaled[sample_idx]

    # Build point clouds
    pcd_mesh = o3d.geometry.PointCloud()
    pcd_mesh.points = o3d.utility.Vector3dVector(mesh_sample.astype(np.float64))
    pcd_partial = o3d.geometry.PointCloud()
    pcd_partial.points = o3d.utility.Vector3dVector(partial_centered.astype(np.float64))

    # Downsample
    voxel_size = partial_extent * 0.02
    pcd_mesh_ds = pcd_mesh.voxel_down_sample(voxel_size)
    pcd_partial_ds = pcd_partial.voxel_down_sample(voxel_size)

    # Normals + FPFH
    radius_normal = voxel_size * 3
    pcd_mesh_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd_partial_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 7
    fpfh_mesh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_mesh_ds, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    fpfh_partial = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_partial_ds, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    # RANSAC
    dist_thresh = voxel_size * 2.0
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_mesh_ds, pcd_partial_ds, fpfh_mesh, fpfh_partial,
        mutual_filter=True, max_correspondence_distance=dist_thresh,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    print(f"  RANSAC fitness: {result_ransac.fitness:.3f}")

    # ICP refinement
    icp_dist = voxel_size * 1.0
    result_icp = o3d.pipelines.registration.registration_icp(
        pcd_mesh_ds, pcd_partial_ds, max_correspondence_distance=icp_dist,
        init=result_ransac.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )
    print(f"  ICP fitness: {result_icp.fitness:.3f}, RMSE: {result_icp.inlier_rmse:.4f}m")

    # Apply transform to ALL mesh vertices
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(mesh_scaled.astype(np.float64))
    pcd_all.transform(result_icp.transformation)
    aligned_verts = np.asarray(pcd_all.points).astype(np.float32) + partial_center

    return aligned_verts, {
        "scale": float(scale),
        "mesh_center": mesh_center.tolist(),
        "partial_center": partial_center.tolist(),
        "icp_transform": result_icp.transformation.tolist(),
        "icp_fitness": float(result_icp.fitness),
        "icp_rmse": float(result_icp.inlier_rmse),
    }


def save_mesh_from_result(result, object_name, partial_pts=None):
    """Save mesh from Modal endpoint result as OBJ files."""
    vertices = np.array(result['vertices'], dtype=np.float32)
    faces = np.array(result['faces'], dtype=np.int32)
    print(f"Mesh: {len(vertices)} verts, {len(faces)} faces")

    # Save canonical-frame mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    canonical_path = f"{MESH_DIR}/{object_name}_complete_canonical.obj"
    mesh.export(canonical_path)
    print(f"Saved canonical mesh: {canonical_path}")

    # Save raw numpy
    np.savez(f"{DATA_DIR}/{object_name}_hunyuan3d_mesh.npz",
             vertices=vertices, faces=faces)

    # Align to camera frame if partial cloud available
    if partial_pts is not None:
        print("Aligning mesh to camera frame...")
        aligned_verts, align_info = align_mesh_to_partial(vertices, partial_pts)
        aligned_mesh = trimesh.Trimesh(vertices=aligned_verts, faces=faces)
        aligned_path = f"{MESH_DIR}/{object_name}_complete.obj"
        aligned_mesh.export(aligned_path)
        print(f"Saved aligned mesh: {aligned_path}")
        return aligned_path, align_info

    return canonical_path, None


def main():
    from pathlib import Path
    from PIL import Image

    # Use saved crop by default
    image_path = f"{DATA_DIR}/mug_crop_hunyuan3d_cloud.png"
    partial_path = f"{DATA_DIR}/mug_partial.npy"
    object_name = "mug"

    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGBA")
    partial_pts = np.load(partial_path)
    print(f"Partial cloud: {partial_pts.shape}")

    # Call Modal endpoint
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    print("\nCalling Modal Hunyuan3D endpoint...")
    Hunyuan3DModel = modal.Cls.from_name("fuse-hunyuan3d", "Hunyuan3DModel")
    model = Hunyuan3DModel()

    t0 = time.time()
    result = model.generate.remote(image_bytes)
    dt = time.time() - t0
    print(f"Generation: {result['generation_time']:.2f}s (total latency: {dt:.2f}s)")

    # Save mesh
    mesh_path, align_info = save_mesh_from_result(result, object_name, partial_pts)
    print(f"\nDone. Mesh saved to: {mesh_path}")
    if align_info:
        print(f"  Alignment: fitness={align_info['icp_fitness']:.3f}, RMSE={align_info['icp_rmse']:.4f}m")


if __name__ == "__main__":
    main()
