"""Step 1: Build a mesh from ZED partial point cloud using Ball Pivoting.

This creates the "no completion" baseline mesh for grasp comparison.
The result is an open surface mesh — only what the depth sensor observed.

Usage:
    python -m sim.build_partial_mesh
    python -m sim.build_partial_mesh --input data/mug_partial.npy --name mug
"""

import argparse
import numpy as np
import open3d as o3d


DATA_DIR = "/home/hunter/Desktop/FUSE/data"
MESH_DIR = f"{DATA_DIR}/meshes"


def build_partial_mesh(points, object_name, visualize=True):
    """Build an open surface mesh from partial point cloud via Ball Pivoting.

    Args:
        points: Nx3 float32 array in camera frame
        object_name: name for output file
        visualize: show result in Open3D window

    Returns:
        path to saved OBJ file
    """
    print(f"Building partial mesh from {len(points)} points...")

    # Create point cloud and estimate normals
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    # Orient normals toward camera (assumed at origin)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))

    # Compute point spacing for Ball Pivoting radii
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    print(f"  Average point spacing: {avg_dist:.4f}m")

    # Ball Pivoting with multiple radii
    radii = [avg_dist * 1.5, avg_dist * 3.0, avg_dist * 6.0]
    print(f"  Ball Pivoting radii: {[f'{r:.4f}' for r in radii]}")

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )

    n_verts = len(mesh.vertices)
    n_tris = len(mesh.triangles)
    print(f"  Result: {n_verts} vertices, {n_tris} triangles")

    if n_tris == 0:
        print("  Ball Pivoting failed, falling back to Alpha Shapes...")
        alpha = avg_dist * 10
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        n_verts = len(mesh.vertices)
        n_tris = len(mesh.triangles)
        print(f"  Alpha Shape result: {n_verts} vertices, {n_tris} triangles")

    if n_tris == 0:
        print("  ERROR: Could not build mesh from partial points")
        return None

    # Clean up
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    # Save
    out_path = f"{MESH_DIR}/{object_name}_partial.obj"
    o3d.io.write_triangle_mesh(out_path, mesh)
    print(f"  Saved: {out_path}")

    if visualize:
        print("  Showing partial mesh (close window to continue)...")
        mesh.paint_uniform_color([1.0, 0.3, 0.3])
        o3d.visualization.draw_geometries(
            [mesh],
            window_name=f"Partial Mesh: {object_name}",
            width=800, height=600,
        )

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build partial mesh from ZED points")
    parser.add_argument("--input", default=f"{DATA_DIR}/mug_partial.npy",
                        help="Path to partial point cloud .npy file")
    parser.add_argument("--name", default="mug", help="Object name for output file")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    points = np.load(args.input)
    print(f"Loaded {args.input}: {points.shape}")

    path = build_partial_mesh(points, args.name, visualize=not args.no_viz)
    if path:
        print(f"\nDone. Partial mesh: {path}")


if __name__ == "__main__":
    main()
