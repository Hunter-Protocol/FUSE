"""Test: capture mug from ZED, send to cloud TRELLIS, align + visualize."""

import sys
import time
import io
import numpy as np
import cv2
from PIL import Image

from pipeline import FUSEPipeline


def capture_mug_crop():
    """Grab a frame from ZED, detect mug, return cropped RGBA image + partial cloud."""
    print("Capturing from ZED...")
    with FUSEPipeline(["mug"]) as pipe:
        for _ in range(10):
            pipe.process_frame(skip_scene=True)
        bgr, objects, _, _ = pipe.process_frame(skip_scene=True)

    if not objects:
        print("No mug detected!")
        return None, None, None

    obj = objects[0]
    print(f"Detected: {obj}")

    # Crop BGR using bounding box with padding
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = obj.box_2d
    pad = int(max(x2 - x1, y2 - y1) * 0.15)
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    crop_bgr = bgr[y1:y2, x1:x2]

    # Use YOLOE mask, erode to remove edge bleed
    mask_crop = obj.mask[y1:y2, x1:x2].astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_crop = cv2.erode(mask_crop, kernel, iterations=1)

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    # Make square, center object
    side = max(crop_rgb.shape[0], crop_rgb.shape[1])
    square_rgb = np.zeros((side, side, 3), dtype=np.uint8)
    square_mask = np.zeros((side, side), dtype=np.uint8)
    dy = (side - crop_rgb.shape[0]) // 2
    dx = (side - crop_rgb.shape[1]) // 2
    square_rgb[dy:dy + crop_rgb.shape[0], dx:dx + crop_rgb.shape[1]] = crop_rgb
    square_mask[dy:dy + mask_crop.shape[0], dx:dx + mask_crop.shape[1]] = mask_crop

    # Build RGBA
    rgba = np.dstack([square_rgb, square_mask])
    rgba = cv2.resize(rgba, (512, 512), interpolation=cv2.INTER_LANCZOS4)

    pil_image = Image.fromarray(rgba, "RGBA")
    return pil_image, obj.points_3d, obj


def align_to_partial(mesh_pts, partial_pts):
    """Align TRELLIS canonical-frame points to ZED camera-frame partial cloud."""
    import open3d as o3d

    # Center both
    mesh_center = mesh_pts.mean(axis=0)
    partial_center = partial_pts.mean(axis=0)
    mesh_centered = mesh_pts - mesh_center
    partial_centered = partial_pts - partial_center

    # Scale: match bbox diagonal
    mesh_extent = np.linalg.norm(mesh_centered.max(axis=0) - mesh_centered.min(axis=0))
    partial_extent = np.linalg.norm(partial_centered.max(axis=0) - partial_centered.min(axis=0))
    scale = partial_extent / max(mesh_extent, 1e-6)
    mesh_scaled = mesh_centered * scale

    print(f"  Scale factor: {scale:.4f}")

    # Build point clouds (both centered at origin)
    pcd_mesh = o3d.geometry.PointCloud()
    pcd_mesh.points = o3d.utility.Vector3dVector(mesh_scaled.astype(np.float64))
    pcd_partial = o3d.geometry.PointCloud()
    pcd_partial.points = o3d.utility.Vector3dVector(partial_centered.astype(np.float64))

    # Downsample
    voxel_size = partial_extent * 0.02
    pcd_mesh_ds = pcd_mesh.voxel_down_sample(voxel_size)
    pcd_partial_ds = pcd_partial.voxel_down_sample(voxel_size)
    print(f"  Downsampled: Mesh {len(pcd_mesh_ds.points)} pts, Partial {len(pcd_partial_ds.points)} pts")

    # Normals + FPFH
    radius_normal = voxel_size * 3
    pcd_mesh_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd_partial_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 7
    fpfh_mesh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_mesh_ds, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    fpfh_partial = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_partial_ds, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    # RANSAC global registration
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
    print(f"  RANSAC fitness: {result_ransac.fitness:.3f}, correspondences: {len(result_ransac.correspondence_set)}")

    # ICP refinement
    icp_dist = voxel_size * 1.0
    result_icp = o3d.pipelines.registration.registration_icp(
        pcd_mesh_ds, pcd_partial_ds, max_correspondence_distance=icp_dist,
        init=result_ransac.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )
    print(f"  ICP fitness: {result_icp.fitness:.3f}, RMSE: {result_icp.inlier_rmse:.4f}m")

    # Apply to full point set, shift back to camera frame
    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(mesh_scaled.astype(np.float64))
    pcd_full.transform(result_icp.transformation)
    return np.asarray(pcd_full.points).astype(np.float32) + partial_center


def main():
    import modal
    import open3d as o3d

    data_dir = "/home/hunter/Desktop/FUSE/data"

    # Step 1: Capture mug from ZED (or use saved data)
    try:
        image, partial_pts, obj = capture_mug_crop()
        if image is None:
            return
        image.save(f"{data_dir}/mug_crop_trellis.png")
        np.save(f"{data_dir}/mug_partial.npy", partial_pts)
        print(f"Saved crop: {image.size}, mode: {image.mode}")
    except RuntimeError as e:
        print(f"Camera unavailable ({e}), using saved data...")
        image = Image.open(f"{data_dir}/mug_crop_hunyuan.png").convert("RGBA")
        partial_pts = np.load(f"{data_dir}/mug_partial.npy")
        print(f"Loaded crop: {image.size}, partial: {partial_pts.shape}")

    # Step 2: Send to cloud TRELLIS via Modal
    print("\nSending to cloud TRELLIS (A100 80GB)...")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    # Look up the deployed Modal class
    TrellisModel = modal.Cls.from_name("fuse-trellis", "TrellisModel")
    model = TrellisModel()

    t_start = time.time()
    result = model.generate.remote(image_bytes)
    t_total = time.time() - t_start

    print(f"\nCloud results:")
    print(f"  End-to-end latency: {t_total:.2f}s (upload + inference + download)")
    print(f"  GPU generation time: {result['generation_time']:.2f}s")
    print(f"  Network overhead: {t_total - result['generation_time']:.2f}s")
    print(f"  Mesh: {result['num_vertices']} verts, {result['num_faces']} faces")
    print(f"  Sampled points: {len(result['points'])}")

    # Step 3: Convert results
    trellis_pts = np.array(result['points'], dtype=np.float32)
    np.save(f"{data_dir}/mug_trellis_cloud.npy", trellis_pts)

    # Step 4: Align to ZED partial cloud
    print("\nAligning TRELLIS -> camera frame...")
    aligned_pts = align_to_partial(trellis_pts, partial_pts)
    np.save(f"{data_dir}/mug_trellis_aligned.npy", aligned_pts)

    # Step 5: Visualize — 3 windows
    print("\nVisualizing...")

    # Window 1: Input image (OpenCV)
    input_rgba = np.array(image)
    alpha = input_rgba[:, :, 3:4].astype(np.float32) / 255.0
    display_rgb = (input_rgba[:, :, :3].astype(np.float32) * alpha +
                   255.0 * (1.0 - alpha)).astype(np.uint8)
    display_bgr = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("TRELLIS Input Image", display_bgr)

    # Window 2: Raw TRELLIS mesh (canonical frame)
    vertices = np.array(result['vertices'], dtype=np.float64)
    faces = np.array(result['faces'], dtype=np.int32)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.compute_vertex_normals()

    vis_mesh = o3d.visualization.Visualizer()
    vis_mesh.create_window("TRELLIS Mesh (canonical)", width=600, height=600, left=450)
    vis_mesh.add_geometry(o3d_mesh)
    vis_mesh.get_render_option().mesh_show_back_face = True

    # Window 3: Aligned point clouds
    pcd_partial = o3d.geometry.PointCloud()
    pcd_partial.points = o3d.utility.Vector3dVector(partial_pts.astype(np.float64))
    pcd_partial.paint_uniform_color([1.0, 0.0, 0.0])

    pcd_aligned = o3d.geometry.PointCloud()
    pcd_aligned.points = o3d.utility.Vector3dVector(aligned_pts.astype(np.float64))
    pcd_aligned.paint_uniform_color([0.0, 0.7, 1.0])

    vis_pc = o3d.visualization.Visualizer()
    vis_pc.create_window("Aligned: RED=ZED, CYAN=TRELLIS", width=600, height=600, left=1100)
    vis_pc.add_geometry(pcd_partial)
    vis_pc.add_geometry(pcd_aligned)

    print("Window 1: Input | Window 2: Mesh | Window 3: Aligned clouds")
    print("Press 'q' to exit")

    while True:
        vis_mesh.poll_events()
        vis_mesh.update_renderer()
        vis_pc.poll_events()
        vis_pc.update_renderer()
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

    vis_mesh.destroy_window()
    vis_pc.destroy_window()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
