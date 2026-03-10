"""Quick test: capture mug from ZED, run TripoSR, clean up + align, visualize."""

import sys
import time
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, "/home/hunter/Desktop/TripoSR")

from pipeline import FUSEPipeline


def capture_mug_crop():
    """Grab a frame from ZED, detect mug, return cropped RGB image + partial cloud."""
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

    # Create mask from YOLOE segmentation, erode slightly to remove edge bleed
    mask_crop = obj.mask[y1:y2, x1:x2].astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_crop = cv2.erode(mask_crop, kernel, iterations=1)

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    # Make square, center object
    side = max(crop_rgb.shape[0], crop_rgb.shape[1])
    square_rgb = np.full((side, side, 3), 255, dtype=np.uint8)  # white bg
    square_mask = np.zeros((side, side), dtype=np.uint8)
    dy = (side - crop_rgb.shape[0]) // 2
    dx = (side - crop_rgb.shape[1]) // 2
    square_rgb[dy:dy + crop_rgb.shape[0], dx:dx + crop_rgb.shape[1]] = crop_rgb
    square_mask[dy:dy + mask_crop.shape[0], dx:dx + mask_crop.shape[1]] = mask_crop

    # Composite onto white background using mask
    alpha = square_mask[:, :, np.newaxis].astype(np.float32) / 255.0
    rgb_f = square_rgb.astype(np.float32)
    white_bg = np.full_like(rgb_f, 255.0)
    composited = (rgb_f * alpha + white_bg * (1.0 - alpha)).astype(np.uint8)

    # Resize to 512x512 for TripoSR (it was trained on this resolution)
    composited = cv2.resize(composited, (512, 512), interpolation=cv2.INTER_LANCZOS4)

    pil_image = Image.fromarray(composited, "RGB")
    return pil_image, obj.points_3d, obj


def clean_mesh(mesh):
    """Remove small disconnected components, keeping only the largest."""
    import trimesh
    components = mesh.split(only_watertight=False)
    if not components:
        return mesh
    largest = max(components, key=lambda c: len(c.faces))
    print(f"  Mesh cleanup: {len(components)} components -> kept largest "
          f"({len(largest.faces)} faces, removed {len(mesh.faces) - len(largest.faces)})")
    return largest


def align_to_partial(triposr_pts, partial_pts):
    """Align TripoSR canonical-frame points to ZED camera-frame partial cloud.

    Steps:
    1. Center both clouds at origin
    2. Scale TripoSR to match partial cloud's extent
    3. Downsample both for speed
    4. Compute FPFH features on both
    5. RANSAC global registration to find coarse alignment
    6. Refine with Point-to-Point ICP
    """
    import open3d as o3d

    # Center both
    tripo_center = triposr_pts.mean(axis=0)
    partial_center = partial_pts.mean(axis=0)
    tripo_centered = triposr_pts - tripo_center
    partial_centered = partial_pts - partial_center

    # Scale: match the bounding box diagonal
    tripo_extent = np.linalg.norm(tripo_centered.max(axis=0) - tripo_centered.min(axis=0))
    partial_extent = np.linalg.norm(partial_centered.max(axis=0) - partial_centered.min(axis=0))
    scale = partial_extent / max(tripo_extent, 1e-6)
    tripo_scaled = tripo_centered * scale

    print(f"  Scale factor: {scale:.4f}")
    print(f"  TripoSR bbox after scale: {tripo_scaled.min(axis=0)} -> {tripo_scaled.max(axis=0)}")
    print(f"  Partial bbox (centered):  {partial_centered.min(axis=0)} -> {partial_centered.max(axis=0)}")

    # Build point clouds (both centered at origin for registration)
    pcd_tripo = o3d.geometry.PointCloud()
    pcd_tripo.points = o3d.utility.Vector3dVector(tripo_scaled.astype(np.float64))

    pcd_partial = o3d.geometry.PointCloud()
    pcd_partial.points = o3d.utility.Vector3dVector(partial_centered.astype(np.float64))

    # Downsample for registration speed
    voxel_size = partial_extent * 0.02  # ~2% of object size
    print(f"  Voxel size: {voxel_size:.4f}m")
    pcd_tripo_ds = pcd_tripo.voxel_down_sample(voxel_size)
    pcd_partial_ds = pcd_partial.voxel_down_sample(voxel_size)
    print(f"  Downsampled: TripoSR {len(pcd_tripo_ds.points)} pts, Partial {len(pcd_partial_ds.points)} pts")

    # Estimate normals
    radius_normal = voxel_size * 3
    pcd_tripo_ds.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd_partial_ds.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # Compute FPFH features
    radius_feature = voxel_size * 7
    fpfh_tripo = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_tripo_ds,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    fpfh_partial = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_partial_ds,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    # RANSAC global registration
    distance_threshold = voxel_size * 2.0
    print(f"  RANSAC distance threshold: {distance_threshold:.4f}m")
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_tripo_ds, pcd_partial_ds,
        fpfh_tripo, fpfh_partial,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    print(f"  RANSAC fitness: {result_ransac.fitness:.3f}, RMSE: {result_ransac.inlier_rmse:.4f}m")
    print(f"  RANSAC correspondences: {len(result_ransac.correspondence_set)}")

    # Refine with ICP (Point-to-Point)
    icp_dist = voxel_size * 1.0
    result_icp = o3d.pipelines.registration.registration_icp(
        pcd_tripo_ds, pcd_partial_ds,
        max_correspondence_distance=icp_dist,
        init=result_ransac.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )
    print(f"  ICP fitness: {result_icp.fitness:.3f}, RMSE: {result_icp.inlier_rmse:.4f}m")

    # Apply final transform to ALL TripoSR points, then shift back to camera frame
    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(tripo_scaled.astype(np.float64))
    pcd_full.transform(result_icp.transformation)
    aligned = np.asarray(pcd_full.points).astype(np.float32) + partial_center

    return aligned


def main():
    import open3d as o3d
    import trimesh

    data_dir = "/home/hunter/Desktop/FUSE/data"

    # Step 1: Capture mug then release ZED to free VRAM
    try:
        image, partial_pts, obj = capture_mug_crop()
        if image is None:
            return
        image.save(f"{data_dir}/mug_crop.png")
        np.save(f"{data_dir}/mug_partial.npy", partial_pts)
        print(f"Saved crop: {image.size}")
    except RuntimeError as e:
        print(f"Camera unavailable ({e}), using saved data...")
        image = Image.open(f"{data_dir}/mug_crop.png").convert("RGB")
        partial_pts = np.load(f"{data_dir}/mug_partial.npy")
        print(f"Loaded crop: {image.size}, partial: {partial_pts.shape}")

    torch.cuda.empty_cache()

    # Step 2: Load TripoSR
    print("\nLoading TripoSR...")
    from tsr.system import TSR
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.to("cuda")
    model.renderer.set_chunk_size(4096)

    # Step 3: Inference
    print("Running TripoSR inference...")
    t0 = time.time()
    with torch.no_grad():
        scene_codes = model([image], device="cuda")
    dt_infer = time.time() - t0
    print(f"Inference: {dt_infer:.2f}s")

    # Step 4: Extract mesh
    t0 = time.time()
    meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=192)
    dt_mesh = time.time() - t0
    print(f"Mesh extraction: {dt_mesh:.2f}s")

    mesh = meshes[0]
    print(f"Raw mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    # Step 5: Clean mesh — remove disconnected components
    print("\nCleaning mesh...")
    mesh = clean_mesh(mesh)
    mesh.export(f"{data_dir}/mug_triposr.obj")

    # Step 6: Sample points from cleaned mesh
    sampled_pts, _ = trimesh.sample.sample_surface(mesh, count=8192)
    sampled_pts = sampled_pts.astype(np.float32)
    print(f"Sampled {len(sampled_pts)} points from cleaned mesh")

    # Free TripoSR from GPU
    del model, scene_codes, meshes
    torch.cuda.empty_cache()

    # Step 7: Align TripoSR to ZED partial cloud
    print("\nAligning TripoSR -> camera frame...")
    aligned_pts = align_to_partial(sampled_pts, partial_pts)
    np.save(f"{data_dir}/mug_triposr_aligned.npy", aligned_pts)

    # Step 8: Visualize — 3 windows
    # Window 1: Input image (OpenCV)
    print("\nVisualizing...")
    input_rgb = np.array(image)
    input_bgr = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR)
    # Scale up for visibility
    display_size = 400
    h, w = input_bgr.shape[:2]
    scale_factor = display_size / max(h, w)
    display_img = cv2.resize(input_bgr, (int(w * scale_factor), int(h * scale_factor)),
                             interpolation=cv2.INTER_LANCZOS4)
    cv2.imshow("TripoSR Input Image", display_img)

    # Window 2: Raw TripoSR mesh (before converting to point cloud)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices.astype(np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
        colors = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float64) / 255.0
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d_mesh.compute_vertex_normals()

    vis_mesh = o3d.visualization.Visualizer()
    vis_mesh.create_window("TripoSR 3D Mesh (canonical)", width=600, height=600, left=450)
    vis_mesh.add_geometry(o3d_mesh)
    vis_mesh.get_render_option().mesh_show_back_face = True

    # Window 3: Aligned point clouds
    pcd_partial = o3d.geometry.PointCloud()
    pcd_partial.points = o3d.utility.Vector3dVector(partial_pts.astype(np.float64))
    pcd_partial.paint_uniform_color([1.0, 0.0, 0.0])  # red = ZED partial

    pcd_aligned = o3d.geometry.PointCloud()
    pcd_aligned.points = o3d.utility.Vector3dVector(aligned_pts.astype(np.float64))
    pcd_aligned.paint_uniform_color([0.0, 0.7, 1.0])  # cyan = TripoSR aligned

    vis_pc = o3d.visualization.Visualizer()
    vis_pc.create_window("Aligned: RED=ZED, CYAN=TripoSR", width=600, height=600, left=1100)
    vis_pc.add_geometry(pcd_partial)
    vis_pc.add_geometry(pcd_aligned)

    print("Window 1: Input image | Window 2: TripoSR mesh | Window 3: Aligned point clouds")
    print("Press 'q' in OpenCV window or close Open3D windows to exit")

    while True:
        vis_mesh.poll_events()
        vis_mesh.update_renderer()
        vis_pc.poll_events()
        vis_pc.update_renderer()

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        # Check if Open3D windows were closed
        try:
            if not vis_mesh.poll_events() or not vis_pc.poll_events():
                break
        except Exception:
            break

    vis_mesh.destroy_window()
    vis_pc.destroy_window()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import torch
    main()
