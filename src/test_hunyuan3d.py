"""Test: capture mug from ZED, run Hunyuan3D-2 shape generation, visualize mesh + alignment."""

import sys
import time
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, "/home/hunter/Desktop/Hunyuan3D-2")

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

    # Build RGBA (Hunyuan3D expects RGBA with alpha channel for background removal)
    rgba = np.dstack([square_rgb, square_mask])

    # Resize to 512x512
    rgba = cv2.resize(rgba, (512, 512), interpolation=cv2.INTER_LANCZOS4)

    pil_image = Image.fromarray(rgba, "RGBA")
    return pil_image, obj.points_3d, obj


def align_to_partial(mesh_pts, partial_pts):
    """Align Hunyuan3D canonical-frame points to ZED camera-frame partial cloud."""
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
    import torch
    import open3d as o3d
    import trimesh

    data_dir = "/home/hunter/Desktop/FUSE/data"

    # Step 1: Capture mug then release ZED to free VRAM
    try:
        image, partial_pts, obj = capture_mug_crop()
        if image is None:
            return
        image.save(f"{data_dir}/mug_crop_hunyuan.png")
        np.save(f"{data_dir}/mug_partial.npy", partial_pts)
        print(f"Saved crop: {image.size}, mode: {image.mode}")
    except RuntimeError as e:
        print(f"Camera unavailable ({e}), using saved data...")
        image = Image.open(f"{data_dir}/mug_crop_hunyuan.png").convert("RGBA")
        partial_pts = np.load(f"{data_dir}/mug_partial.npy")
        print(f"Loaded crop: {image.size}, partial: {partial_pts.shape}")

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Step 2: Load Hunyuan3D shape generation pipeline (no texture needed)
    # Use mini variant (0.6B params) for faster inference, or full model
    use_mini = '--mini' in sys.argv
    use_turbo = '--turbo' in sys.argv
    if use_mini or use_turbo:
        model_id = 'tencent/Hunyuan3D-2mini'
        if use_turbo:
            subfolder = 'hunyuan3d-dit-v2-mini-turbo'
        else:
            subfolder = 'hunyuan3d-dit-v2-mini'
    else:
        model_id = 'tencent/Hunyuan3D-2'
        subfolder = 'hunyuan3d-dit-v2-0'
    print(f"\nLoading {model_id} ({subfolder})...")
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    t0 = time.time()
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_id, subfolder=subfolder)
    print(f"Model loaded: {time.time() - t0:.1f}s")

    # Step 3: Generate 3D shape with progress tracking
    # Turbo uses consistency distillation — needs far fewer steps (4-8 vs 50)
    num_steps = 8 if use_turbo else 50
    gen_start = time.time()

    def progress_callback(step, timestep, outputs):
        elapsed = time.time() - gen_start
        pct = (step + 1) / num_steps * 100
        eta = elapsed / (step + 1) * (num_steps - step - 1) if step > 0 else 0
        print(f"  Step {step + 1}/{num_steps} ({pct:.0f}%) | {elapsed:.1f}s elapsed | ETA {eta:.1f}s")

    print(f"Running shape generation ({num_steps} diffusion steps)...")
    mesh = pipeline(
        image=image,
        num_inference_steps=num_steps,
        callback=progress_callback,
        callback_steps=1,
    )[0]
    dt = time.time() - gen_start
    print(f"Shape generation: {dt:.2f}s")
    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    suffix = "_turbo" if use_turbo else ("_mini" if use_mini else "")
    mesh.export(f"{data_dir}/mug_hunyuan3d{suffix}.glb")

    # Step 4: Sample points from mesh
    sampled_pts, _ = trimesh.sample.sample_surface(mesh, count=8192)
    sampled_pts = sampled_pts.astype(np.float32)
    print(f"Sampled {len(sampled_pts)} points")

    # Free model from GPU
    del pipeline
    torch.cuda.empty_cache()

    # Step 5: Align to ZED partial cloud
    print("\nAligning Hunyuan3D -> camera frame...")
    aligned_pts = align_to_partial(sampled_pts, partial_pts)
    np.save(f"{data_dir}/mug_hunyuan3d_aligned.npy", aligned_pts)

    # Step 6: Visualize — 3 windows
    print("\nVisualizing...")

    # Window 1: Input image (OpenCV)
    input_rgba = np.array(image)
    # Composite onto white for display
    alpha = input_rgba[:, :, 3:4].astype(np.float32) / 255.0
    display_rgb = (input_rgba[:, :, :3].astype(np.float32) * alpha +
                   255.0 * (1.0 - alpha)).astype(np.uint8)
    display_bgr = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hunyuan3D Input Image", display_bgr)

    # Window 2: Raw mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
        colors = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float64) / 255.0
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d_mesh.compute_vertex_normals()

    vis_mesh = o3d.visualization.Visualizer()
    variant = "Mini" if use_mini else "Full"
    vis_mesh.create_window(f"Hunyuan3D {variant} Mesh (canonical)", width=600, height=600, left=450)
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
    vis_pc.create_window("Aligned: RED=ZED, CYAN=Hunyuan3D", width=600, height=600, left=1100)
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
