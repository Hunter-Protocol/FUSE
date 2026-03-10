"""Live ZED capture + PoinTr completion running in parallel threads."""

import sys
import threading
import time
import cv2
import numpy as np
import open3d as o3d
from pipeline import FUSEPipeline
from shape_completer import ShapeCompleter


def lighter_color(rgb, blend=0.5):
    return tuple(c + (1.0 - c) * blend for c in rgb)


def main():
    classes = ["mug", "phone", "cup", "fork", "bottle"]

    # --- Load completer eagerly so both are ready before the loop ---
    print("Loading ShapeCompleter...")
    completer = ShapeCompleter()

    print("Starting FUSE pipeline...")
    pipe = FUSEPipeline(classes)
    pipe.start()
    print(f"Detecting: {classes}")
    print("Press 'q' to quit, 's' to save current clouds.")

    # --- Shared state between threads ---
    lock = threading.Lock()
    latest_partial = {}    # label -> points_3d
    latest_completed = {}  # label -> completed_points_3d
    latest_colors = {}     # label -> color tuple
    completion_pending = {}  # label -> points_3d (queued for completion)
    saved = False

    # --- Completion worker thread ---
    stop_event = threading.Event()

    def completion_worker():
        """Runs PoinTr on queued objects in background."""
        while not stop_event.is_set():
            # Grab pending work
            with lock:
                work = dict(completion_pending)
                completion_pending.clear()

            if not work:
                time.sleep(0.01)
                continue

            for label, points in work.items():
                if not completer.can_complete(label):
                    continue
                t0 = time.time()
                result = completer.complete(points)
                dt = time.time() - t0
                if result is not None:
                    with lock:
                        latest_completed[label] = result
                    print(f"  [completion] {label}: {result.shape} in {dt*1000:.1f}ms")

    worker = threading.Thread(target=completion_worker, daemon=True)
    worker.start()

    # --- Open3D windows ---
    vis_partial = o3d.visualization.Visualizer()
    vis_partial.create_window("Partial (ZED)", width=720, height=480, left=0, top=50)
    pcd_partial = o3d.geometry.PointCloud()

    vis_completed = o3d.visualization.Visualizer()
    vis_completed.create_window("Completed (PoinTr)", width=720, height=480,
                                left=750, top=50)
    pcd_completed = o3d.geometry.PointCloud()

    first_frame = True
    needs_reset_p = True
    needs_reset_c = True
    prev_time = time.time()
    fps = 0.0
    cache_centroids = {}  # label -> last centroid sent to completion
    CACHE_THRESHOLD = 0.03

    try:
        while True:
            bgr, objects, _, _ = pipe.process_frame(skip_scene=True)
            if bgr is None:
                continue

            # --- Queue new completions if centroid moved ---
            for obj in objects:
                if obj.source != "fused" or obj.num_points < 64:
                    continue
                with lock:
                    latest_partial[obj.label] = obj.points_3d
                    latest_colors[obj.label] = obj.color

                need_recompute = True
                if obj.label in cache_centroids:
                    dist = np.linalg.norm(
                        np.array(obj.centroid) - np.array(cache_centroids[obj.label]))
                    if dist < CACHE_THRESHOLD:
                        need_recompute = False

                if need_recompute:
                    with lock:
                        completion_pending[obj.label] = obj.points_3d.copy()
                    cache_centroids[obj.label] = obj.centroid

            # --- Build partial point cloud ---
            p_xyz, p_clr = [], []
            with lock:
                for label, pts in latest_partial.items():
                    color = latest_colors.get(label, (1.0, 1.0, 1.0))
                    p_xyz.append(pts)
                    p_clr.append(np.tile(color, (len(pts), 1)))

            if p_xyz:
                pts_all = np.vstack(p_xyz).astype(np.float64)
                clr_all = np.vstack(p_clr).astype(np.float64)
                pcd_partial.points = o3d.utility.Vector3dVector(
                    np.ascontiguousarray(pts_all))
                pcd_partial.colors = o3d.utility.Vector3dVector(
                    np.ascontiguousarray(clr_all))
            else:
                pcd_partial.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                pcd_partial.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))

            # --- Build completed point cloud (original + completed) ---
            c_xyz, c_clr = [], []
            with lock:
                for label, comp_pts in latest_completed.items():
                    color = latest_colors.get(label, (1.0, 1.0, 1.0))
                    light = lighter_color(color)
                    # Original in solid
                    if label in latest_partial:
                        orig = latest_partial[label]
                        c_xyz.append(orig)
                        c_clr.append(np.tile(color, (len(orig), 1)))
                    # Completed in lighter shade
                    c_xyz.append(comp_pts)
                    c_clr.append(np.tile(light, (len(comp_pts), 1)))

            if c_xyz:
                cpts = np.vstack(c_xyz).astype(np.float64)
                cclr = np.vstack(c_clr).astype(np.float64)
                pcd_completed.points = o3d.utility.Vector3dVector(
                    np.ascontiguousarray(cpts))
                pcd_completed.colors = o3d.utility.Vector3dVector(
                    np.ascontiguousarray(cclr))
            else:
                pcd_completed.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                pcd_completed.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))

            # --- Render ---
            if first_frame:
                vis_partial.add_geometry(pcd_partial)
                vis_partial.get_render_option().point_size = 2.0
                vis_completed.add_geometry(pcd_completed)
                vis_completed.get_render_option().point_size = 2.0
                first_frame = False
            else:
                vis_partial.update_geometry(pcd_partial)
                vis_completed.update_geometry(pcd_completed)

            if needs_reset_p and p_xyz:
                vis_partial.reset_view_point(True)
                needs_reset_p = False
            if needs_reset_c and c_xyz:
                vis_completed.reset_view_point(True)
                needs_reset_c = False

            vis_partial.poll_events()
            vis_partial.update_renderer()
            vis_completed.poll_events()
            vis_completed.update_renderer()

            # FPS
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            # 2D overlay
            for obj in objects:
                x1, y1, x2, y2 = obj.box_2d
                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(bgr, f"{obj.label} {obj.confidence:.2f}",
                            (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1)
                with lock:
                    has_comp = obj.label in latest_completed
                if has_comp:
                    cv2.putText(bgr, "[completed]", (x1, y2 + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 200), 1)

            cv2.putText(bgr, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("FUSE - Live Capture", bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                with lock:
                    for label, pts in latest_partial.items():
                        np.save(f"/home/hunter/Desktop/FUSE/data/{label}_partial.npy", pts)
                        print(f"Saved {label}_partial.npy ({pts.shape})")
                    for label, pts in latest_completed.items():
                        np.save(f"/home/hunter/Desktop/FUSE/data/{label}_completed.npy", pts)
                        print(f"Saved {label}_completed.npy ({pts.shape})")

    finally:
        stop_event.set()
        worker.join(timeout=2)
        pipe.stop()
        vis_partial.destroy_window()
        vis_completed.destroy_window()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
