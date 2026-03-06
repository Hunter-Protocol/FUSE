"""Phase 3 demo: YOLOE segmentation masks → 3D point cloud extraction per object."""

import sys
import cv2
import numpy as np
import open3d as o3d
from camera import ZEDCamera
from detector import Detector

# Colors per class: (R, G, B) normalized for Open3D, (B, G, R) for OpenCV
CLASS_COLORS = {
    "mug":    {"o3d": (1.0, 0.0, 0.0), "cv2": (0, 0, 255)},       # red
    "phone":  {"o3d": (1.0, 1.0, 0.0), "cv2": (0, 255, 255)},     # yellow
    "cup":    {"o3d": (0.0, 1.0, 0.0), "cv2": (0, 255, 0)},       # green
    "fork":   {"o3d": (0.0, 0.0, 1.0), "cv2": (255, 0, 0)},       # blue
    "bottle": {"o3d": (1.0, 0.0, 1.0), "cv2": (255, 0, 255)},     # magenta
}
DEFAULT_COLOR = {"o3d": (1.0, 1.0, 1.0), "cv2": (255, 255, 255)}


def extract_3d_points(mask, point_cloud_data):
    """Extract 3D points where mask is True.

    Args:
        mask: (H, W) bool array
        point_cloud_data: (H, W, 4) XYZRGBA from ZED

    Returns:
        xyz: (N, 3) float32 array of valid 3D points
    """
    xyz = point_cloud_data[:, :, :3]
    masked_xyz = xyz[mask]
    valid = np.isfinite(masked_xyz).all(axis=1)
    return masked_xyz[valid].astype(np.float32)


def draw_detections(frame, detections):
    """Draw bounding boxes, labels, and semi-transparent masks on frame."""
    overlay = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box_2d"]
        label = det["label"]
        conf = det["confidence"]
        color = CLASS_COLORS.get(label, DEFAULT_COLOR)["cv2"]
        mask = det["mask"]

        # Draw filled mask
        overlay[mask] = (
            overlay[mask] * 0.5 + np.array(color) * 0.5
        ).astype(np.uint8)

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    return frame


def main():
    svo_path = sys.argv[1] if len(sys.argv) > 1 else None
    classes = ["mug", "phone", "cup", "fork", "bottle"]

    print("Loading YOLOE segmentation model...")
    detector = Detector(model_size="11s", confidence=0.3)
    detector.set_classes(classes)
    print(f"Detecting: {classes}")

    with ZEDCamera(svo_path=svo_path) as cam:
        # Set up Open3D visualizer for detected objects
        vis_obj = o3d.visualization.Visualizer()
        vis_obj.create_window("FUSE - 3D Objects", width=720, height=480, left=750, top=50)
        pcd_obj = o3d.geometry.PointCloud()

        # Set up Open3D visualizer for full scene
        vis_scene = o3d.visualization.Visualizer()
        vis_scene.create_window("FUSE - Full Point Cloud", width=720, height=480, left=750, top=580)
        pcd_scene = o3d.geometry.PointCloud()

        first_frame = True

        print("Press 'q' in the RGB window to quit.")

        while True:
            if not cam.grab():
                if svo_path:
                    print("End of SVO file.")
                    break
                continue

            bgr = cam.get_bgr()
            detections = detector.detect(bgr)

            # Get raw point cloud (H, W, 4) for mask-based extraction
            import pyzed.sl as sl
            pc_mat = sl.Mat()
            cam.zed.retrieve_measure(pc_mat, sl.MEASURE.XYZRGBA)
            pc_data = pc_mat.get_data()

            # Extract 3D points per detected object, colored by class
            all_xyz = []
            all_colors = []
            for det in detections:
                xyz = extract_3d_points(det["mask"], pc_data)
                if len(xyz) == 0:
                    continue
                color = CLASS_COLORS.get(det["label"], DEFAULT_COLOR)["o3d"]
                colors = np.tile(color, (len(xyz), 1))

                centroid = xyz.mean(axis=0)
                det["centroid"] = tuple(centroid)
                det["num_points"] = len(xyz)

                all_xyz.append(xyz)
                all_colors.append(colors)

            # Update detected objects 3D visualization
            if all_xyz:
                xyz_combined = np.vstack(all_xyz)
                colors_combined = np.vstack(all_colors)
                pcd_obj.points = o3d.utility.Vector3dVector(xyz_combined)
                pcd_obj.colors = o3d.utility.Vector3dVector(colors_combined)
            else:
                pcd_obj.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                pcd_obj.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))

            # Update full scene point cloud
            scene_xyz, scene_rgb = cam.get_point_cloud()
            if len(scene_xyz) > 200_000:
                idx = np.random.choice(len(scene_xyz), 200_000, replace=False)
                scene_xyz, scene_rgb = scene_xyz[idx], scene_rgb[idx]
            pcd_scene.points = o3d.utility.Vector3dVector(scene_xyz)
            pcd_scene.colors = o3d.utility.Vector3dVector(scene_rgb)

            if first_frame:
                vis_obj.add_geometry(pcd_obj)
                vis_obj.get_render_option().point_size = 2.0
                vis_scene.add_geometry(pcd_scene)
                vis_scene.get_render_option().point_size = 2.0
                first_frame = False
            else:
                vis_obj.update_geometry(pcd_obj)
                vis_scene.update_geometry(pcd_scene)

            vis_obj.poll_events()
            vis_obj.update_renderer()
            vis_scene.poll_events()
            vis_scene.update_renderer()

            # Draw 2D overlay with masks + centroids
            frame = draw_detections(bgr, detections)
            for det in detections:
                if "centroid" in det:
                    cx, cy, cz = det["centroid"]
                    text = f"({cx:.2f}, {cy:.2f}, {cz:.2f})m  [{det['num_points']}pts]"
                    x1, _, _, y2 = det["box_2d"]
                    cv2.putText(frame, text, (x1, y2 + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.imshow("FUSE - Detection + Segmentation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vis_obj.destroy_window()
        vis_scene.destroy_window()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
