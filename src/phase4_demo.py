"""Phase 4 demo: Full FUSE pipeline with FusedObject output."""

import sys
import cv2
import numpy as np
import open3d as o3d
from pipeline import FUSEPipeline


def bgr_color(rgb_color):
    """Convert normalized RGB to BGR uint8 for OpenCV."""
    return (int(rgb_color[2] * 255), int(rgb_color[1] * 255), int(rgb_color[0] * 255))


def draw_objects(frame, objects):
    """Draw bounding boxes, masks, labels, and 3D info on frame."""
    overlay = frame.copy()
    for obj in objects:
        color_bgr = bgr_color(obj.color)

        # Draw filled mask
        overlay[obj.mask] = (
            overlay[obj.mask] * 0.5 + np.array(color_bgr) * 0.5
        ).astype(np.uint8)

        # Draw box
        x1, y1, x2, y2 = obj.box_2d
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)

        # Label + confidence
        text = f"{obj.label} {obj.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color_bgr, -1)
        cv2.putText(frame, text, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # 3D info
        if obj.source == "fused":
            cx, cy, cz = obj.centroid
            info = f"({cx:.2f}, {cy:.2f}, {cz:.2f})m [{obj.num_points}pts]"
            cv2.putText(frame, info, (x1, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "[2D only]", (x1, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 255), 1)

    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    return frame


def main():
    svo_path = sys.argv[1] if len(sys.argv) > 1 else None
    classes = ["mug", "phone", "cup", "fork", "bottle"]

    print("Starting FUSE pipeline...")
    with FUSEPipeline(classes, svo_path=svo_path) as pipe:
        print(f"Calibration: {pipe.get_calibration()}")
        print(f"Detecting: {classes}")

        # Open3D viewers
        vis_obj = o3d.visualization.Visualizer()
        vis_obj.create_window("FUSE - 3D Objects", width=720, height=480, left=750, top=50)
        pcd_obj = o3d.geometry.PointCloud()

        vis_scene = o3d.visualization.Visualizer()
        vis_scene.create_window("FUSE - Full Point Cloud", width=720, height=480, left=750, top=580)
        pcd_scene = o3d.geometry.PointCloud()

        first_frame = True
        print("Press 'q' in the RGB window to quit.")

        while True:
            bgr, objects, scene_xyz, scene_rgb = pipe.process_frame()
            if bgr is None:
                if svo_path:
                    print("End of SVO file.")
                    break
                continue

            # Print detections
            for obj in objects:
                print(obj)

            # Build object point cloud
            all_xyz, all_colors = [], []
            for obj in objects:
                if obj.num_points > 0:
                    all_xyz.append(obj.points_3d)
                    all_colors.append(np.tile(obj.color, (obj.num_points, 1)))

            if all_xyz:
                pcd_obj.points = o3d.utility.Vector3dVector(np.vstack(all_xyz))
                pcd_obj.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
            else:
                pcd_obj.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                pcd_obj.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))

            # Full scene
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

            # 2D overlay
            frame = draw_objects(bgr, objects)
            cv2.imshow("FUSE - Pipeline Output", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vis_obj.destroy_window()
        vis_scene.destroy_window()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
