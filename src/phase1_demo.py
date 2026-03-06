"""Phase 1 demo: ZED camera -> OpenCV RGB display + Open3D point cloud viewer."""

import sys
import cv2
import numpy as np
import open3d as o3d
from camera import ZEDCamera


def main():
    svo_path = sys.argv[1] if len(sys.argv) > 1 else None

    with ZEDCamera(svo_path=svo_path) as cam:
        calib = cam.get_calibration()
        print(f"Camera intrinsics: fx={calib['fx']:.1f}, fy={calib['fy']:.1f}, "
              f"cx={calib['cx']:.1f}, cy={calib['cy']:.1f}")

        # Set up Open3D visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window("FUSE - Point Cloud", width=720, height=480)
        pcd = o3d.geometry.PointCloud()
        first_frame = True

        print("Press 'q' in the RGB window to quit.")

        while True:
            if not cam.grab():
                if svo_path:
                    print("End of SVO file.")
                    break
                continue

            # RGB display
            bgr = cam.get_bgr()
            cv2.imshow("FUSE - RGB", bgr)

            # Point cloud display
            xyz, rgb = cam.get_point_cloud()
            if len(xyz) > 0:
                # Subsample to max 200k points for smooth rendering
                if len(xyz) > 200_000:
                    idx = np.random.choice(len(xyz), 200_000, replace=False)
                    xyz, rgb = xyz[idx], rgb[idx]
                pcd.points = o3d.utility.Vector3dVector(xyz)
                pcd.colors = o3d.utility.Vector3dVector(rgb)
                if first_frame:
                    vis.add_geometry(pcd)
                    # Increase point size for denser appearance
                    vis.get_render_option().point_size = 2.0
                    first_frame = False
                else:
                    vis.update_geometry(pcd)

            vis.poll_events()
            vis.update_renderer()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vis.destroy_window()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
