"""Visualize v2.0 vs v2.1: input image + two point cloud windows."""

import numpy as np
import open3d as o3d
import cv2
from PIL import Image

data_dir = "/home/hunter/Desktop/FUSE/data"

pts_v20 = np.load(f"{data_dir}/mug_hunyuan3d_v20.npy")
pts_v21 = np.load(f"{data_dir}/mug_hunyuan3d_v21.npy")

print(f"v2.0: {pts_v20.shape}, v2.1: {pts_v21.shape}")

# Window 1: Input image (OpenCV)
image = Image.open(f"{data_dir}/mug_crop_hunyuan3d_cloud.png").convert("RGBA")
input_rgba = np.array(image)
alpha = input_rgba[:, :, 3:4].astype(np.float32) / 255.0
display_rgb = (input_rgba[:, :, :3].astype(np.float32) * alpha +
               255.0 * (1.0 - alpha)).astype(np.uint8)
display_bgr = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)
cv2.imshow("Input Image", display_bgr)

# Window 2: v2.0 point cloud
pcd_v20 = o3d.geometry.PointCloud()
pcd_v20.points = o3d.utility.Vector3dVector(pts_v20.astype(np.float64))
pcd_v20.paint_uniform_color([1.0, 0.3, 0.3])

vis_20 = o3d.visualization.Visualizer()
vis_20.create_window("v2.0 (1.1B) — 27s", width=600, height=600, left=450)
vis_20.add_geometry(pcd_v20)

# Window 3: v2.1 point cloud
pcd_v21 = o3d.geometry.PointCloud()
pcd_v21.points = o3d.utility.Vector3dVector(pts_v21.astype(np.float64))
pcd_v21.paint_uniform_color([0.3, 0.5, 1.0])

vis_21 = o3d.visualization.Visualizer()
vis_21.create_window("v2.1 (3B) — 31s", width=600, height=600, left=1100)
vis_21.add_geometry(pcd_v21)

print("Window 1: Input | Window 2: v2.0 (RED) | Window 3: v2.1 (BLUE)")
print("Press 'q' to exit")

while True:
    vis_20.poll_events()
    vis_20.update_renderer()
    vis_21.poll_events()
    vis_21.update_renderer()
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

vis_20.destroy_window()
vis_21.destroy_window()
cv2.destroyAllWindows()
