"""ZED Mini camera interface for grabbing RGB frames and point clouds."""

import pyzed.sl as sl
import numpy as np


class ZEDCamera:
    def __init__(self, svo_path=None):
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

        if svo_path:
            self.init_params.set_from_svo_file(svo_path)

        self.image = sl.Mat()
        self.point_cloud = sl.Mat()

    def open(self):
        status = self.zed.open(self.init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {status}")

    def grab(self):
        """Grab a new frame. Returns True if successful."""
        runtime = sl.RuntimeParameters()
        return self.zed.grab(runtime) == sl.ERROR_CODE.SUCCESS

    def get_rgb(self):
        """Get RGB image as numpy array (H, W, 3) uint8."""
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        # ZED returns BGRA, convert to RGB
        bgra = self.image.get_data()
        return bgra[:, :, :3][:, :, ::-1].copy()

    def get_bgr(self):
        """Get BGR image as numpy array (H, W, 3) uint8. Ready for OpenCV."""
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        return self.image.get_data()[:, :, :3].copy()

    def get_point_cloud(self):
        """Get point cloud xyz (N,3) and rgb colors (N,3) as float32. NaN points removed."""
        self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
        pc_data = self.point_cloud.get_data()
        xyz = pc_data[:, :, :3].reshape(-1, 3)
        # Extract RGB from the packed RGBA float in channel 3
        rgba_float = pc_data[:, :, 3].reshape(-1)
        valid = np.isfinite(xyz).all(axis=1) & np.isfinite(rgba_float)
        xyz = xyz[valid].astype(np.float32)
        # Unpack RGBA: the float32 holds packed RGBA bytes
        rgba_bytes = rgba_float[valid].view(np.uint8).reshape(-1, 4)
        rgb = rgba_bytes[:, :3].astype(np.float64) / 255.0  # RGB, normalize
        return xyz, rgb

    def get_calibration(self):
        """Get camera intrinsics for 3D->2D projection."""
        cam_info = self.zed.get_camera_information()
        calib = cam_info.camera_configuration.calibration_parameters.left_cam
        return {
            "fx": calib.fx,
            "fy": calib.fy,
            "cx": calib.cx,
            "cy": calib.cy,
        }

    def close(self):
        self.zed.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
