"""FUSE pipeline: ZED camera → YOLOE Seg → 3D extraction → FusedObject output."""

import numpy as np
import pyzed.sl as sl
from camera import ZEDCamera
from detector import Detector
from fused_object import FusedObject, LABEL_COLORS, DEFAULT_LABEL_COLOR


class FUSEPipeline:
    def __init__(self, classes, svo_path=None, model_size="11s", confidence=0.3):
        self.classes = classes
        self.cam = ZEDCamera(svo_path=svo_path)
        self.detector = Detector(model_size=model_size, confidence=confidence)
        self.detector.set_classes(classes)
        self.pc_mat = sl.Mat()

    def start(self):
        self.cam.open()

    def stop(self):
        self.cam.close()

    def get_calibration(self):
        return self.cam.get_calibration()

    def process_frame(self):
        """Grab a frame, run detection + 3D extraction.

        Returns:
            bgr: (H, W, 3) uint8 — raw BGR frame
            objects: list[FusedObject] — detected objects with 3D data
            scene_xyz: (N, 3) float32 — full scene point cloud
            scene_rgb: (N, 3) float64 — full scene colors
            Returns (None, None, None, None) if grab fails.
        """
        if not self.cam.grab():
            return None, None, None, None

        bgr = self.cam.get_bgr()
        detections = self.detector.detect(bgr)

        # Get raw point cloud for mask-based extraction
        self.cam.zed.retrieve_measure(self.pc_mat, sl.MEASURE.XYZRGBA)
        pc_data = self.pc_mat.get_data()

        # Build FusedObjects
        objects = []
        for det in detections:
            mask = det["mask"]
            xyz = pc_data[:, :, :3][mask]
            valid = np.isfinite(xyz).all(axis=1)
            points_3d = xyz[valid].astype(np.float32)

            color = LABEL_COLORS.get(det["label"], DEFAULT_LABEL_COLOR)

            if len(points_3d) > 0:
                centroid = tuple(points_3d.mean(axis=0))
                source = "fused"
            else:
                centroid = (0.0, 0.0, 0.0)
                source = "2d_only"

            objects.append(FusedObject(
                label=det["label"],
                confidence=det["confidence"],
                source=source,
                box_2d=det["box_2d"],
                mask=mask,
                points_3d=points_3d,
                centroid=centroid,
                color=color,
            ))

        # Full scene point cloud
        scene_xyz, scene_rgb = self.cam.get_point_cloud()

        return bgr, objects, scene_xyz, scene_rgb

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
