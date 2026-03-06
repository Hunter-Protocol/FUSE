"""Open-vocabulary detection + segmentation using YOLOE."""

import numpy as np
from ultralytics import YOLOE


class Detector:
    def __init__(self, model_size="11s", confidence=0.3):
        self.model = YOLOE(f"yoloe-{model_size}-seg.pt")
        self.confidence = confidence

    def set_classes(self, classes):
        """Set open-vocab classes to detect. E.g. ["mug", "phone", "cup", "fork", "bottle"]"""
        self.model.set_classes(classes)

    def detect(self, bgr_frame):
        """Run detection + segmentation on a BGR frame.

        Returns list of dicts:
            label: str
            confidence: float
            box_2d: (x1, y1, x2, y2)
            mask: np.ndarray (H, W) bool — pixel-level mask at original resolution
        """
        results = self.model.predict(bgr_frame, conf=self.confidence, verbose=False)
        detections = []
        for r in results:
            if r.masks is None:
                continue
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                # masks.data is (N, mask_h, mask_w), need to resize to original frame
                mask = r.masks.data[i].cpu().numpy().astype(bool)
                # Resize mask to original image dimensions if needed
                if mask.shape != bgr_frame.shape[:2]:
                    import cv2
                    mask = cv2.resize(mask.astype(np.uint8),
                                      (bgr_frame.shape[1], bgr_frame.shape[0]),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
                detections.append({
                    "label": r.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "box_2d": (int(x1), int(y1), int(x2), int(y2)),
                    "mask": mask,
                })
        return detections
