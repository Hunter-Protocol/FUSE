"""2D object detection using YOLO World (open-vocabulary)."""

from ultralytics import YOLOWorld


class Detector2D:
    def __init__(self, model_size="s", confidence=0.3):
        self.model = YOLOWorld(f"yolov8{model_size}-worldv2.pt")
        self.confidence = confidence

    def set_classes(self, classes):
        """Set open-vocab classes to detect. E.g. ["mug", "phone", "cup", "fork", "bottle"]"""
        self.model.set_classes(classes)

    def detect(self, bgr_frame):
        """Run detection on a BGR frame.

        Returns list of dicts: {label, confidence, box_2d: (x1, y1, x2, y2)}
        """
        results = self.model.predict(bgr_frame, conf=self.confidence, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                detections.append({
                    "label": r.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "box_2d": (int(x1), int(y1), int(x2), int(y2)),
                })
        return detections
