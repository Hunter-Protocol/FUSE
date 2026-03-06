"""Phase 2 demo: ZED camera + YOLO World 2D detection with bounding boxes."""

import sys
import cv2
from camera import ZEDCamera
from detector_2d import Detector2D

# Colors for each class (BGR for OpenCV)
COLORS = {
    "mug": (0, 0, 255),       # red
    "phone": (0, 255, 255),    # yellow
    "cup": (0, 255, 0),        # green
    "fork": (255, 0, 0),       # blue
    "bottle": (255, 0, 255),   # magenta
}
DEFAULT_COLOR = (255, 255, 255)


def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box_2d"]
        label = det["label"]
        conf = det["confidence"]
        color = COLORS.get(label, DEFAULT_COLOR)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return frame


def main():
    svo_path = sys.argv[1] if len(sys.argv) > 1 else None
    classes = ["mug", "phone", "cup", "fork", "bottle"]

    print("Loading YOLO World model...")
    detector = Detector2D(model_size="s", confidence=0.3)
    detector.set_classes(classes)
    print(f"Detecting: {classes}")

    with ZEDCamera(svo_path=svo_path) as cam:
        print("Press 'q' to quit.")

        while True:
            if not cam.grab():
                if svo_path:
                    print("End of SVO file.")
                    break
                continue

            bgr = cam.get_bgr()
            detections = detector.detect(bgr)
            frame = draw_detections(bgr, detections)

            cv2.imshow("FUSE - 2D Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
