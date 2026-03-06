"""FusedObject dataclass — the unified output of the FUSE pipeline."""

from dataclasses import dataclass
import numpy as np


# Per-label colors (R, G, B) normalized 0-1
LABEL_COLORS = {
    "mug":    (1.0, 0.0, 0.0),    # red
    "phone":  (1.0, 1.0, 0.0),    # yellow
    "cup":    (0.0, 1.0, 0.0),    # green
    "fork":   (0.0, 0.0, 1.0),    # blue
    "bottle": (1.0, 0.0, 1.0),    # magenta
}
DEFAULT_LABEL_COLOR = (1.0, 1.0, 1.0)


@dataclass
class FusedObject:
    label: str               # "cup"
    confidence: float        # 0.94
    source: str              # "fused" | "2d_only"
    box_2d: tuple            # (x_min, y_min, x_max, y_max)
    mask: np.ndarray         # (H, W) bool — pixel-level segmentation mask
    points_3d: np.ndarray    # N x 3 (partial cluster)
    centroid: tuple          # (x, y, z) meters
    color: tuple             # (R, G, B) for point cloud visualization

    @property
    def num_points(self):
        return len(self.points_3d)

    def __repr__(self):
        return (f"FusedObject(label='{self.label}', conf={self.confidence:.2f}, "
                f"source='{self.source}', points={self.num_points}, "
                f"centroid=({self.centroid[0]:.2f}, {self.centroid[1]:.2f}, {self.centroid[2]:.2f}))")
