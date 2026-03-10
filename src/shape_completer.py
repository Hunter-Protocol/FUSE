"""Shape completion using PoinTr (ShapeNet55 checkpoint).

Wraps PoinTr inference: partial point cloud in camera frame → completed shape.
"""

import importlib
import sys
import time
import types
import numpy as np
import torch

# PoinTr repo path — adjust if cloned elsewhere
POINTR_DIR = "/home/hunter/Desktop/PoinTr"


def _load_pointr_model_class():
    """Import PoinTr model class without triggering models/__init__.py.

    The PoinTr repo's models/__init__.py imports ALL models (GRNet, etc.)
    which need CUDA extensions we don't have. We load only the files we need.
    """
    if POINTR_DIR not in sys.path:
        sys.path.insert(0, POINTR_DIR)

    # Create a fake 'models' package to satisfy relative imports
    if "models" not in sys.modules:
        models_pkg = types.ModuleType("models")
        models_pkg.__path__ = [f"{POINTR_DIR}/models"]
        models_pkg.__package__ = "models"
        sys.modules["models"] = models_pkg

    # Load build.py (needed for MODELS registry)
    build_spec = importlib.util.spec_from_file_location(
        "models.build", f"{POINTR_DIR}/models/build.py")
    build_mod = importlib.util.module_from_spec(build_spec)
    sys.modules["models.build"] = build_mod
    build_spec.loader.exec_module(build_mod)

    # Load dgcnn_group.py
    dg_spec = importlib.util.spec_from_file_location(
        "models.dgcnn_group", f"{POINTR_DIR}/models/dgcnn_group.py")
    dg_mod = importlib.util.module_from_spec(dg_spec)
    sys.modules["models.dgcnn_group"] = dg_mod
    dg_spec.loader.exec_module(dg_mod)

    # Load Transformer.py
    tf_spec = importlib.util.spec_from_file_location(
        "models.Transformer", f"{POINTR_DIR}/models/Transformer.py")
    tf_mod = importlib.util.module_from_spec(tf_spec)
    sys.modules["models.Transformer"] = tf_mod
    tf_spec.loader.exec_module(tf_mod)

    # Load PoinTr.py
    pt_spec = importlib.util.spec_from_file_location(
        "models.PoinTr", f"{POINTR_DIR}/models/PoinTr.py")
    pt_mod = importlib.util.module_from_spec(pt_spec)
    sys.modules["models.PoinTr"] = pt_mod
    pt_spec.loader.exec_module(pt_mod)

    return pt_mod.PoinTr

# ShapeNet55 category mapping: FUSE label → ShapeNet synset ID
# Based on data/shapenet_synset_dict.json from PoinTr repo
LABEL_TO_SYNSET = {
    "mug":    "03797390",  # mug
    "bottle": "02876657",  # bottle
    "phone":  "02992529",  # cellphone
    "cup":    "03797390",  # mug (closest match)
    "fork":   "03624134",  # knife (approximate)
}


class ShapeCompleter:
    """PoinTr-based shape completion for partial point clouds."""

    def __init__(self, checkpoint_path=None, device="cuda"):
        if checkpoint_path is None:
            checkpoint_path = f"{POINTR_DIR}/pretrained/pointr_shapenet55.pth"

        self.device = torch.device(device)

        from easydict import EasyDict
        PoinTr = _load_pointr_model_class()

        config = EasyDict(
            trans_dim=384,
            knn_layer=1,
            num_pred=6144,
            num_query=96,
        )
        self.model = PoinTr(config)

        # Load checkpoint (keys prefixed with "module." from DDP training)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"]
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(cleaned)

        self.model.to(self.device)
        self.model.eval()
        self.num_input = 2048

        # Warmup
        with torch.no_grad():
            dummy = torch.randn(1, self.num_input, 3, device=self.device)
            self.model(dummy)

        print(f"[ShapeCompleter] PoinTr loaded on {device}, num_pred={config.num_pred}")

    def can_complete(self, label: str) -> bool:
        """Check if this label has a ShapeNet category mapping."""
        return label in LABEL_TO_SYNSET

    def preprocess(self, points: np.ndarray):
        """Downsample to 2048 points via FPS and normalize to unit sphere.

        Args:
            points: (N, 3) float32 in camera frame

        Returns:
            tensor: (1, 2048, 3) on device, normalized
            centroid: (3,) original centroid
            scale: float, max radius for denormalization
        """
        from pointnet2_ops import pointnet2_utils

        n = len(points)
        if n < self.num_input:
            # Pad by repeating random points
            idx = np.random.choice(n, self.num_input - n, replace=True)
            points = np.vstack([points, points[idx]])
        elif n > self.num_input:
            # FPS downsample on GPU
            pts_tensor = torch.from_numpy(points).float().unsqueeze(0).to(self.device)
            fps_idx = pointnet2_utils.furthest_point_sample(pts_tensor, self.num_input)
            pts_tensor = pointnet2_utils.gather_operation(
                pts_tensor.transpose(1, 2).contiguous(), fps_idx
            ).transpose(1, 2).contiguous()
            points = pts_tensor.squeeze(0).cpu().numpy()

        # Normalize to unit sphere
        centroid = points.mean(axis=0)
        points = points - centroid
        scale = np.max(np.linalg.norm(points, axis=1))
        if scale > 0:
            points = points / scale

        tensor = torch.from_numpy(points).float().unsqueeze(0).to(self.device)
        return tensor, centroid, scale

    def postprocess(self, completed, centroid, scale):
        """Denormalize completed points back to camera frame.

        Args:
            completed: (M, 3) numpy array in normalized space
            centroid: (3,) original centroid
            scale: float, original max radius

        Returns:
            (M, 3) float32 in camera frame (meters)
        """
        return (completed * scale + centroid).astype(np.float32)

    @torch.no_grad()
    def complete(self, points: np.ndarray):
        """Run full completion pipeline.

        Args:
            points: (N, 3) float32 partial cloud in camera frame

        Returns:
            (M, 3) float32 completed cloud in camera frame, or None if too few points
        """
        if len(points) < 64:
            return None

        tensor, centroid, scale = self.preprocess(points)
        coarse, dense = self.model(tensor)
        completed = dense.squeeze(0).cpu().numpy()  # (num_pred + num_input, 3)
        return self.postprocess(completed, centroid, scale)


if __name__ == "__main__":
    import open3d as o3d

    completer = ShapeCompleter()

    # If a .npy file is provided, use it; otherwise generate a synthetic partial sphere
    if len(sys.argv) > 1:
        points = np.load(sys.argv[1]).astype(np.float32)
        print(f"Loaded {len(points)} points from {sys.argv[1]}")
    else:
        # Generate synthetic partial point cloud (half sphere)
        print("No .npy file provided, generating synthetic half-sphere...")
        theta = np.random.uniform(0, np.pi, 5000)
        phi = np.random.uniform(0, np.pi, 5000)  # only front half
        r = 0.05  # 5cm radius
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta) + 0.5  # 50cm away from camera
        points = np.stack([x, y, z], axis=1).astype(np.float32)

    print(f"Input: {points.shape}")

    t0 = time.time()
    completed = completer.complete(points)
    dt = time.time() - t0
    print(f"Completion: {completed.shape} in {dt*1000:.1f}ms")

    # Visualize side by side
    pcd_orig = o3d.geometry.PointCloud()
    pcd_orig.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd_orig.paint_uniform_color([1.0, 0.0, 0.0])  # red = original

    pcd_comp = o3d.geometry.PointCloud()
    pcd_comp.points = o3d.utility.Vector3dVector(completed.astype(np.float64))
    pcd_comp.paint_uniform_color([0.0, 0.7, 1.0])  # cyan = completed

    # Shift completed to the right for side-by-side view
    offset = points[:, 0].max() - points[:, 0].min() + 0.05
    pcd_comp.translate([offset, 0, 0])

    print("Visualizing: RED = original, CYAN = completed")
    o3d.visualization.draw_geometries([pcd_orig, pcd_comp],
                                       window_name="Shape Completion Test",
                                       width=1200, height=600)
