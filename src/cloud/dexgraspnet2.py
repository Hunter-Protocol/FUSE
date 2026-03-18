"""Modal serverless endpoint: DexGraspNet 2.0 grasp inference on A100.

Takes a mesh (vertices + faces) and returns LEAP Hand grasp poses
(wrist SE3 + 16 joint angles).

Pipeline on Modal:
  1. Render synthetic depth from mesh (multiple viewpoints)
  2. Build network_input (point cloud + seg + edge + extrinsics)
  3. Run DexGraspNet 2.0 inference (ResUNet14 + SE3 diffusion)
  4. Return ranked grasp poses

Deploy:   modal deploy src/cloud/dexgraspnet2.py
Test:     modal run src/cloud/dexgraspnet2.py
"""

import modal

app = modal.App("fuse-dexgraspnet2")

dexgraspnet_image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.7.1-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "libgomp1",
                 "build-essential", "g++", "libopenblas-dev", "libosmesa6-dev",
                 "libgl1-mesa-dev", "libglfw3", "patchelf", "ninja-build")
    # Pin setuptools FIRST before anything else
    .run_commands("pip install 'setuptools<70' wheel")
    .pip_install(
        "torch==2.0.1", "torchvision==0.15.2",
        index_url="https://download.pytorch.org/whl/cu117",
    )
    .pip_install(
        "numpy==1.24.4", "trimesh", "scipy", "pillow",
        "huggingface_hub", "safetensors", "einops",
        "tqdm", "transforms3d", "urdf_parser_py",
        "ikpy", "rich", "coacd", "diffusers",
        "open3d==0.17.0", "pyrender", "opencv-python-headless",
    )
    # Install MinkowskiEngine from source (CUDA 11.7 matches)
    .run_commands(
        "git clone https://github.com/NVIDIA/MinkowskiEngine.git /opt/MinkowskiEngine",
        "cd /opt/MinkowskiEngine && "
        "CXX=g++ CC=gcc CUDA_HOME=/usr/local/cuda-11.7 "
        "python setup.py install --blas=openblas",
    )
    # Install nflows
    .run_commands(
        "git clone https://github.com/nkolot/nflows.git /opt/nflows",
        "cd /opt/nflows && pip install -e .",
    )
    # Install PyTorch3D from GitHub source
    .run_commands(
        "pip install fvcore iopath",
        "git clone https://github.com/facebookresearch/pytorch3d.git /opt/pytorch3d",
        "cd /opt/pytorch3d && "
        "CXX=g++ CC=gcc CUDA_HOME=/usr/local/cuda-11.7 "
        "TORCH_CUDA_ARCH_LIST='7.0;7.5;8.0;8.6' "
        "FORCE_CUDA=1 pip install --no-build-isolation .",
    )
    # Clone DexGraspNet2 and download checkpoint
    .run_commands(
        "git clone https://github.com/PKU-EPIC/DexGraspNet2.git /opt/DexGraspNet2",
    )
    # Checkpoint will be downloaded at runtime if not present
    # HuggingFace repo may require auth — handle at runtime
    .run_commands(
        "mkdir -p /opt/DexGraspNet2/experiments/dex_ours/ckpt",
    )
    .env({
        "PYTHONPATH": "/opt/DexGraspNet2",
        "PYOPENGL_PLATFORM": "osmesa",
    })
)


def render_depth_views(vertices, faces, num_views=8, image_size=640):
    """Render synthetic depth images from a mesh at multiple viewpoints.

    Returns list of (depth_image, extrinsic_matrix) tuples.
    """
    import numpy as np
    import trimesh
    import pyrender

    mesh_tri = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Center mesh at origin, scale to fit in view
    center = mesh_tri.centroid
    mesh_tri.vertices -= center
    extent = mesh_tri.bounding_sphere.primitive.radius

    # Camera intrinsics (RealSense-like)
    fx = fy = 616.0
    cx, cy = image_size / 2, image_size / 2

    # Create pyrender scene
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh_tri)

    # Generate viewpoints on a sphere
    views = []
    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        elevation = np.pi / 6  # 30 degrees above horizontal

        # Camera position on sphere
        dist = extent * 3.0
        cx_pos = dist * np.cos(elevation) * np.cos(angle)
        cy_pos = dist * np.cos(elevation) * np.sin(angle)
        cz_pos = dist * np.sin(elevation)
        cam_pos = np.array([cx_pos, cy_pos, cz_pos])

        # Look-at matrix
        forward = -cam_pos / np.linalg.norm(cam_pos)
        right = np.cross(np.array([0, 0, 1]), forward)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1, 0, 0])
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)

        # Camera extrinsic (world→camera)
        R = np.stack([right, -up, forward], axis=0)
        t = -R @ cam_pos
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t

        # Camera pose for pyrender (camera→world) = inverse of extrinsic
        camera_pose = np.linalg.inv(extrinsic)

        # Render
        scene = pyrender.Scene()
        scene.add(mesh_pyrender)
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy,
                                           znear=0.01, zfar=10.0)
        scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=camera_pose)

        renderer = pyrender.OffscreenRenderer(image_size, image_size)
        _, depth = renderer.render(scene)
        renderer.delete()

        views.append((depth, extrinsic, fx, fy, cx, cy))

    return views


def depth_to_point_cloud(depth, fx, fy, cx, cy):
    """Convert depth image to point cloud in camera frame."""
    import numpy as np

    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    mask = depth > 0

    z = depth[mask]
    x = (u[mask] - cx) * z / fx
    y = (v[mask] - cy) * z / fy

    points = np.stack([x, y, z], axis=-1).astype(np.float32)
    return points, mask


def build_network_input(vertices, faces, num_views=8):
    """Build DexGraspNet 2.0 network input from mesh.

    Returns dict with pc, seg, edge, extrinsics arrays.
    """
    import numpy as np
    import cv2

    views = render_depth_views(vertices, faces, num_views=num_views)
    num_points = 40000

    pc_all = []
    seg_all = []
    edge_all = []
    extrinsics_all = []

    for depth, extrinsic, fx, fy, cx, cy in views:
        # Convert depth to point cloud
        points, mask = depth_to_point_cloud(depth, fx, fy, cx, cy)

        if len(points) < 100:
            continue

        # Segmentation: all points are the object (seg=1)
        seg = np.ones(len(points), dtype=np.int64)

        # Edge detection on depth
        depth_norm = depth.copy()
        depth_norm[depth_norm > 0] = (depth_norm[depth_norm > 0] /
                                       depth_norm[depth_norm > 0].max() * 200)
        depth_uint8 = depth_norm.astype(np.uint8)
        edges_img = cv2.Canny(depth_uint8, 10, 20)
        kernel = np.ones((5, 5), np.uint8)
        edges_img = cv2.dilate(edges_img, kernel, iterations=1)
        # Mark borders as edges
        edges_img[0, :] = 255
        edges_img[-1, :] = 255
        edges_img[:, 0] = 255
        edges_img[:, -1] = 255

        edge_flat = edges_img[mask].astype(np.int64)

        # Sample to fixed size
        n = len(points)
        if n >= num_points:
            idx = np.random.choice(n, num_points, replace=False)
        else:
            idx = np.random.choice(n, num_points, replace=True)

        pc_all.append(points[idx])
        seg_all.append(seg[idx])
        edge_all.append(edge_flat[idx])
        extrinsics_all.append(extrinsic)

    pc_all = np.array(pc_all, dtype=np.float32)
    seg_all = np.array(seg_all, dtype=np.int64)
    edge_all = np.array(edge_all, dtype=np.int64)
    extrinsics_all = np.array(extrinsics_all, dtype=np.float64)

    return {
        'pc': pc_all,
        'seg': seg_all,
        'edge': edge_all,
        'extrinsics': extrinsics_all,
    }


@app.cls(
    image=dexgraspnet_image,
    gpu="a100-80gb",
    timeout=600,
    scaledown_window=60,
)
class DexGraspNet2Model:
    @modal.enter()
    def load_model(self):
        """Load DexGraspNet 2.0 model."""
        import sys
        import os
        import torch

        sys.path.insert(0, '/opt/DexGraspNet2')
        os.chdir('/opt/DexGraspNet2')

        # Download checkpoint if not present
        ckpt_path = 'experiments/dex_ours/ckpt/ckpt_50000.pth'
        if not os.path.exists(ckpt_path):
            print("Downloading DexGraspNet 2.0 checkpoint...")
            try:
                from huggingface_hub import hf_hub_download
                hf_hub_download(
                    'lhrlhr/DexGraspNet2.0',
                    'experiments/dex_ours/ckpt/ckpt_50000.pth',
                    local_dir='/opt/DexGraspNet2',
                    repo_type='dataset',
                )
            except Exception as e:
                print(f"HuggingFace download failed: {e}")
                print("Trying alternative download...")
                # Try as model repo instead of dataset
                hf_hub_download(
                    'lhrlhr/DexGraspNet2.0',
                    'experiments/dex_ours/ckpt/ckpt_50000.pth',
                    local_dir='/opt/DexGraspNet2',
                )

        from src.utils.robot_model import RobotModel
        from src.utils.config import ckpt_to_config
        from src.network.model import get_model

        urdf_path = 'robot_models/urdf/leap_hand_simplified.urdf'
        meta_path = 'robot_models/meta/leap_hand/meta.yaml'

        print("Loading DexGraspNet 2.0...")
        self.robot_model = RobotModel(urdf_path, meta_path)
        self.config = ckpt_to_config(ckpt_path)
        self.model = get_model(self.config.model)
        self.model.config.voxel_size = self.config.data.voxel_size

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(ckpt['model'], strict=False)
        self.model.to('cuda:0')
        self.model.eval()
        self.device = torch.device('cuda:0')
        print("DexGraspNet 2.0 loaded")

    @modal.method()
    def predict_grasps(self, vertices_list, faces_list, num_views=8, top_k=10):
        """Predict LEAP Hand grasps from mesh.

        Args:
            vertices_list: list of [x,y,z] vertex coordinates
            faces_list: list of [i,j,k] face indices
            num_views: number of synthetic depth views to render
            top_k: number of top grasps to return

        Returns:
            dict with 'rotations', 'translations', 'joint_angles', 'scores',
                       'joint_names'
        """
        import numpy as np
        import torch
        import time

        from src.utils.dataset import get_sparse_tensor

        vertices = np.array(vertices_list, dtype=np.float32)
        faces = np.array(faces_list, dtype=np.int32)
        print(f"Input mesh: {len(vertices)} verts, {len(faces)} faces")

        # Build network input
        t0 = time.time()
        net_input = build_network_input(vertices, faces, num_views=num_views)
        dt_preprocess = time.time() - t0
        n_views = len(net_input['pc'])
        print(f"Preprocessing: {dt_preprocess:.2f}s, {n_views} views")

        pc_all = torch.tensor(net_input['pc'], dtype=torch.float)
        seg_all = torch.tensor(net_input['seg'], dtype=torch.long)
        edge_all = torch.tensor(net_input['edge'], dtype=torch.long)
        extrinsics_all = net_input['extrinsics']

        # Run inference
        t0 = time.time()
        with torch.no_grad():
            rotations, translations, qposs, scores = [], [], [], []
            stride = min(32, n_views)
            for i in range(0, n_views, stride):
                end = min(i + stride, n_views)
                data_part = get_sparse_tensor(pc_all[i:end], self.config.data.voxel_size)
                data_part['seg'] = seg_all[i:end]
                data_part = {k: v.to(self.device) for k, v in data_part.items()}
                edge_part = edge_all[i:end]

                rotation, translation, qpos, score, _, _, _ = (
                    t.cpu() for t in self.model.sample(
                        data_part, 1024,
                        graspness_scale=5,
                        allow_fail=True,
                        cate=False,
                        edge=edge_part.to(self.device),
                        with_score_parts=True,
                    )
                )
                rotations.append(rotation)
                translations.append(translation)
                qposs.append(qpos)
                scores.append(score)

            rotation = torch.cat(rotations, dim=0)
            translation = torch.cat(translations, dim=0)
            qpos = torch.cat(qposs, dim=0)
            score = torch.cat(scores, dim=0)

        dt_inference = time.time() - t0
        print(f"Inference: {dt_inference:.2f}s")

        # Select best grasp per view, then overall top-k
        best_per_view = score.argmax(dim=1)
        arange = torch.arange(len(best_per_view))
        sel_rot = rotation[arange, best_per_view].numpy()
        sel_trans = translation[arange, best_per_view].numpy()
        sel_qpos = qpos[arange, best_per_view].numpy()
        sel_score = score[arange, best_per_view].numpy()

        # Transform to world frame using extrinsics
        sel_rot_world = extrinsics_all[:n_views, :3, :3] @ sel_rot
        sel_trans_world = (extrinsics_all[:n_views, :3, :3] @ sel_trans[:, :, None] +
                          extrinsics_all[:n_views, :3, 3:])[:, :, 0]

        # Sort by score and return top-k
        order = np.argsort(-sel_score)[:top_k]

        result = {
            'rotations': sel_rot_world[order].tolist(),
            'translations': sel_trans_world[order].tolist(),
            'joint_angles': sel_qpos[order].tolist(),
            'scores': sel_score[order].tolist(),
            'joint_names': self.robot_model.joint_names,
            'preprocess_time': dt_preprocess,
            'inference_time': dt_inference,
            'num_views': n_views,
        }
        print(f"Returning {len(order)} grasps, best score: {sel_score[order[0]]:.4f}")
        return result


@app.local_entrypoint()
def main():
    """Test: run DexGraspNet 2.0 on saved Hunyuan3D mesh."""
    import numpy as np

    data_dir = "/home/hunter/Desktop/FUSE/data"

    # Load mesh
    mesh_file = f"{data_dir}/mug_hunyuan3d_mesh.npz"
    try:
        mesh_data = np.load(mesh_file)
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
    except FileNotFoundError:
        print(f"No mesh file at {mesh_file}")
        print("Run save_mesh.py first to get mesh from Hunyuan3D")
        return

    print(f"Mesh: {len(vertices)} verts, {len(faces)} faces")

    model = DexGraspNet2Model()
    result = model.predict_grasps.remote(
        vertices.tolist(), faces.tolist(),
        num_views=8, top_k=5,
    )

    print(f"\nResults:")
    print(f"  Preprocess: {result['preprocess_time']:.2f}s")
    print(f"  Inference: {result['inference_time']:.2f}s")
    print(f"  Views: {result['num_views']}")
    print(f"  Joint names: {result['joint_names']}")

    for i, (rot, trans, qpos, score) in enumerate(zip(
        result['rotations'], result['translations'],
        result['joint_angles'], result['scores']
    )):
        print(f"\n  Grasp {i}: score={score:.4f}")
        print(f"    Translation: ({trans[0]:.3f}, {trans[1]:.3f}, {trans[2]:.3f})")
        print(f"    Joint angles: {[f'{q:.2f}' for q in qpos]}")

    # Save results
    np.savez(
        f"{data_dir}/../results/mug_dexgrasps.npz",
        rotations=np.array(result['rotations']),
        translations=np.array(result['translations']),
        joint_angles=np.array(result['joint_angles']),
        scores=np.array(result['scores']),
        joint_names=result['joint_names'],
    )
    print(f"\nSaved to results/mug_dexgrasps.npz")
