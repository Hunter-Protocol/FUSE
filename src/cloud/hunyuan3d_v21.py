"""Modal serverless endpoint: run Hunyuan3D-2.1 (3B DiT) image-to-3D on a cloud A100.

Experiment: compare v2.1 against our existing v2.0 endpoint on inference time and mesh quality.

Deploy:   modal deploy src/cloud/hunyuan3d_v21.py
Test:     modal run src/cloud/hunyuan3d_v21.py
"""

import modal

# ---------- Modal setup ----------
app = modal.App("fuse-hunyuan3d-v21")

hunyuan3d_v21_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "libgomp1", "build-essential")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        # Core deps for shape generation only (no texture pipeline)
        "numpy", "pillow", "trimesh", "scipy",
        "huggingface_hub", "safetensors", "einops",
        "tqdm", "ninja", "diffusers==0.30.0", "pybind11",
        "opencv-python-headless", "omegaconf",
        "transformers==4.46.0", "accelerate==1.1.1",
        "pymeshlab", "pygltflib", "xatlas",
        "rembg", "onnxruntime",
        "imageio", "scikit-image",
        "timm", "deepspeed",
    )
    .run_commands(
        # Clone Hunyuan3D-2.1 repo
        "git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /opt/hunyuan3d-2.1",
    )
    .run_commands(
        # Pre-download model weights
        "python -c \""
        "from huggingface_hub import snapshot_download; "
        "snapshot_download('tencent/Hunyuan3D-2.1'); "
        "\"",
    )
    .env({"PYTHONPATH": "/opt/hunyuan3d-2.1/hy3dshape"})
)


@app.cls(
    image=hunyuan3d_v21_image,
    gpu="a100-80gb",
    timeout=600,
    scaledown_window=60,
)
class Hunyuan3DV21Model:
    @modal.enter()
    def load_model(self):
        """Load Hunyuan3D-2.1 shape generation pipeline."""
        import torch
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

        print("Loading Hunyuan3D-2.1 pipeline...")
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2.1",
            device="cuda",
            dtype=torch.float16,
        )
        print("Hunyuan3D-2.1 loaded on GPU")

    @modal.method()
    def generate(self, image_bytes: bytes) -> dict:
        """Run image-to-3D shape generation.

        Args:
            image_bytes: PNG/JPEG image as bytes

        Returns:
            dict with 'vertices', 'faces', 'points', timing info
        """
        import io
        import time
        import numpy as np
        import trimesh
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        print(f"Input image: {image.size}, mode: {image.mode}")

        t0 = time.time()
        mesh = self.pipeline(
            image=image,
            num_inference_steps=50,
        )[0]
        dt_gen = time.time() - t0
        print(f"Shape generation: {dt_gen:.2f}s")

        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)
        print(f"Mesh: {len(vertices)} verts, {len(faces)} faces")

        # Sample points from mesh surface
        sampled_pts, _ = trimesh.sample.sample_surface(mesh, count=8192)
        sampled_pts = sampled_pts.astype(np.float32)

        return {
            "vertices": vertices.tolist(),
            "faces": faces.tolist(),
            "points": sampled_pts.tolist(),
            "generation_time": dt_gen,
            "num_vertices": len(vertices),
            "num_faces": len(faces),
            "model_version": "2.1",
        }


# ---------- Local test ----------
@app.local_entrypoint()
def main():
    """Test: send a local image to both v2.0 and v2.1 endpoints, compare results."""
    import time
    from pathlib import Path

    image_path = "/home/hunter/Desktop/FUSE/data/mug_crop_hunyuan3d_cloud.png"
    print(f"Image: {image_path}")
    image_bytes = Path(image_path).read_bytes()

    data_dir = "/home/hunter/Desktop/FUSE/data"

    # --- Run v2.1 ---
    print("\n" + "=" * 60)
    print("HUNYUAN3D v2.1 (3B DiT)")
    print("=" * 60)
    model_v21 = Hunyuan3DV21Model()

    t_start = time.time()
    result_v21 = model_v21.generate.remote(image_bytes)
    t_total_v21 = time.time() - t_start

    print(f"  End-to-end latency: {t_total_v21:.2f}s")
    print(f"  GPU generation time: {result_v21['generation_time']:.2f}s")
    print(f"  Network overhead: {t_total_v21 - result_v21['generation_time']:.2f}s")
    print(f"  Mesh: {result_v21['num_vertices']} verts, {result_v21['num_faces']} faces")
    print(f"  Points: {len(result_v21['points'])}")

    # --- Run v2.0 for comparison ---
    print("\n" + "=" * 60)
    print("HUNYUAN3D v2.0 (1.1B DiT) — existing endpoint")
    print("=" * 60)
    import modal as modal_lib
    Hunyuan3DModel = modal_lib.Cls.from_name("fuse-hunyuan3d", "Hunyuan3DModel")
    model_v20 = Hunyuan3DModel()

    t_start = time.time()
    result_v20 = model_v20.generate.remote(image_bytes)
    t_total_v20 = time.time() - t_start

    print(f"  End-to-end latency: {t_total_v20:.2f}s")
    print(f"  GPU generation time: {result_v20['generation_time']:.2f}s")
    print(f"  Network overhead: {t_total_v20 - result_v20['generation_time']:.2f}s")
    print(f"  Mesh: {result_v20['num_vertices']} verts, {result_v20['num_faces']} faces")
    print(f"  Points: {len(result_v20['points'])}")

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<25} {'v2.0':>12} {'v2.1':>12} {'Diff':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")

    gpu_20 = result_v20['generation_time']
    gpu_21 = result_v21['generation_time']
    print(f"  {'GPU gen time (s)':<25} {gpu_20:>12.2f} {gpu_21:>12.2f} {gpu_21 - gpu_20:>+12.2f}")

    verts_20 = result_v20['num_vertices']
    verts_21 = result_v21['num_vertices']
    print(f"  {'Vertices':<25} {verts_20:>12,} {verts_21:>12,} {verts_21 - verts_20:>+12,}")

    faces_20 = result_v20['num_faces']
    faces_21 = result_v21['num_faces']
    print(f"  {'Faces':<25} {faces_20:>12,} {faces_21:>12,} {faces_21 - faces_20:>+12,}")

    # --- Save results ---
    import numpy as np
    pts_v20 = np.array(result_v20['points'], dtype=np.float32)
    pts_v21 = np.array(result_v21['points'], dtype=np.float32)
    np.save(f"{data_dir}/mug_hunyuan3d_v20.npy", pts_v20)
    np.save(f"{data_dir}/mug_hunyuan3d_v21.npy", pts_v21)
    print(f"\n  Saved point clouds to data/mug_hunyuan3d_v20.npy and v21.npy")

    # --- Visual comparison ---
    print("\nOpening visual comparison...")
    try:
        import open3d as o3d

        # v2.0 mesh
        verts_20_arr = np.array(result_v20['vertices'], dtype=np.float64)
        faces_20_arr = np.array(result_v20['faces'], dtype=np.int32)
        mesh_20 = o3d.geometry.TriangleMesh()
        mesh_20.vertices = o3d.utility.Vector3dVector(verts_20_arr)
        mesh_20.triangles = o3d.utility.Vector3iVector(faces_20_arr)
        mesh_20.compute_vertex_normals()

        # v2.1 mesh
        verts_21_arr = np.array(result_v21['vertices'], dtype=np.float64)
        faces_21_arr = np.array(result_v21['faces'], dtype=np.int32)
        mesh_21 = o3d.geometry.TriangleMesh()
        mesh_21.vertices = o3d.utility.Vector3dVector(verts_21_arr)
        mesh_21.triangles = o3d.utility.Vector3iVector(faces_21_arr)
        mesh_21.compute_vertex_normals()

        # Window 1: v2.0
        vis_20 = o3d.visualization.Visualizer()
        vis_20.create_window(
            f"v2.0 — {gpu_20:.1f}s, {verts_20:,} verts",
            width=600, height=600, left=50,
        )
        vis_20.add_geometry(mesh_20)
        vis_20.get_render_option().mesh_show_back_face = True

        # Window 2: v2.1
        vis_21 = o3d.visualization.Visualizer()
        vis_21.create_window(
            f"v2.1 — {gpu_21:.1f}s, {verts_21:,} verts",
            width=600, height=600, left=700,
        )
        vis_21.add_geometry(mesh_21)
        vis_21.get_render_option().mesh_show_back_face = True

        import cv2
        print("Window 1: v2.0 | Window 2: v2.1")
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

    except ImportError:
        print("  Open3D not installed locally — skipping visualization")
        print("  Compare point clouds manually: data/mug_hunyuan3d_v20.npy vs v21.npy")
