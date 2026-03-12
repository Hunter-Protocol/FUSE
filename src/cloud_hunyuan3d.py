"""Modal serverless endpoint: run Hunyuan3D-2 Full image-to-3D on a cloud A100.

Deploy:   modal deploy cloud_hunyuan3d.py
Test:     modal run cloud_hunyuan3d.py
"""

import modal

# ---------- Modal setup ----------
app = modal.App("fuse-hunyuan3d")

hunyuan3d_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "libgomp1", "build-essential")
    .pip_install(
        "torch==2.4.0", "torchvision==0.19.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy", "pillow", "trimesh", "scipy",
        "huggingface_hub", "safetensors", "einops",
        "tqdm", "ninja", "diffusers", "pybind11",
        "opencv-python-headless", "omegaconf",
        "transformers>=4.48.0", "accelerate",
        "pymeshlab", "pygltflib", "xatlas",
        "rembg", "onnxruntime",
    )
    .run_commands(
        # Clone and install Hunyuan3D-2
        "git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git /opt/hunyuan3d",
        "cd /opt/hunyuan3d && pip install -e .",
    )
    .run_commands(
        # Pre-download model weights into the image
        "python -c \""
        "from huggingface_hub import snapshot_download; "
        "snapshot_download('tencent/Hunyuan3D-2', allow_patterns=['hunyuan3d-dit-v2-0/*', 'hunyuan3d-vae-v2-0/*']); "
        "\"",
    )
    .env({"PYTHONPATH": "/opt/hunyuan3d"})
)


@app.cls(
    image=hunyuan3d_image,
    gpu="a100-80gb",
    timeout=600,
    scaledown_window=60,
)
class Hunyuan3DModel:
    @modal.enter()
    def load_model(self):
        """Load Hunyuan3D shape generation pipeline."""
        import torch
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        print("Loading Hunyuan3D-2 Full pipeline...")
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2",
            subfolder="hunyuan3d-dit-v2-0",
            device="cuda",
            dtype=torch.float16,
        )
        print("Hunyuan3D-2 Full loaded on GPU")

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
        }


# ---------- Local test ----------
@app.local_entrypoint()
def main():
    """Test: send a local image to the cloud endpoint."""
    from pathlib import Path

    image_path = "/home/hunter/Desktop/FUSE/data/mug_crop_hunyuan.png"
    print(f"Sending {image_path} to cloud Hunyuan3D-2 Full...")

    image_bytes = Path(image_path).read_bytes()
    model = Hunyuan3DModel()
    result = model.generate.remote(image_bytes)

    print(f"\nResults:")
    print(f"  Generation time: {result['generation_time']:.2f}s")
    print(f"  Mesh: {result['num_vertices']} verts, {result['num_faces']} faces")
    print(f"  Sampled points: {len(result['points'])}")

    # Save results locally
    import numpy as np
    pts = np.array(result['points'], dtype=np.float32)
    np.save("/home/hunter/Desktop/FUSE/data/mug_hunyuan3d_cloud.npy", pts)
    print(f"  Saved points to data/mug_hunyuan3d_cloud.npy")
