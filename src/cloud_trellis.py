"""Modal serverless endpoint: run TRELLIS image-to-3D on a cloud A100.

Deploy:   modal deploy cloud_trellis.py
Test:     modal run cloud_trellis.py
"""

import modal

# ---------- Modal setup ----------
app = modal.App("fuse-trellis")

# Build the container image with all TRELLIS dependencies
# Use NVIDIA CUDA devel base for CUDA toolkit (nvcc, headers) needed to compile extensions
trellis_image = (
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
        "imageio", "imageio-ffmpeg", "xformers==0.0.27.post2",
        "huggingface_hub", "safetensors", "einops",
        # TRELLIS --basic deps (from setup.sh)
        "easydict", "tqdm", "opencv-python-headless", "ninja",
        "rembg", "onnxruntime", "xatlas", "igraph", "transformers",
        "open3d",
    )
    .pip_install("git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8")
    .run_commands(
        "pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html",
        "pip install spconv-cu120",
        # Clone TRELLIS
        "git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git /opt/trellis",
    )
    .env({"CUDA_HOME": "/usr/local/cuda", "CXX": "g++", "CC": "gcc"})
    .pip_install("wheel", "setuptools")
    .run_commands(
        # Build CUDA extensions that need GPU compilation
        "git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast && "
        "pip install --no-build-isolation /tmp/nvdiffrast",
        "git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/diffoctreerast && "
        "pip install /tmp/diffoctreerast",
        "git clone https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting && "
        "pip install /tmp/mip-splatting/submodules/diff-gaussian-rasterization/",
        gpu="a10g",
    )
    .env({
        "SPCONV_ALGO": "native",
        "ATTN_BACKEND": "xformers",
        "PYTHONPATH": "/opt/trellis",
    })
)


@app.cls(
    image=trellis_image,
    gpu="a100-80gb",
    timeout=300,
    scaledown_window=60,
)
class TrellisModel:
    @modal.enter()
    def load_model(self):
        """Load TRELLIS pipeline once when container starts."""
        import sys
        sys.path.insert(0, "/opt/trellis")
        from trellis.pipelines import TrellisImageTo3DPipeline
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(
            "microsoft/TRELLIS-image-large"
        )
        self.pipeline.cuda()
        print("TRELLIS model loaded on GPU")

    @modal.method()
    def generate(self, image_bytes: bytes) -> dict:
        """Run image-to-3D generation.

        Args:
            image_bytes: PNG/JPEG image as bytes

        Returns:
            dict with 'vertices', 'faces', 'points' as lists
        """
        import io
        import time
        import numpy as np
        import trimesh
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        print(f"Input image: {image.size}, mode: {image.mode}")

        t0 = time.time()
        outputs = self.pipeline.run(
            image,
            seed=42,
            sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
            slat_sampler_params={"steps": 12, "cfg_strength": 3},
        )
        dt_gen = time.time() - t0
        print(f"Generation: {dt_gen:.2f}s")

        # Extract mesh (vertices/faces may be CUDA tensors)
        mesh = outputs['mesh'][0]
        verts = mesh.vertices
        fcs = mesh.faces
        if hasattr(verts, 'cpu'):
            verts = verts.cpu()
        if hasattr(fcs, 'cpu'):
            fcs = fcs.cpu()
        vertices = np.array(verts, dtype=np.float32)
        faces = np.array(fcs, dtype=np.int32)
        print(f"Mesh: {len(vertices)} verts, {len(faces)} faces")

        # Sample points from mesh surface
        t_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        sampled_pts, _ = trimesh.sample.sample_surface(t_mesh, count=8192)
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
    print(f"Sending {image_path} to cloud TRELLIS...")

    image_bytes = Path(image_path).read_bytes()
    model = TrellisModel()
    result = model.generate.remote(image_bytes)

    print(f"\nResults:")
    print(f"  Generation time: {result['generation_time']:.2f}s")
    print(f"  Mesh: {result['num_vertices']} verts, {result['num_faces']} faces")
    print(f"  Sampled points: {len(result['points'])}")

    # Save results locally
    import numpy as np
    pts = np.array(result['points'], dtype=np.float32)
    np.save("/home/hunter/Desktop/FUSE/data/mug_trellis_cloud.npy", pts)
    print(f"  Saved points to data/mug_trellis_cloud.npy")
