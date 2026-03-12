"""Modal serverless endpoint: run InstantMesh image-to-3D on a cloud A100.

Deploy:   modal deploy cloud_instantmesh.py
Test:     modal run cloud_instantmesh.py
"""

import modal

# ---------- Modal setup ----------
app = modal.App("fuse-instantmesh")

instantmesh_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        "git", "libgl1-mesa-glx", "libglib2.0-0", "libgomp1",
        "build-essential", "libegl1-mesa-dev",
    )
    .pip_install(
        "torch==2.1.0", "torchvision==0.16.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy<2.0", "pillow", "trimesh", "scipy",
        "imageio", "imageio-ffmpeg", "einops", "omegaconf",
        "huggingface_hub", "safetensors",
        "pytorch-lightning==2.1.2", "torchmetrics",
        "transformers==4.34.1", "diffusers==0.20.2",
        "accelerate==0.25.0", "bitsandbytes",
        "xformers==0.0.22.post7", "ninja",
        "pymcubes", "xatlas", "plyfile", "rembg", "onnxruntime",
    )
    .env({"CUDA_HOME": "/usr/local/cuda", "CXX": "g++", "CC": "gcc",
          "TORCH_CUDA_ARCH_LIST": "8.0;8.6"})
    .pip_install("wheel", "setuptools")
    .run_commands(
        # nvdiffrast CUDA extension — build for both A100 (sm_80) and A10G (sm_86)
        "git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast && "
        "pip install --no-build-isolation /tmp/nvdiffrast",
        gpu="a10g",
    )
    .run_commands(
        # Clone InstantMesh repo
        "git clone https://github.com/TencentARC/InstantMesh.git /opt/instantmesh",
        # Pre-download model weights into the image
        "python -c \""
        "from huggingface_hub import hf_hub_download; "
        "hf_hub_download('TencentARC/InstantMesh', 'diffusion_pytorch_model.bin', repo_type='model'); "
        "hf_hub_download('TencentARC/InstantMesh', 'instant_mesh_large.ckpt', repo_type='model'); "
        "\"",
        "python -c \""
        "from huggingface_hub import snapshot_download; "
        "snapshot_download('sudo-ai/zero123plus-v1.2'); "
        "snapshot_download('sudo-ai/zero123plus-pipeline'); "
        "\"",
    )
    .env({"PYTHONPATH": "/opt/instantmesh"})
)


@app.cls(
    image=instantmesh_image,
    gpu="a100-80gb",
    timeout=300,
    scaledown_window=60,
)
class InstantMeshModel:
    @modal.enter()
    def load_model(self):
        """Load both stages: Zero123++ diffusion + InstantMesh reconstruction."""
        import sys
        import torch
        from huggingface_hub import hf_hub_download
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
        from omegaconf import OmegaConf

        sys.path.insert(0, "/opt/instantmesh")
        from src.utils.train_util import instantiate_from_config

        device = torch.device("cuda")

        # Stage 1: Zero123++ multiview diffusion
        print("Loading Zero123++ diffusion pipeline...")
        self.pipe = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2",
            custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch.float16,
        )
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
        )
        # Load fine-tuned UNet weights
        unet_path = hf_hub_download(
            repo_id="TencentARC/InstantMesh",
            filename="diffusion_pytorch_model.bin",
            repo_type="model",
        )
        self.pipe.unet.load_state_dict(
            torch.load(unet_path, map_location="cpu"), strict=True
        )
        self.pipe.to(device)
        print("Zero123++ loaded")

        # Stage 2: InstantMesh LRM reconstruction
        print("Loading InstantMesh reconstruction model...")
        config = OmegaConf.load("/opt/instantmesh/configs/instant-mesh-large.yaml")
        self.model = instantiate_from_config(config.model_config)
        model_path = hf_hub_download(
            repo_id="TencentARC/InstantMesh",
            filename="instant_mesh_large.ckpt",
            repo_type="model",
        )
        state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
        # Strip 'lrm_generator.' prefix from PL checkpoint keys
        state_dict = {k.replace("lrm_generator.", "", 1): v
                      for k, v in state_dict.items()
                      if k.startswith("lrm_generator.")}
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(device)
        self.model.eval()
        # FlexiCubes geometry is lazy-inited in PL — call explicitly
        self.model.init_flexicubes_geometry(device)
        self.config = config
        self.device = device
        print("InstantMesh loaded on GPU")

    @modal.method()
    def generate(self, image_bytes: bytes) -> dict:
        """Run image-to-3D: single image → 6 views → mesh.

        Args:
            image_bytes: PNG/JPEG image as bytes

        Returns:
            dict with 'vertices', 'faces', 'points', timing info
        """
        import io
        import sys
        import time
        import torch
        import numpy as np
        import trimesh
        from PIL import Image
        from torchvision import transforms

        sys.path.insert(0, "/opt/instantmesh")
        from src.utils.camera_util import get_zero123plus_input_cameras
        from src.utils.infer_util import remove_background, resize_foreground

        image = Image.open(io.BytesIO(image_bytes))
        print(f"Input image: {image.size}, mode: {image.mode}")

        # Preprocess: remove background if needed, resize
        if image.mode != "RGBA":
            image = remove_background(image)
        image = resize_foreground(image, ratio=0.85)
        # Composite onto white background for diffusion
        image_white = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image_white.paste(image, mask=image)
        image_input = image_white.convert("RGB")

        # Stage 1: Generate 6 multiview images
        t0 = time.time()
        mv_output = self.pipe(
            image_input,
            num_inference_steps=75,
        ).images[0]
        dt_mv = time.time() - t0
        print(f"Multiview diffusion: {dt_mv:.2f}s")

        # Parse the 3x2 grid output into 6 individual views (320x320 each)
        mv_array = np.array(mv_output)
        h_step = mv_array.shape[0] // 3
        w_step = mv_array.shape[1] // 2
        views = []
        for i in range(3):
            for j in range(2):
                view = mv_array[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                views.append(view)
        views = np.stack(views)  # (6, 320, 320, 3)

        # Convert to tensor
        images_tensor = torch.from_numpy(views).float().permute(0, 3, 1, 2) / 255.0
        images_tensor = images_tensor.unsqueeze(0).to(self.device)  # (1, 6, 3, 320, 320)

        # Stage 2: Reconstruct mesh from multiview
        t1 = time.time()
        input_cameras = get_zero123plus_input_cameras(batch_size=1).to(self.device)

        with torch.no_grad():
            planes = self.model.forward_planes(images_tensor, input_cameras)
            mesh_out = self.model.extract_mesh(
                planes,
                use_texture_map=False,
                **self.config.infer_config,
            )
        dt_recon = time.time() - t1
        print(f"Reconstruction: {dt_recon:.2f}s")

        # Extract vertices and faces
        if isinstance(mesh_out, tuple):
            vertices, faces, vertex_colors = mesh_out[0], mesh_out[1], mesh_out[2] if len(mesh_out) > 2 else None
        else:
            vertices, faces = mesh_out.vertices, mesh_out.faces
            vertex_colors = None

        if hasattr(vertices, 'cpu'):
            vertices = vertices.cpu()
        if hasattr(faces, 'cpu'):
            faces = faces.cpu()

        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
        print(f"Mesh: {len(vertices)} verts, {len(faces)} faces")

        # Sample points from mesh surface
        t_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        sampled_pts, _ = trimesh.sample.sample_surface(t_mesh, count=8192)
        sampled_pts = sampled_pts.astype(np.float32)

        dt_total = dt_mv + dt_recon
        print(f"Total generation: {dt_total:.2f}s")

        return {
            "vertices": vertices.tolist(),
            "faces": faces.tolist(),
            "points": sampled_pts.tolist(),
            "generation_time": dt_total,
            "multiview_time": dt_mv,
            "reconstruction_time": dt_recon,
            "num_vertices": len(vertices),
            "num_faces": len(faces),
        }


# ---------- Local test ----------
@app.local_entrypoint()
def main():
    """Test: send a local image to the cloud endpoint."""
    from pathlib import Path

    image_path = "/home/hunter/Desktop/FUSE/data/mug_crop_hunyuan.png"
    print(f"Sending {image_path} to cloud InstantMesh...")

    image_bytes = Path(image_path).read_bytes()
    model = InstantMeshModel()
    result = model.generate.remote(image_bytes)

    print(f"\nResults:")
    print(f"  Multiview diffusion: {result['multiview_time']:.2f}s")
    print(f"  Reconstruction: {result['reconstruction_time']:.2f}s")
    print(f"  Total generation: {result['generation_time']:.2f}s")
    print(f"  Mesh: {result['num_vertices']} verts, {result['num_faces']} faces")
    print(f"  Sampled points: {len(result['points'])}")

    # Save results locally
    import numpy as np
    pts = np.array(result['points'], dtype=np.float32)
    np.save("/home/hunter/Desktop/FUSE/data/mug_instantmesh_cloud.npy", pts)
    print(f"  Saved points to data/mug_instantmesh_cloud.npy")
