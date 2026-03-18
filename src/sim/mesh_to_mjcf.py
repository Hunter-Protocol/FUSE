"""Step 2: Convert mesh to MuJoCo XML via CoACD convex decomposition.

Takes an OBJ mesh, decomposes into convex hulls (for MuJoCo collision),
and generates an MJCF XML file with the object as a free body on a table.

Usage:
    python -m sim.mesh_to_mjcf --mesh data/meshes/mug_complete.obj --name mug_complete
    python -m sim.mesh_to_mjcf --mesh data/meshes/mug_partial.obj --name mug_partial
"""

import argparse
import os
import numpy as np
import trimesh


DATA_DIR = "/home/hunter/Desktop/FUSE/data"
MJCF_DIR = f"{DATA_DIR}/mjcf"


def decompose_mesh(mesh_path, threshold=0.05, max_convex_hull=32, target_height=None):
    """Run CoACD convex decomposition on a mesh.

    Args:
        mesh_path: path to OBJ file
        threshold: CoACD concavity threshold (lower = more parts, more accurate)
        max_convex_hull: max number of convex parts
        target_height: if set, scale the mesh so its height matches this (meters).
                       E.g., 0.10 for a 10cm mug.

    Returns:
        list of trimesh.Trimesh convex hulls, the recentered mesh
    """
    import coacd

    mesh = trimesh.load(mesh_path, force='mesh')
    print(f"Input mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    print(f"  Bounds: min={mesh.bounds[0]}, max={mesh.bounds[1]}")
    print(f"  Extents: {mesh.extents}")

    # Auto-upright: rotate mesh so tallest axis becomes Z
    tallest_axis = int(mesh.extents.argmax())
    if tallest_axis != 2:
        print(f"  Tallest axis is {'XYZ'[tallest_axis]} — rotating to stand upright")
        if tallest_axis == 0:  # X is tallest, rotate 90° around Y
            rot = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        else:  # Y is tallest, rotate -90° around X
            rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        mesh.apply_transform(rot)
        print(f"  After rotation extents: {mesh.extents}")

    # Recenter mesh: move centroid to origin, bottom on z=0
    centroid = mesh.centroid.copy()
    mesh.vertices -= centroid
    bottom_z = mesh.bounds[0][2]
    mesh.vertices[:, 2] -= bottom_z  # bottom sits at z=0

    # Scale to real-world size if target_height is given
    if target_height is not None:
        current_height = mesh.extents[2]  # z extent (now the tallest)
        scale = target_height / current_height
        mesh.vertices *= scale
        print(f"  Scaled by {scale:.4f} to target height {target_height}m")
        print(f"  New extents: {mesh.extents}")

    print(f"  Recentered: bottom at z=0")

    # CoACD expects specific format
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=threshold,
        max_convex_hull=max_convex_hull,
    )
    print(f"CoACD decomposition: {len(parts)} convex parts")

    hulls = []
    for i, (verts, faces) in enumerate(parts):
        hull = trimesh.Trimesh(vertices=verts, faces=faces)
        print(f"  Part {i}: {len(verts)} verts, {len(faces)} faces")
        hulls.append(hull)

    return hulls, mesh


def generate_mjcf(object_name, hulls, original_mesh, output_dir):
    """Generate MuJoCo XML with convex collision geoms.

    Creates:
        output_dir/
            {object_name}.xml     -- MJCF model
            meshes/
                {object_name}_visual.obj
                {object_name}_col_0.obj
                {object_name}_col_1.obj
                ...
    """
    mesh_subdir = os.path.join(output_dir, "meshes")
    os.makedirs(mesh_subdir, exist_ok=True)

    # Save visual mesh
    visual_path = os.path.join(mesh_subdir, f"{object_name}_visual.obj")
    original_mesh.export(visual_path)

    # Save collision hulls
    col_paths = []
    for i, hull in enumerate(hulls):
        col_path = os.path.join(mesh_subdir, f"{object_name}_col_{i}.obj")
        hull.export(col_path)
        col_paths.append(col_path)

    # Compute object properties — mesh is already recentered with bottom at z=0
    bbox = original_mesh.bounds
    size = bbox[1] - bbox[0]

    # Estimate mass (assume ~300g for a mug-sized object)
    volume = original_mesh.convex_hull.volume if original_mesh.is_watertight else np.prod(size) * 0.3
    density = 0.3 / max(volume, 1e-6)  # ~300g target
    mass = max(0.05, min(density * volume, 2.0))  # clamp 50g-2kg

    # Build MJCF XML
    mesh_assets = ""
    # Visual mesh
    mesh_assets += f'    <mesh name="{object_name}_visual" file="meshes/{object_name}_visual.obj" />\n'
    # Collision meshes
    for i in range(len(hulls)):
        mesh_assets += f'    <mesh name="{object_name}_col_{i}" file="meshes/{object_name}_col_{i}.obj" />\n'

    # Collision geoms
    collision_geoms = ""
    for i in range(len(hulls)):
        collision_geoms += f'      <geom type="mesh" mesh="{object_name}_col_{i}" class="collision" />\n'

    # Table top surface is at z=0. Object bottom is at z=0 in mesh coords.
    # Place object at z=0.001 (tiny gap to avoid initial penetration).
    obj_z = 0.001

    xml = f"""<mujoco model="{object_name}">
  <compiler angle="radian" meshdir="." />

  <option gravity="0 0 -9.81" timestep="0.002" />

  <default>
    <default class="collision">
      <geom type="mesh" contype="1" conaffinity="1" friction="0.8 0.02 0.01" rgba="0.6 0.6 0.6 0.3" />
    </default>
    <default class="visual">
      <geom type="mesh" contype="0" conaffinity="0" rgba="0.8 0.4 0.2 1" />
    </default>
  </default>

  <asset>
{mesh_assets}  </asset>

  <worldbody>
    <!-- Table: top surface at z=0 -->
    <body name="table" pos="0 0 -0.025">
      <geom type="box" size="0.3 0.3 0.025" rgba="0.4 0.3 0.2 1"
            contype="1" conaffinity="1" friction="0.8 0.02 0.01" />
    </body>

    <!-- Object: mesh recentered with bottom at z=0 -->
    <body name="{object_name}" pos="0 0 {obj_z:.4f}">
      <freejoint name="{object_name}_joint" />
      <inertial pos="0 0 {size[2]/2:.4f}" mass="{mass:.3f}"
                diaginertia="{mass*0.001:.6f} {mass*0.001:.6f} {mass*0.001:.6f}" />
      <geom type="mesh" mesh="{object_name}_visual" class="visual" />
{collision_geoms}    </body>
  </worldbody>
</mujoco>
"""

    xml_path = os.path.join(output_dir, f"{object_name}.xml")
    with open(xml_path, 'w') as f:
        f.write(xml)
    print(f"Saved MJCF: {xml_path}")
    return xml_path


def mesh_to_mjcf(mesh_path, object_name, threshold=0.05, max_convex_hull=32, target_height=None):
    """Full pipeline: mesh → CoACD → MJCF.

    Returns path to the generated MJCF XML.
    """
    output_dir = os.path.join(MJCF_DIR, object_name)
    os.makedirs(output_dir, exist_ok=True)

    hulls, original_mesh = decompose_mesh(mesh_path, threshold, max_convex_hull, target_height)
    xml_path = generate_mjcf(object_name, hulls, original_mesh, output_dir)

    return xml_path


def verify_mjcf(xml_path):
    """Quick verification: load the MJCF in MuJoCo and step a few times."""
    import mujoco

    print(f"\nVerifying {xml_path}...")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Step simulation
    for _ in range(100):
        mujoco.mj_step(model, data)

    # Check object didn't fall through table
    obj_z = data.qpos[2]  # z position of free joint
    print(f"  Object z after 100 steps: {obj_z:.4f}m")
    if obj_z > -0.1:
        print("  PASS: Object resting on table")
    else:
        print("  FAIL: Object fell through table")

    return obj_z > -0.1


def main():
    parser = argparse.ArgumentParser(description="Convert mesh to MuJoCo MJCF")
    parser.add_argument("--mesh", required=True, help="Path to OBJ mesh file")
    parser.add_argument("--name", required=True, help="Object name for output")
    parser.add_argument("--threshold", type=float, default=0.05, help="CoACD threshold")
    parser.add_argument("--max-hulls", type=int, default=32, help="Max convex hulls")
    parser.add_argument("--height", type=float, default=None,
                        help="Target object height in meters (e.g., 0.10 for 10cm mug)")
    parser.add_argument("--verify", action="store_true", help="Run MuJoCo verification")
    args = parser.parse_args()

    xml_path = mesh_to_mjcf(args.mesh, args.name, args.threshold, args.max_hulls, args.height)

    if args.verify:
        verify_mjcf(xml_path)

    print(f"\nDone. MJCF: {xml_path}")


if __name__ == "__main__":
    main()
