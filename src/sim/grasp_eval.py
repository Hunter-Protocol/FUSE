"""Step 3: MuJoCo grasp evaluator — lift test for parallel-jaw grasps.

Given an object MJCF and a 6-DOF grasp pose (4x4 SE3), simulates:
  1. Position gripper at grasp pose
  2. Close fingers
  3. Lift 5cm
  4. Check if object came along

Uses a simple parallel-jaw gripper (two box geoms) rather than a full
robot model — minimal dependencies, fast simulation.

Usage:
    python -m sim.grasp_eval --mjcf data/mjcf/mug_complete/mug_complete.xml
"""

import argparse
import numpy as np
import mujoco


# Simple parallel-jaw gripper as inline MJCF
# Finger offset and range are parameterized to match object scale
GRIPPER_XML_TEMPLATE = """
<mujoco model="grasp_eval">
  <compiler angle="radian" />
  <option gravity="0 0 -9.81" timestep="0.002" />

  <default>
    <default class="gripper">
      <geom contype="1" conaffinity="1" friction="1.0 0.05 0.01" rgba="0.2 0.5 0.8 0.8" />
    </default>
  </default>

  {object_assets}

  <worldbody>
    <!-- Table -->
    <body name="table" pos="0 0 {table_z}">
      <geom type="box" size="0.3 0.3 0.025" rgba="0.4 0.3 0.2 1"
            contype="1" conaffinity="1" friction="0.8 0.02 0.01" />
    </body>

    <!-- Object -->
    {object_body}

    <!-- Gripper (mocap-controlled) -->
    <body name="gripper_base" mocap="true" pos="{gx} {gy} {gz}"
          quat="{gqw} {gqx} {gqy} {gqz}">
      <geom type="box" size="0.01 {finger_offset} 0.015" pos="0 0 0.04" rgba="0.3 0.3 0.3 0.5"
            contype="0" conaffinity="0" />

      <!-- Left finger -->
      <body name="finger_left" pos="0 {finger_offset} 0">
        <joint name="finger_left_slide" type="slide" axis="0 -1 0"
               range="0 {finger_range}" damping="10" />
        <geom type="box" size="0.008 0.004 0.025" class="gripper" />
      </body>

      <!-- Right finger -->
      <body name="finger_right" pos="0 -{finger_offset} 0">
        <joint name="finger_right_slide" type="slide" axis="0 1 0"
               range="0 {finger_range}" damping="10" />
        <geom type="box" size="0.008 0.004 0.025" class="gripper" />
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="finger_left_act" joint="finger_left_slide" kp="100" ctrlrange="0 {finger_range}" />
    <position name="finger_right_act" joint="finger_right_slide" kp="100" ctrlrange="0 {finger_range}" />
  </actuator>
</mujoco>
"""


def pose_to_pos_quat(grasp_pose_4x4):
    """Convert 4x4 SE3 matrix to position + quaternion (wxyz for MuJoCo)."""
    pos = grasp_pose_4x4[:3, 3]
    rot = grasp_pose_4x4[:3, :3]

    # Rotation matrix to quaternion (wxyz)
    from scipy.spatial.transform import Rotation
    r = Rotation.from_matrix(rot)
    quat_xyzw = r.as_quat()  # scipy returns xyzw
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    return pos, quat_wxyz


def build_scene_xml(object_mjcf_dir, object_name, grasp_pose_4x4):
    """Build a complete MuJoCo scene XML combining object + gripper.

    Instead of including the object MJCF, we read its mesh assets and
    inline them into a single XML for simplicity.
    """
    import os
    import trimesh

    # Parse grasp pose
    pos, quat = pose_to_pos_quat(grasp_pose_4x4)

    # Load visual mesh for sizing
    meshes_dir = os.path.join(object_mjcf_dir, "meshes")
    visual_mesh_path = os.path.join(meshes_dir, f"{object_name}_visual.obj")

    if os.path.exists(visual_mesh_path):
        mesh = trimesh.load(visual_mesh_path)
        obj_max_width = max(mesh.extents[0], mesh.extents[1])
    else:
        obj_max_width = 0.10

    # Gripper sizing: fingers start wider than object, close inward
    finger_offset = obj_max_width * 0.7  # start wider than object
    finger_range = finger_offset + 0.01  # can close past center

    # Table top at z=0 (object mesh already has bottom at z=0)
    table_z = -0.025

    # Build asset section from collision meshes
    asset_lines = []
    col_geom_lines = []
    mesh_files = sorted([f for f in os.listdir(meshes_dir) if f.startswith(f"{object_name}_col_")])

    for mf in mesh_files:
        mesh_name = mf.replace('.obj', '')
        full_path = os.path.join(meshes_dir, mf)
        asset_lines.append(
            f'    <mesh name="{mesh_name}" file="{full_path}" />'
        )
        col_geom_lines.append(
            f'      <geom type="mesh" mesh="{mesh_name}" '
            f'contype="1" conaffinity="1" friction="0.8 0.02 0.01" rgba="0.8 0.4 0.2 1" />'
        )

    # Visual mesh
    if os.path.exists(visual_mesh_path):
        asset_lines.append(
            f'    <mesh name="{object_name}_visual" file="{visual_mesh_path}" />'
        )

    object_assets = "<asset>\n" + "\n".join(asset_lines) + "\n  </asset>"

    # Object body at z=0.001 (just above table surface)
    if os.path.exists(visual_mesh_path):
        obj_height = mesh.extents[2]
        obj_inertia = 0.3 * (obj_max_width**2 + obj_height**2) / 12
    else:
        obj_inertia = 0.0003

    object_body = f"""<body name="object" pos="0 0 0.001">
      <freejoint name="object_joint" />
      <inertial pos="0 0 {mesh.extents[2]/2:.4f}" mass="0.3"
                diaginertia="{obj_inertia:.6f} {obj_inertia:.6f} {obj_inertia:.6f}" />
      <geom type="mesh" mesh="{object_name}_visual" contype="0" conaffinity="0" rgba="0.8 0.4 0.2 1" />
{chr(10).join(col_geom_lines)}
    </body>"""

    xml = GRIPPER_XML_TEMPLATE.format(
        object_assets=object_assets,
        object_body=object_body,
        table_z=f"{table_z:.4f}",
        gx=f"{pos[0]:.4f}", gy=f"{pos[1]:.4f}", gz=f"{pos[2]:.4f}",
        gqw=f"{quat[0]:.4f}", gqx=f"{quat[1]:.4f}",
        gqy=f"{quat[2]:.4f}", gqz=f"{quat[3]:.4f}",
        finger_offset=f"{finger_offset:.4f}",
        finger_range=f"{finger_range:.4f}",
    )

    return xml


def evaluate_grasp(object_mjcf_dir, object_name, grasp_pose_4x4,
                   close_time=0.5, lift_time=1.0, hold_time=1.0,
                   lift_height=0.05, success_threshold=0.04):
    """Run a single grasp trial in MuJoCo.

    Args:
        object_mjcf_dir: directory containing {object_name}.xml and meshes/
        object_name: name of the object
        grasp_pose_4x4: 4x4 SE3 grasp pose (gripper center in world frame)
        close_time: seconds to close fingers
        lift_time: seconds to lift
        hold_time: seconds to hold and verify
        lift_height: meters to lift
        success_threshold: object must be at least this high above start to count

    Returns:
        dict with 'success' (bool), 'final_z' (float), 'start_z' (float)
    """
    xml = build_scene_xml(object_mjcf_dir, object_name, grasp_pose_4x4)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    dt = model.opt.timestep
    steps_close = int(close_time / dt)
    steps_lift = int(lift_time / dt)
    steps_hold = int(hold_time / dt)

    # Let object settle on table first
    for _ in range(500):
        mujoco.mj_step(model, data)

    # Record object start position
    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    start_z = data.xpos[obj_body_id][2]

    # Get mocap body id for gripper
    gripper_mocap_id = model.body_mocapid[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
    ]
    gripper_start_pos = data.mocap_pos[gripper_mocap_id].copy()

    # Get max finger range from actuator
    finger_range = model.actuator_ctrlrange[0, 1]

    # Phase 1: Close fingers (move inward)
    data.ctrl[0] = finger_range  # left finger close
    data.ctrl[1] = finger_range  # right finger close
    for _ in range(steps_close):
        mujoco.mj_step(model, data)

    # Phase 2: Lift (keep fingers closed)
    for step in range(steps_lift):
        data.ctrl[0] = finger_range
        data.ctrl[1] = finger_range
        t = step / steps_lift
        data.mocap_pos[gripper_mocap_id][2] = gripper_start_pos[2] + lift_height * t
        mujoco.mj_step(model, data)

    # Phase 3: Hold
    for _ in range(steps_hold):
        data.ctrl[0] = finger_range
        data.ctrl[1] = finger_range
        mujoco.mj_step(model, data)

    # Check result
    final_z = data.xpos[obj_body_id][2]
    lift_delta = final_z - start_z
    success = lift_delta > success_threshold

    return {
        "success": success,
        "start_z": float(start_z),
        "final_z": float(final_z),
        "lift_delta": float(lift_delta),
    }


def evaluate_grasps_batch(object_mjcf_dir, object_name, grasp_poses, verbose=True):
    """Evaluate a batch of grasps and return success rate.

    Args:
        object_mjcf_dir: directory containing MJCF + meshes
        object_name: name of the object
        grasp_poses: list of 4x4 SE3 matrices
        verbose: print per-grasp results

    Returns:
        dict with 'success_rate', 'n_success', 'n_total', 'results'
    """
    results = []
    n_success = 0

    for i, pose in enumerate(grasp_poses):
        try:
            result = evaluate_grasp(object_mjcf_dir, object_name, pose)
            results.append(result)
            if result["success"]:
                n_success += 1
            if verbose:
                status = "PASS" if result["success"] else "FAIL"
                print(f"  Grasp {i:3d}: {status}  (lift={result['lift_delta']:.3f}m)")
        except Exception as e:
            if verbose:
                print(f"  Grasp {i:3d}: ERROR ({e})")
            results.append({"success": False, "error": str(e)})

    n_total = len(grasp_poses)
    rate = n_success / max(n_total, 1)

    return {
        "success_rate": rate,
        "n_success": n_success,
        "n_total": n_total,
        "results": results,
    }


def main():
    """Quick test: load an object MJCF and run a top-down grasp."""
    parser = argparse.ArgumentParser(description="Test grasp evaluator")
    parser.add_argument("--mjcf-dir", required=True, help="Directory with object MJCF")
    parser.add_argument("--name", required=True, help="Object name")
    args = parser.parse_args()

    # Simple top-down grasp at object center
    grasp_pose = np.eye(4)
    grasp_pose[2, 3] = 0.05  # 5cm above origin

    print(f"Testing top-down grasp on {args.name}...")
    result = evaluate_grasp(args.mjcf_dir, args.name, grasp_pose)
    print(f"  Result: {'SUCCESS' if result['success'] else 'FAIL'}")
    print(f"  Start z: {result['start_z']:.4f}, Final z: {result['final_z']:.4f}")
    print(f"  Lift delta: {result['lift_delta']:.4f}m")


if __name__ == "__main__":
    main()
