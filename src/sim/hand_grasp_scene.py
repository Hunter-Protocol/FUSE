"""Build and visualize a LEAP Hand + object grasping scene in MuJoCo.

Creates a scene with the LEAP Hand positioned above the object on a table.
The hand is mocap-controlled for positioning, with actuated finger joints.

Usage:
    python -m sim.hand_grasp_scene --mjcf-dir data/mjcf/mug_complete --name mug_complete
    python -m sim.hand_grasp_scene --mjcf-dir data/mjcf/mug_complete --name mug_complete --close
"""

import argparse
import os
import numpy as np
import mujoco
import mujoco.viewer

MENAGERIE = "/home/hunter/Desktop/FUSE/third_party/mujoco_menagerie"
LEAP_HAND_DIR = os.path.join(MENAGERIE, "leap_hand")


def build_hand_scene_xml(object_mjcf_dir, object_name, hand_pos=(0, 0, 0.25)):
    """Build XML with LEAP Hand + object on table.

    Reads the LEAP Hand XML and modifies the palm position so the hand
    is properly positioned above the object.
    """
    meshes_dir = os.path.join(object_mjcf_dir, "meshes")

    # Collect object collision meshes
    asset_lines = []
    col_geom_lines = []
    mesh_files = sorted([f for f in os.listdir(meshes_dir) if f.startswith(f"{object_name}_col_")])

    for mf in mesh_files:
        mesh_name = mf.replace('.obj', '')
        full_path = os.path.abspath(os.path.join(meshes_dir, mf))
        asset_lines.append(f'    <mesh name="{mesh_name}" file="{full_path}" />')
        col_geom_lines.append(
            f'        <geom type="mesh" mesh="{mesh_name}" '
            f'contype="1" conaffinity="1" friction="0.8 0.02 0.01" rgba="0.8 0.4 0.2 0.3" group="3" />')

    # Visual mesh
    visual_path = os.path.abspath(os.path.join(meshes_dir, f"{object_name}_visual.obj"))
    if os.path.exists(visual_path):
        asset_lines.append(f'    <mesh name="{object_name}_visual" file="{visual_path}" />')

    # Get object height for inertia
    import trimesh
    mesh = trimesh.load(visual_path) if os.path.exists(visual_path) else None
    obj_height = mesh.extents[2] if mesh else 0.10
    obj_inertia = 0.3 * (0.10**2 + obj_height**2) / 12

    object_assets = "\n".join(asset_lines)
    collision_geoms = "\n".join(col_geom_lines)

    # Create a modified copy of the LEAP hand XML with adjusted palm position
    leap_hand_path = os.path.join(LEAP_HAND_DIR, "right_hand.xml")
    with open(leap_hand_path) as f:
        hand_xml = f.read()

    hx, hy, hz = hand_pos
    # Original quat "0 1 0 0" (180° Rx) makes palm face UP.
    # Use identity quat so palm faces DOWN (toward the mug).
    # Add freejoint so we can move the palm during simulation.
    hand_xml = hand_xml.replace(
        '<body name="palm" pos="0 0 0.1" quat="0 1 0 0">',
        f'<body name="palm" pos="{hx} {hy} {hz}" quat="1 0 0 0">\n'
        f'      <freejoint name="palm_joint"/>'
    )

    # Write modified hand XML to temp file
    modified_hand_path = os.path.join(LEAP_HAND_DIR, "_modified_right_hand.xml")
    with open(modified_hand_path, 'w') as f:
        f.write(hand_xml)

    # Place object under the hand's finger area
    # LEAP hand: palm center at (hx, hy, hz), fingers extend in -x direction
    obj_x = hx - 0.05
    obj_y = hy - 0.04
    table_surface_z = 0.025
    obj_z = table_surface_z + 0.001

    xml = f"""<mujoco model="hand_grasp_scene">
  <compiler angle="radian" meshdir="." />
  <option gravity="0 0 -9.81" timestep="0.002" />

  <statistic center="{obj_x:.2f} {obj_y:.2f} {hz/2:.2f}" extent="0.3"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="35" elevation="-50"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
{object_assets}
  </asset>

  <include file="{modified_hand_path}"/>

  <!-- Weld palm to mocap target so we can move the hand -->
  <equality>
    <weld body1="palm" body2="palm_target" solref="0.01 1" solimp="0.9 0.95 0.001"/>
  </equality>

  <worldbody>
    <light pos="0 0 1" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

    <!-- Table -->
    <body name="table" pos="{obj_x:.3f} {obj_y:.3f} 0">
      <geom type="box" size="0.15 0.15 0.025" rgba="0.4 0.3 0.2 1"
            contype="1" conaffinity="1" friction="0.8 0.02 0.01" />
    </body>

    <!-- Mocap target for hand palm -->
    <body name="palm_target" mocap="true" pos="{hx} {hy} {hz}" quat="1 0 0 0">
      <geom type="sphere" size="0.005" rgba="1 0 0 0.3" contype="0" conaffinity="0" />
    </body>

    <!-- Object on table -->
    <body name="object" pos="{obj_x:.4f} {obj_y:.4f} {obj_z:.4f}">
      <freejoint name="object_joint" />
      <inertial pos="0 0 {obj_height/2:.4f}" mass="0.3"
                diaginertia="{obj_inertia:.6f} {obj_inertia:.6f} {obj_inertia:.6f}" />
      <geom type="mesh" mesh="{object_name}_visual" contype="0" conaffinity="0" rgba="0.8 0.4 0.2 1" />
{collision_geoms}
    </body>
  </worldbody>
</mujoco>
"""
    return xml


def load_scene(object_mjcf_dir, object_name):
    """Load the hand+object MuJoCo scene. Returns (model, data)."""
    xml = build_hand_scene_xml(object_mjcf_dir, object_name)

    tmp_path = os.path.join(LEAP_HAND_DIR, "_tmp_scene.xml")
    with open(tmp_path, 'w') as f:
        f.write(xml)
    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.remove(tmp_path)
        modified_path = os.path.join(LEAP_HAND_DIR, "_modified_right_hand.xml")
        if os.path.exists(modified_path):
            os.remove(modified_path)

    data = mujoco.MjData(model)
    return model, data


def get_finger_close_targets(model):
    """Get actuator control values that curl all fingers inward."""
    targets = np.zeros(model.nu)
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name is None:
            continue
        ctrl_range = model.actuator_ctrlrange[i]
        # Curl fingers: use ~70% of max range for a power grasp
        if 'pip' in name or 'dip' in name:
            targets[i] = ctrl_range[1] * 0.8  # strong curl
        elif 'mcp' in name:
            targets[i] = ctrl_range[1] * 0.6  # moderate flex
        elif 'rot' in name:
            targets[i] = 0.0  # keep rotation neutral
        elif 'cmc' in name or 'axl' in name:
            targets[i] = ctrl_range[1] * 0.5  # thumb opposition
        elif 'ipl' in name:
            targets[i] = ctrl_range[1] * 0.7  # thumb curl
    return targets


def animate_grasp(object_mjcf_dir, object_name):
    """Run the full grasp sequence with MuJoCo passive viewer.

    Sequence: settle → lower hand → close fingers → lift → hold
    """
    model, data = load_scene(object_mjcf_dir, object_name)
    dt = model.opt.timestep

    # Find mocap and object body IDs
    palm_target_mocap_id = model.body_mocapid[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm_target")
    ]
    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")

    # Get object center height (for lowering target)
    import trimesh
    meshes_dir = os.path.join(object_mjcf_dir, "meshes")
    visual_path = os.path.join(meshes_dir, f"{object_name}_visual.obj")
    if os.path.exists(visual_path):
        mesh = trimesh.load(visual_path)
        obj_grasp_height = mesh.extents[2] * 0.5  # grasp at mid-height
    else:
        obj_grasp_height = 0.05

    # Phase timings
    settle_steps = int(0.5 / dt)
    lower_steps = int(1.5 / dt)
    close_steps = int(1.0 / dt)
    lift_steps = int(1.5 / dt)
    hold_steps = int(2.0 / dt)

    # Get finger close targets
    finger_targets = get_finger_close_targets(model)

    # Record start position
    start_pos = data.mocap_pos[palm_target_mocap_id].copy()
    # Compute lowered position: hand goes down to object grasp height
    # Table surface at z=0.025, object bottom at ~0.026
    grasp_z = 0.026 + obj_grasp_height + 0.02  # slight offset above object center
    lower_pos = start_pos.copy()
    lower_pos[2] = grasp_z

    # Lift position
    lift_pos = lower_pos.copy()
    lift_pos[2] = lower_pos[2] + 0.10  # lift 10cm

    step = 0
    phase = "settle"
    obj_start_z = None

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Grasp sequence starting...")
        while viewer.is_running():
            if phase == "settle":
                step += 1
                if step >= settle_steps:
                    phase = "lower"
                    step = 0
                    obj_start_z = data.xpos[obj_body_id][2]
                    print("Phase: LOWER hand to object")

            elif phase == "lower":
                t = min(step / lower_steps, 1.0)
                # Smooth interpolation
                t_smooth = t * t * (3 - 2 * t)  # smoothstep
                data.mocap_pos[palm_target_mocap_id] = start_pos + (lower_pos - start_pos) * t_smooth
                step += 1
                if step >= lower_steps:
                    phase = "close"
                    step = 0
                    print("Phase: CLOSE fingers")

            elif phase == "close":
                t = min(step / close_steps, 1.0)
                # Gradually close fingers
                for i in range(model.nu):
                    data.ctrl[i] = finger_targets[i] * t
                step += 1
                if step >= close_steps:
                    phase = "lift"
                    step = 0
                    print("Phase: LIFT")

            elif phase == "lift":
                # Keep fingers closed
                for i in range(model.nu):
                    data.ctrl[i] = finger_targets[i]
                t = min(step / lift_steps, 1.0)
                t_smooth = t * t * (3 - 2 * t)
                data.mocap_pos[palm_target_mocap_id] = lower_pos + (lift_pos - lower_pos) * t_smooth
                step += 1
                if step >= lift_steps:
                    phase = "hold"
                    step = 0
                    print("Phase: HOLD")

            elif phase == "hold":
                for i in range(model.nu):
                    data.ctrl[i] = finger_targets[i]
                step += 1
                if step == int(0.5 / dt):  # check after 0.5s of holding
                    final_z = data.xpos[obj_body_id][2]
                    delta = final_z - obj_start_z
                    success = delta > 0.04
                    print(f"\nResult: {'SUCCESS' if success else 'FAIL'}")
                    print(f"  Object lift: {delta:.3f}m", flush=True)

            mujoco.mj_step(model, data)
            viewer.sync()

    print("Viewer closed.")


def view_hand_scene(object_mjcf_dir, object_name):
    """Launch static MuJoCo viewer with hand above object."""
    model, data = load_scene(object_mjcf_dir, object_name)

    for _ in range(1000):
        mujoco.mj_step(model, data)

    print("Launching MuJoCo viewer with LEAP Hand + object")
    mujoco.viewer.launch(model, data)


def main():
    parser = argparse.ArgumentParser(description="LEAP Hand + object scene")
    parser.add_argument("--mjcf-dir", required=True, help="Object MJCF directory")
    parser.add_argument("--name", required=True, help="Object name")
    parser.add_argument("--grasp", action="store_true",
                        help="Animate full grasp: lower → close → lift → hold")
    args = parser.parse_args()

    if args.grasp:
        animate_grasp(args.mjcf_dir, args.name)
    else:
        view_hand_scene(args.mjcf_dir, args.name)


if __name__ == "__main__":
    main()
