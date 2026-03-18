"""Visualize MuJoCo scenes — object on table, grasp attempts, etc.

Usage:
    python -m sim.visualize --mjcf data/mjcf/mug_complete/mug_complete.xml
    python -m sim.visualize --mjcf data/mjcf/mug_complete/mug_complete.xml --grasp top_down
"""

import argparse
import numpy as np
import mujoco
import mujoco.viewer


def view_object(xml_path):
    """Launch interactive MuJoCo viewer showing the object on a table."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Step a few times to let object settle
    for _ in range(500):
        mujoco.mj_step(model, data)

    print("Launching MuJoCo viewer...")
    print("  - Drag to rotate, scroll to zoom")
    print("  - Close window to exit")
    mujoco.viewer.launch(model, data)


def view_grasp_attempt(object_mjcf_dir, object_name, grasp_pose_4x4=None):
    """Launch interactive MuJoCo viewer showing a grasp attempt.

    If grasp_pose is None, uses a default top-down grasp.
    """
    from sim.grasp_eval import build_scene_xml

    if grasp_pose_4x4 is None:
        # Default: top-down grasp at object center, slightly above
        grasp_pose_4x4 = np.eye(4)
        # Read the object mesh to find center height
        import trimesh
        import os
        visual_path = os.path.join(object_mjcf_dir, "meshes", f"{object_name}_visual.obj")
        if os.path.exists(visual_path):
            mesh = trimesh.load(visual_path)
            center_z = (mesh.bounds[0][2] + mesh.bounds[1][2]) / 2
            grasp_pose_4x4[2, 3] = center_z + 0.001  # at object center height
        else:
            grasp_pose_4x4[2, 3] = 0.04

    xml = build_scene_xml(object_mjcf_dir, object_name, grasp_pose_4x4)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Let object settle
    for _ in range(500):
        mujoco.mj_step(model, data)

    print("Launching MuJoCo viewer with gripper...")
    print("  - Gripper is positioned at grasp pose")
    print("  - Close window to exit")
    mujoco.viewer.launch(model, data)


def simulate_grasp_with_viewer(object_mjcf_dir, object_name, grasp_pose_4x4=None):
    """Run a grasp simulation with live MuJoCo viewer.

    Shows the full grasp sequence: approach → close fingers → lift → hold.
    """
    from sim.grasp_eval import build_scene_xml, pose_to_pos_quat

    if grasp_pose_4x4 is None:
        grasp_pose_4x4 = np.eye(4)
        import trimesh, os
        visual_path = os.path.join(object_mjcf_dir, "meshes", f"{object_name}_visual.obj")
        if os.path.exists(visual_path):
            mesh = trimesh.load(visual_path)
            center_z = (mesh.bounds[0][2] + mesh.bounds[1][2]) / 2
            grasp_pose_4x4[2, 3] = center_z + 0.001
        else:
            grasp_pose_4x4[2, 3] = 0.04

    xml = build_scene_xml(object_mjcf_dir, object_name, grasp_pose_4x4)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    # Get body/actuator ids
    gripper_mocap_id = model.body_mocapid[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
    ]
    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")

    # Settle
    for _ in range(500):
        mujoco.mj_step(model, data)

    start_z = data.xpos[obj_body_id][2]
    gripper_start_pos = data.mocap_pos[gripper_mocap_id].copy()

    # Phase timings
    close_steps = int(0.5 / dt)
    lift_steps = int(1.0 / dt)
    hold_steps = int(1.0 / dt)
    lift_height = 0.05

    phase = "settle"
    step = 0

    def controller(model, data):
        nonlocal phase, step

        if phase == "settle":
            step += 1
            if step >= 100:
                phase = "close"
                step = 0
                print("Phase: CLOSE fingers")

        elif phase == "close":
            data.ctrl[0] = -0.005
            data.ctrl[1] = -0.005
            step += 1
            if step >= close_steps:
                phase = "lift"
                step = 0
                print("Phase: LIFT")

        elif phase == "lift":
            data.ctrl[0] = -0.005
            data.ctrl[1] = -0.005
            t = min(step / lift_steps, 1.0)
            data.mocap_pos[gripper_mocap_id][2] = gripper_start_pos[2] + lift_height * t
            step += 1
            if step >= lift_steps:
                phase = "hold"
                step = 0
                print("Phase: HOLD")

        elif phase == "hold":
            data.ctrl[0] = -0.005
            data.ctrl[1] = -0.005
            step += 1
            if step == hold_steps:
                final_z = data.xpos[obj_body_id][2]
                delta = final_z - start_z
                success = delta > 0.04
                print(f"\nResult: {'SUCCESS' if success else 'FAIL'}")
                print(f"  Lift delta: {delta:.4f}m")

    print("Launching MuJoCo viewer — grasp simulation")
    print("  Phase sequence: settle → close → lift → hold")
    print("  Watch the gripper close, lift, and hold the object")
    mujoco.viewer.launch(model, data)


def main():
    parser = argparse.ArgumentParser(description="MuJoCo scene visualizer")
    parser.add_argument("--mjcf", help="Path to MJCF XML file (for simple viewing)")
    parser.add_argument("--mjcf-dir", help="Path to MJCF directory (for grasp viewing)")
    parser.add_argument("--name", help="Object name")
    parser.add_argument("--grasp", choices=["view", "simulate"],
                        help="Show grasp: 'view' = static, 'simulate' = animated")
    args = parser.parse_args()

    if args.mjcf and not args.grasp:
        view_object(args.mjcf)
    elif args.mjcf_dir and args.name:
        if args.grasp == "simulate":
            simulate_grasp_with_viewer(args.mjcf_dir, args.name)
        else:
            view_grasp_attempt(args.mjcf_dir, args.name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
