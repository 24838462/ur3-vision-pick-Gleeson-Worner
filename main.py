from pathlib import Path
import argparse
import time
import xml.etree.ElementTree as ET

import cv2
import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
SOURCE = MODELS / "ur3_pick_gripper_scene.xml"
COMPLETE = MODELS / "ur3_complete.xml"

HOME_DEG = [0, -90, 90, -90, -90, 0]

KP = np.array([80, 100, 80, 3, 3, 2], dtype=float)
KD = np.array([20, 25, 20, 0.15, 0.15, 0.08], dtype=float)

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

MOTOR_NAMES = [
    "shoulder_pan_motor",
    "shoulder_lift_motor",
    "elbow_motor",
    "wrist_1_motor",
    "wrist_2_motor",
    "wrist_3_motor",
]

# Cube half-sizes: full widths are 3, 4 and 5 cm.
HALF_SIZES = [0.015, 0.020, 0.025]
COLOURS = [
    "0.9 0.1 0.1 1",
    "0.1 0.8 0.2 1",
    "0.1 0.3 0.9 1",
]


def expand_includes(path):
    """Flatten the top-level includes used by our scene files."""
    root = ET.parse(path).getroot()

    for element in list(root):
        if element.tag != "include":
            continue

        included = expand_includes(
            path.parent / element.get("file")
        )
        index = list(root).index(element)
        root.remove(element)

        for offset, child in enumerate(list(included)):
            root.insert(index + offset, child)

    return root


def compile_scene():
    """Build one XML containing the robot, gripper and scene."""
    root = expand_includes(SOURCE)
    root.set("model", "ur3_complete")

    # Remove the original fixed cube and any previous demo cubes.
    for world in root.findall("worldbody"):
        for body in list(world.findall("body")):
            name = body.get("name", "")
            if name == "target_cube" or name.startswith("cube_"):
                world.remove(body)

    world = root.find("worldbody")
    if world is None:
        raise ValueError("Scene has no worldbody.")

    for index, half_size in enumerate(HALF_SIZES):
        body = ET.SubElement(
            world,
            "body",
            name=f"cube_{index}",
            pos=f"{0.25 + index * 0.06} 0.15 "
                f"{-0.01 + half_size + 0.002}",
        )

        ET.SubElement(
            body,
            "freejoint",
            name=f"cube_{index}_joint",
        )

        ET.SubElement(
            body,
            "geom",
            name=f"cube_{index}_geom",
            type="box",
            size=f"{half_size} {half_size} {half_size}",
            mass="0.05",
            rgba=COLOURS[index],
            friction="0.8 0.005 0.0001",
            contype="1",
            conaffinity="1",
        )

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(
        COMPLETE,
        encoding="utf-8",
        xml_declaration=True,
    )

    # Parse/compile and save a canonical combined XML.
    # This checks model loading; it does not run physics.
    model = mujoco.MjModel.from_xml_path(str(COMPLETE))
    mujoco.mj_saveLastXML(str(COMPLETE), model)

    print("Created:", COMPLETE)


def get_id(model, object_type, name):
    index = mujoco.mj_name2id(model, object_type, name)
    if index < 0:
        raise ValueError(f"Missing model element: {name}")
    return index


def run_scene(seed):
    model = mujoco.MjModel.from_xml_path(str(COMPLETE))
    data = mujoco.MjData(model)

    joint_ids = np.array([
        get_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in JOINT_NAMES
    ])
    q_addresses = model.jnt_qposadr[joint_ids]
    v_addresses = model.jnt_dofadr[joint_ids]

    motor_ids = np.array([
        get_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in MOTOR_NAMES
    ])
    gripper_id = get_id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "gripper_fingers_actuator",
    )

    home = np.deg2rad(HOME_DEG)
    data.qpos[q_addresses] = home

    # Randomise cube positions in a small tabletop region.
    rng = np.random.default_rng(seed)
    positions = []

    for index, half_size in enumerate(HALF_SIZES):
        for attempt in range(1000):
            xy = rng.uniform([0.22, 0.08], [0.38, 0.24])

            if all(
                np.linalg.norm(xy - previous) > 0.085
                for previous in positions
            ):
                break
        else:
            raise RuntimeError("Could not space the cubes apart.")

        positions.append(xy)

        cube_joint = get_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"cube_{index}_joint",
        )
        address = model.jnt_qposadr[cube_joint]
        yaw = rng.uniform(-np.pi, np.pi)

        data.qpos[address:address + 3] = [
            xy[0], xy[1], -0.01 + half_size + 0.002,
        ]
        data.qpos[address + 3:address + 7] = [
            np.cos(yaw / 2), 0, 0, np.sin(yaw / 2),
        ]

    mujoco.mj_forward(model, data)

    options = mujoco.MjvOption()
    options.geomgroup[3] = 0

    window = "Wrist camera"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 640, 480)

    gripper_command = 0.0
    frozen = False

    dt = model.opt.timestep
    steps_per_frame = max(1, round(1 / (30 * dt)))
    frame_seconds = steps_per_frame * dt

    print("Three movable cubes spawned.")
    print("Click the camera window before using keys.")
    print("O = open | C = close | Q = exit")
    print("Arm holds its home pose.")

    try:
        with mujoco.Renderer(
            model, height=480, width=640
        ) as renderer:
            with mujoco.viewer.launch_passive(
                model, data
            ) as viewer:
                with viewer.lock():
                    viewer.opt.geomgroup[3] = 0
                    viewer.cam.distance = 1.4
                    viewer.cam.lookat[:] = [0.2, 0, 0.15]

                while viewer.is_running():
                    frame_start = time.perf_counter()

                    if not frozen:
                        for _ in range(steps_per_frame):
                            torque = (
                                data.qfrc_bias[v_addresses]
                                + KP * (
                                    home - data.qpos[q_addresses]
                                )
                                - KD * data.qvel[v_addresses]
                            )

                            data.ctrl[motor_ids] = np.clip(
                                torque,
                                model.actuator_ctrlrange[
                                    motor_ids, 0
                                ],
                                model.actuator_ctrlrange[
                                    motor_ids, 1
                                ],
                            )
                            data.ctrl[gripper_id] = (
                                gripper_command
                            )

                            mujoco.mj_step(model, data)

                            error_deg = np.rad2deg(
                                home - data.qpos[q_addresses]
                            )
                            if (
                                not np.all(np.isfinite(data.qpos))
                                or np.max(np.abs(error_deg)) > 10
                            ):
                                frozen = True
                                print(
                                    "Physics paused: arm tracking "
                                    "error exceeded 10 degrees "
                                    "or state became invalid."
                                )
                                break

                    viewer.sync()

                    renderer.update_scene(
                        data,
                        camera="wrist_camera",
                        scene_option=options,
                    )
                    rgb = renderer.render()
                    cv2.imshow(
                        window,
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    )

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("o"):
                        gripper_command = 0.0
                    if key == ord("c"):
                        gripper_command = 255.0

                    if cv2.getWindowProperty(
                        window, cv2.WND_PROP_VISIBLE
                    ) < 1:
                        break

                    remaining = (
                        frame_seconds
                        - (time.perf_counter() - frame_start)
                    )
                    if remaining > 0:
                        time.sleep(remaining)

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.compile:
        compile_scene()
    else:
        if not COMPLETE.exists():
            raise FileNotFoundError(
                "Run python .\\main.py --compile first."
            )
        run_scene(args.seed)