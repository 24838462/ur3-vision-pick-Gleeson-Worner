from pathlib import Path
import argparse
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from stacking import stack_cubes


ROOT = Path(__file__).resolve().parent
COMPLETE = ROOT / "models" / "ur3_complete.xml"

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

CUBE_COUNT = 3
CAMERA_FPS = 30
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


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

    # Read the table height from the compiled model.
    mujoco.mj_forward(model, data)

    # Top surface of the table in the current scene.
    table_top = -0.01

    # Randomise cube positions and orientations.
    rng = np.random.default_rng(seed)
    positions = []

    for index in range(CUBE_COUNT):
        cube_geom = get_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"cube_{index}_geom",
        )
        half_size = float(model.geom_size[cube_geom, 0])

        for _ in range(1000):
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
            xy[0],
            xy[1],
            table_top + half_size + 0.002,
        ]

        data.qpos[address + 3:address + 7] = [
            np.cos(yaw / 2),
            0,
            0,
            np.sin(yaw / 2),
        ]

    mujoco.mj_forward(model, data)

    sequence = stack_cubes(
        model,
        data,
        q_addresses,
        v_addresses,
        home,
    )

    desired_q = home.copy()
    desired_qd = np.zeros(6)
    gripper_command = 0.0
    frozen = False

    options = mujoco.MjvOption()
    options.geomgroup[3] = 0

    window = "Wrist camera"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, CAMERA_WIDTH, CAMERA_HEIGHT)

    dt = model.opt.timestep
    steps_per_frame = max(1, round(1 / (CAMERA_FPS * dt)))
    frame_seconds = steps_per_frame * dt

    print("Three movable cubes spawned.")
    print("Stacking starts automatically after settling.")
    print("Press Q in the camera window to exit.")

    try:
        with mujoco.Renderer(
            model,
            height=CAMERA_HEIGHT,
            width=CAMERA_WIDTH,
        ) as renderer:
            with mujoco.viewer.launch_passive(
                model,
                data,
            ) as viewer:
                with viewer.lock():
                    viewer.opt.geomgroup[3] = 0
                    viewer.cam.distance = 1.4
                    viewer.cam.lookat[:] = [0.2, 0, 0.15]

                while viewer.is_running():
                    frame_start = time.perf_counter()

                    if not frozen:
                        for _ in range(steps_per_frame):
                            try:
                                (
                                    desired_q,
                                    desired_qd,
                                    gripper_command,
                                ) = next(sequence)

                            except StopIteration:
                                frozen = True
                                print("Stacking sequence finished.")
                                break

                            except (RuntimeError, ValueError) as error:
                                frozen = True
                                print("STOPPED:", error)
                                break

                            torque = (
                                data.qfrc_bias[v_addresses]
                                + KP * (
                                    desired_q
                                    - data.qpos[q_addresses]
                                )
                                + KD * (
                                    desired_qd
                                    - data.qvel[v_addresses]
                                )
                            )

                            data.ctrl[motor_ids] = np.clip(
                                torque,
                                model.actuator_ctrlrange[motor_ids, 0],
                                model.actuator_ctrlrange[motor_ids, 1],
                            )

                            data.ctrl[gripper_id] = gripper_command

                            mujoco.mj_step(model, data)

                            error_deg = np.rad2deg(
                                desired_q - data.qpos[q_addresses]
                            )

                            if (
                                not np.all(np.isfinite(data.qpos))
                                or not np.all(np.isfinite(data.qvel))
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

                    if cv2.getWindowProperty(
                        window,
                        cv2.WND_PROP_VISIBLE,
                    ) < 1:
                        break

                    remaining = (
                        frame_seconds
                        - (time.perf_counter() - frame_start)
                    )

                    if remaining > 0:
                        time.sleep(remaining)

    except KeyboardInterrupt:
        print("\nSimulation stopped.")

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the UR3 cube-stacking simulation."
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not COMPLETE.exists():
        raise FileNotFoundError(
            "Compiled model not found. "
            "Run python .\\compile_scene.py first."
        )

    run_scene(args.seed)