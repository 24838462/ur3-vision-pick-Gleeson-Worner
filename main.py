from pathlib import Path
import argparse
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from stacking import stack_cubes
from vision import CubeDetector, CameraFrames


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
TABLE_TOP = -0.01

# Lower rendering load for the laptop.
CAMERA_FPS = 15
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240


def get_id(model, object_type, name):
    index = mujoco.mj_name2id(model, object_type, name)

    if index < 0:
        raise ValueError(f"Missing model element: {name}")

    return index


def make_rgbd_display(rgb, depth):
    """Display colour and depth; preserve raw depth in metres."""
    colour = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Fixed depth colour scale: 5 cm to 1 metre.
    near = 0.05
    far = 1.0

    valid = np.isfinite(depth) & (depth > 0)
    safe_depth = np.where(valid, depth, far)

    scaled = np.clip(
        (safe_depth - near) / (far - near),
        0.0,
        1.0,
    )

    # Near surfaces appear warm; distant surfaces appear cool.
    intensity = ((1.0 - scaled) * 255).astype(np.uint8)
    depth_colour = cv2.applyColorMap(
        intensity,
        cv2.COLORMAP_TURBO,
    )
    depth_colour[~valid] = 0

    height, width = depth.shape
    centre = (width // 2, height // 2)

    for panel in (colour, depth_colour):
        cv2.drawMarker(
            panel,
            centre,
            (255, 255, 255),
            cv2.MARKER_CROSS,
            12,
            1,
        )

    centre_depth = depth[height // 2, width // 2]

    if np.isfinite(centre_depth) and centre_depth > 0:
        reading = f"Centre: {centre_depth:.3f} m"
    else:
        reading = "Centre: invalid"

    cv2.putText(
        colour,
        "RGB",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        depth_colour,
        "Depth: near = warm",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        depth_colour,
        reading,
        (10, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return np.hstack((colour, depth_colour))


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

    camera_id = get_id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        "wrist_camera",
    )

    detector = CubeDetector(
        CAMERA_WIDTH,
        CAMERA_HEIGHT,
        model.cam_fovy[camera_id],
    )

    camera_frames = CameraFrames(model, JOINT_NAMES)

    next_detection_report = 0.0

    home = np.deg2rad(HOME_DEG)
    data.qpos[q_addresses] = home

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
            TABLE_TOP + half_size + 0.002,
        ]

        data.qpos[address + 3:address + 7] = [
            np.cos(yaw / 2),
            0,
            0,
            np.sin(yaw / 2),
        ]

    mujoco.mj_forward(model, data)

    # Development baseline: stacking still uses simulator state.
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

    dt = model.opt.timestep
    steps_per_frame = max(1, round(1 / (CAMERA_FPS * dt)))
    frame_seconds = steps_per_frame * dt

    window = "Wrist camera: RGB and depth"

    print("Three movable cubes spawned.")
    print("Stacking starts automatically after settling.")
    print("RGB on the left; depth on the right.")
    print("Press Q in the camera window to exit.")

    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            window,
            CAMERA_WIDTH * 2,
            CAMERA_HEIGHT,
        )

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
                                or not np.all(np.isfinite(error_deg))
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

                    # Render RGB.
                    renderer.disable_depth_rendering()
                    renderer.update_scene(
                        data,
                        camera="wrist_camera",
                        scene_option=options,
                    )
                    rgb = renderer.render().copy()

                    # Render aligned depth from the same viewpoint.
                    renderer.enable_depth_rendering()
                    renderer.update_scene(
                        data,
                        camera="wrist_camera",
                        scene_option=options,
                    )
                    depth = renderer.render().copy()
                    renderer.disable_depth_rendering()

                    annotated_rgb, detections = detector.detect(
                        rgb,
                        depth,
                    )

                    camera_frames.update(data.sensordata)

                    for detection in detections.values():
                        detection["surface_base"] = (
                            camera_frames.point_to_base(
                                detection["surface_camera"]
                            )
                        )

                    combined = make_rgbd_display(
                        annotated_rgb,
                        depth,
                    )
                    cv2.imshow(window, combined)

                    if data.time >= next_detection_report:
                        next_detection_report = data.time + 1.0

                        for name, detection in detections.items():
                            point = detection["surface_base"]

                            print(
                                f"{name} surface in robot base (m): "
                                f"X={point[0]:.3f}, "
                                f"Y={point[1]:.3f}, "
                                f"Z={point[2]:.3f}"
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
        description="Run UR3 stacking with wrist RGB-D display."
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not COMPLETE.exists():
        raise FileNotFoundError(
            "Compiled model not found. "
            "Run python .\\compile_scene.py first."
        )

    run_scene(args.seed)