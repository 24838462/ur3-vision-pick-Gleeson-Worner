from pathlib import Path
from collections import deque
import argparse
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from observation import WorkspaceScanner
from stacking import stack_cubes
from vision import (
    CubeDetector,
    CameraFrames,
    estimate_table_cube,
    estimate_placement_square,
)


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
CUBE_NAMES = ("red", "green", "blue")
TABLE_TOP = -0.01

CAMERA_FPS = 15
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

STABLE_FRAMES = 10

PLACEMENT_MIN = np.array([0.26, -0.18])
PLACEMENT_MAX = np.array([0.34, -0.12])


def get_id(model, object_type, name):
    index = mujoco.mj_name2id(
        model,
        object_type,
        name,
    )

    if index < 0:
        raise ValueError(f"Missing model element: {name}")

    return index


def make_rgbd_display(rgb, depth, status):
    colour = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )

    valid = np.isfinite(depth) & (depth > 0)
    safe_depth = np.where(valid, depth, 1.0)

    scaled = np.clip(
        (safe_depth - 0.05) / 0.95,
        0.0,
        1.0,
    )
    intensity = (
        (1.0 - scaled) * 255
    ).astype(np.uint8)

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

    for panel, label in (
        (colour, "RGB"),
        (depth_colour, "Depth: near = warm"),
    ):
        cv2.putText(
            panel,
            label,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        colour,
        status,
        (8, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    centre_depth = depth[height // 2, width // 2]

    if np.isfinite(centre_depth) and centre_depth > 0:
        reading = f"Centre: {centre_depth:.3f} m"
    else:
        reading = "Centre: invalid"

    cv2.putText(
        depth_colour,
        reading,
        (10, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return np.hstack((colour, depth_colour))


def stable_cube(history):
    """Record one cube independently of the other objects."""
    if len(history) < STABLE_FRAMES:
        return None

    centres = np.array([
        item["centre_base"] for item in history
    ])
    sizes = np.array([
        item["size"] for item in history
    ])
    angles = np.array([
        item["yaw"] for item in history
    ])

    if not (
        np.all(np.isfinite(centres))
        and np.all(np.isfinite(sizes))
        and np.all(np.isfinite(angles))
    ):
        return None

    centre = np.median(centres, axis=0)
    size = float(np.median(sizes))

    orientation_mean = np.mean(
        np.exp(4j * angles)
    )
    if abs(orientation_mean) < 0.95:
        return None

    yaw = float(
        np.angle(orientation_mean) / 4.0
    )

    angle_errors = (
        angles - yaw + np.pi / 4
    ) % (np.pi / 2) - np.pi / 4

    if np.max(np.linalg.norm(
        centres - centre,
        axis=1,
    )) > 0.002:
        return None

    if np.ptp(sizes) > 0.002:
        return None

    if np.max(np.abs(angle_errors)) > np.deg2rad(5):
        return None

    return {
        "centre_base": centre.copy(),
        "size": size,
        "yaw": yaw,
    }


def stable_placement(history):
    if len(history) < STABLE_FRAMES:
        return None

    centres = np.array([
        item["centre_base"] for item in history
    ])

    if not np.all(np.isfinite(centres)):
        return None

    centre = np.median(centres, axis=0)

    if np.max(np.linalg.norm(
        centres - centre,
        axis=1,
    )) > 0.002:
        return None

    return {"centre_base": centre.copy()}


def prepare_targets(records, placement, camera_frames):
    """Convert recorded measurements to world-frame movement targets."""
    kinematics = camera_frames.kinematics
    base_id = camera_frames.base_id

    rotation = (
        kinematics.xmat[base_id]
        .reshape(3, 3)
        .copy()
    )
    translation = kinematics.xpos[base_id].copy()

    catalog = []

    for name in CUBE_NAMES:
        record = records[name]

        direction = rotation @ np.array([
            np.cos(record["yaw"]),
            np.sin(record["yaw"]),
            0.0,
        ])

        catalog.append({
            "name": name,
            "size": record["size"],
            "centre_base": record["centre_base"].copy(),
            "yaw": record["yaw"],
            "centre_world": (
                rotation @ record["centre_base"]
                + translation
            ),
            "yaw_world": float(
                np.arctan2(
                    direction[1],
                    direction[0],
                )
            ),
        })

    catalog.sort(
        key=lambda cube: cube["size"],
        reverse=True,
    )

    for larger, smaller in zip(catalog, catalog[1:]):
        if larger["size"] - smaller["size"] < 0.003:
            raise RuntimeError(
                "Measured cube sizes are too close "
                "to determine a reliable stacking order."
            )

    placement_world = (
        rotation @ placement["centre_base"]
        + translation
    )

    return catalog, placement_world[:2].copy()


def run_scene(seed):
    model = mujoco.MjModel.from_xml_path(
        str(COMPLETE)
    )
    data = mujoco.MjData(model)

    joint_ids = np.array([
        get_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
        for name in JOINT_NAMES
    ])

    q_addresses = model.jnt_qposadr[joint_ids]
    v_addresses = model.jnt_dofadr[joint_ids]

    motor_ids = np.array([
        get_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
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
    placement_id = get_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "placement_square",
    )

    detector = CubeDetector(
        CAMERA_WIDTH,
        CAMERA_HEIGHT,
        model.cam_fovy[camera_id],
    )
    camera_frames = CameraFrames(
        model,
        JOINT_NAMES,
    )

    home = np.deg2rad(HOME_DEG)
    data.qpos[q_addresses] = home

    rng = np.random.default_rng(seed)
    positions = []

    # Simulator setup only. These values are not pickup targets.
    for index in range(CUBE_COUNT):
        geom_id = get_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"cube_{index}_geom",
        )
        half_size = float(
            model.geom_size[geom_id, 0]
        )

        for _ in range(1000):
            xy = rng.uniform(
                [0.22, 0.08],
                [0.38, 0.24],
            )

            if all(
                np.linalg.norm(xy - previous) > 0.085
                for previous in positions
            ):
                break
        else:
            raise RuntimeError(
                "Could not space the cubes apart."
            )

        positions.append(xy)

        joint_id = get_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"cube_{index}_joint",
        )
        address = model.jnt_qposadr[joint_id]
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

    # Simulator setup only. The task must discover this with vision.
    spawned_placement_xy = rng.uniform(
        PLACEMENT_MIN,
        PLACEMENT_MAX,
    )
    model.geom_pos[placement_id, :2] = (
        spawned_placement_xy
    )
    model.geom_pos[placement_id, 2] = (
        TABLE_TOP + 0.0002
    )

    mujoco.mj_forward(model, data)

    scanner = WorkspaceScanner(
        model,
        data,
        q_addresses,
        v_addresses,
        camera_id,
    )
    search_sequence = scanner.run()
    stack_sequence = None

    histories = {
        name: deque(maxlen=STABLE_FRAMES)
        for name in CUBE_NAMES
    }
    placement_history = deque(
        maxlen=STABLE_FRAMES
    )

    records = {}
    placement_record = None

    desired_q = home.copy()
    desired_qd = np.zeros(6)
    gripper_command = 0.0

    frozen = False
    next_report = 0.0

    options = mujoco.MjvOption()
    options.geomgroup[3] = 0

    dt = model.opt.timestep
    steps_per_frame = max(
        1,
        round(1 / (CAMERA_FPS * dt)),
    )
    frame_seconds = steps_per_frame * dt

    window = "Wrist camera: RGB and depth"

    print("Three cubes and a purple marker spawned.")
    print("Searching the workspace using the wrist camera.")
    print("Objects can be recorded from different views.")
    print("Press Q in the camera window to exit.")

    try:
        cv2.namedWindow(
            window,
            cv2.WINDOW_NORMAL,
        )
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
                            active_sequence = (
                                stack_sequence
                                if stack_sequence is not None
                                else search_sequence
                            )

                            if active_sequence is not None:
                                try:
                                    (
                                        desired_q,
                                        desired_qd,
                                        gripper_command,
                                    ) = next(active_sequence)

                                except StopIteration:
                                    desired_qd = np.zeros(6)

                                    if stack_sequence is not None:
                                        frozen = True
                                        print("Stacking finished.")
                                        break

                                    search_sequence = None

                                except (
                                    RuntimeError,
                                    ValueError,
                                ) as error:
                                    frozen = True
                                    print("STOPPED:", error)
                                    break

                            else:
                                desired_qd = np.zeros(6)
                                gripper_command = 0.0

                            # Development controller.
                            # Observer-based feedback is still to come.
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
                                desired_q
                                - data.qpos[q_addresses]
                            )

                            if (
                                not np.all(np.isfinite(data.qpos))
                                or not np.all(np.isfinite(data.qvel))
                                or not np.all(np.isfinite(error_deg))
                                or np.max(np.abs(error_deg)) > 10
                            ):
                                frozen = True
                                print(
                                    "Physics paused: tracking error "
                                    "or invalid state."
                                )
                                break

                    viewer.sync()

                    renderer.disable_depth_rendering()
                    renderer.update_scene(
                        data,
                        camera="wrist_camera",
                        scene_option=options,
                    )
                    rgb = renderer.render().copy()

                    renderer.enable_depth_rendering()
                    renderer.update_scene(
                        data,
                        camera="wrist_camera",
                        scene_option=options,
                    )
                    depth = renderer.render().copy()
                    renderer.disable_depth_rendering()

                    annotated_rgb, detections = (
                        detector.detect(rgb, depth)
                    )
                    camera_frames.update(
                        data.sensordata
                    )

                    if stack_sequence is None and not frozen:
                        if scanner.collecting and data.time >= 2.0:
                            for name in CUBE_NAMES:
                                if name in records:
                                    continue

                                measurement = None

                                if name in detections:
                                    measurement = estimate_table_cube(
                                        detections[name],
                                        camera_frames,
                                        TABLE_TOP,
                                    )

                                if measurement is None:
                                    histories[name].clear()
                                    continue

                                histories[name].append({
                                    "centre_base": (
                                        measurement["centre_base"].copy()
                                    ),
                                    "size": float(measurement["size"]),
                                    "yaw": float(measurement["yaw"]),
                                })

                                record = stable_cube(
                                    histories[name]
                                )

                                if record is not None:
                                    records[name] = record
                                    print(
                                        f"Recorded {name}: "
                                        f"{record['size'] * 1000:.1f} mm, "
                                        f"centre="
                                        f"{np.round(record['centre_base'], 3)}"
                                    )

                            if placement_record is None:
                                measurement = None

                                if "purple" in detections:
                                    measurement = (
                                        estimate_placement_square(
                                            detections["purple"],
                                            camera_frames,
                                            TABLE_TOP,
                                        )
                                    )

                                if measurement is None:
                                    placement_history.clear()

                                else:
                                    placement_history.append({
                                        "centre_base": (
                                            measurement["centre_base"].copy()
                                        )
                                    })

                                    placement_record = stable_placement(
                                        placement_history
                                    )

                                    if placement_record is not None:
                                        print(
                                            "Recorded purple centre "
                                            "from camera:",
                                            np.round(
                                                placement_record[
                                                    "centre_base"
                                                ],
                                                3,
                                            ),
                                        )

                        else:
                            # Do not combine samples across moving views.
                            for history in histories.values():
                                history.clear()
                            placement_history.clear()

                        ready = (
                            len(records) == CUBE_COUNT
                            and placement_record is not None
                        )

                        if ready and scanner.collecting:
                            try:
                                catalog, measured_placement_xy = (
                                    prepare_targets(
                                        records,
                                        placement_record,
                                        camera_frames,
                                    )
                                )

                                print(
                                    "Picking order: "
                                    + " -> ".join(
                                        cube["name"]
                                        for cube in catalog
                                    )
                                )
                                print(
                                    "Camera-derived placement XY:",
                                    np.round(
                                        measured_placement_xy,
                                        3,
                                    ),
                                )

                                stack_sequence = stack_cubes(
                                    model,
                                    data,
                                    q_addresses,
                                    v_addresses,
                                    desired_q.copy(),
                                    catalog,
                                    measured_placement_xy,
                                )

                                search_sequence = None
                                scanner.collecting = False
                                scanner.status = "Stacking"

                            except (
                                RuntimeError,
                                ValueError,
                            ) as error:
                                frozen = True
                                print("STOPPED:", error)

                        elif search_sequence is None:
                            missing = [
                                name
                                for name in CUBE_NAMES
                                if name not in records
                            ]

                            if placement_record is None:
                                missing.append("purple")

                            frozen = True
                            print(
                                "SEARCH INCOMPLETE: "
                                + ", ".join(missing)
                                + " not recorded."
                            )
                            print(
                                "Holding for inspection. "
                                "Search views or detection checks "
                                "need adjustment."
                            )

                        if data.time >= next_report:
                            next_report = data.time + 2.0
                            known = list(sorted(records))

                            if placement_record is not None:
                                known.append("purple")

                            print(
                                f"{scanner.status}; recorded: "
                                + (
                                    ", ".join(known)
                                    if known
                                    else "none"
                                )
                            )

                    status = (
                        "Paused"
                        if frozen
                        else (
                            "Stacking"
                            if stack_sequence is not None
                            else scanner.status
                        )
                    )

                    cv2.imshow(
                        window,
                        make_rgbd_display(
                            annotated_rgb,
                            depth,
                            status,
                        ),
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
        description="Search, measure and stack using wrist RGB-D."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    if not COMPLETE.exists():
        raise FileNotFoundError(
            "Run python .\\compile_scene.py first."
        )

    run_scene(args.seed)