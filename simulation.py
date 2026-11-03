"""Scene setup, physics, RGB-D perception, control and evaluation logs."""

from pathlib import Path
from types import SimpleNamespace
from collections import deque
import csv
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from robot_utils import JOINT_NAMES, MOTOR_NAMES, get_id
from observation import WorkspaceScanner
from stacking import stack_cubes
from estimation import ObserverMonitor
from estimation_plotter import EstimationPlotter
from torque_controller import ComputedTorqueController
from safety_logger import SafetyLogger

from safety_monitor import (
    StationarySafetyCamera,
    InteractiveObstacle,
    SafetyStatus,
    ObjectTrackingKalmanFilter,
    VisualServoingAvoidanceController,
    CubeCarrySafetySupervisor,
    detect_cyan_intruder_wrist,
)

from vision import (
    CubeDetector,
    CameraFrames,
    estimate_table_cube,
    estimate_placement_square,
    estimate_support_square,
    STABLE_FRAMES,
    stable_cube,
    stable_marker,
    world_target,
)


ROOT = Path(__file__).resolve().parent
COMPLETE = ROOT / "models" / "ur3_complete.xml"

HOME_DEG = [0, -90, 90, -90, -90, 0]
KP = np.array([80, 100, 80, 3, 3, 2], dtype=float)
KD = np.array([20, 25, 20, 0.15, 0.15, 0.08], dtype=float)
USE_LQR_TORQUE = True

CUBE_NAMES = ("red", "green", "blue")
TABLE_TOP = -0.01

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 15


def depth_to_colormap(depth):
    valid = np.isfinite(depth) & (depth > 0)
    safe_depth = np.where(valid, depth, 1.0)
    scaled = np.clip((safe_depth - 0.05) / 0.95, 0.0, 1.0)
    panel = cv2.applyColorMap(
        ((1 - scaled) * 255).astype(np.uint8), cv2.COLORMAP_TURBO,
    )
    panel[~valid] = 0
    return panel


def display_dashboard(
    wrist_rgb, wrist_depth, wrist_status,
    safety_rgb, safety_depth, safety_camera, obstacle_active,
    wrist_det=None, servo_telemetry=None,
    safety_status=SafetyStatus.NORMAL, wrist_kf=None,
):
    colour = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)
    wrist_panel = depth_to_colormap(wrist_depth)

    if wrist_det is not None:
        bx, by, bw, bh = wrist_det["bbox"]
        u, v = wrist_det["pixel"]
        cv2.rectangle(
            colour, (bx, by), (bx + bw, by + bh), (255, 255, 0), 2,
        )
        cv2.circle(colour, (u, v), 4, (0, 255, 255), -1)
        cv2.putText(
            colour, f"Obstacle: {wrist_det['depth']:.3f} m",
            (bx, max(16, by - 6)), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, (255, 255, 0), 1, cv2.LINE_AA,
        )

    badges = {
        SafetyStatus.HOLDING_SAFETY_STOP: (
            "CUBE HELD: ARM PAUSED", (0, 0, 220), 2,
        ),
        SafetyStatus.SAFETY_STOP: (
            "SAFETY STOP (< 0.06m) - EMERGENCY HALT", (0, 0, 220), 2,
        ),
        SafetyStatus.JOINT_LIMIT: (
            "WARNING: JOINT LIMIT REACHED - CLAMPED", (0, 140, 255), 2,
        ),
        SafetyStatus.TRACKING: (
            "VISUAL SERVOING: AVOIDING & TRACKING", (80, 80, 0), 1,
        ),
        SafetyStatus.RESUMING: (
            "RETURNING TO TASK (SMOOTH RESUME)", (150, 100, 0), 1,
        ),
        SafetyStatus.RECOVERING: (
            "OVERHEAD CAM RECOVERY (RE-ALIGNING)", (180, 0, 180), 1,
        ),
    }

    if safety_status in badges:
        text, background, thickness = badges[safety_status]
        cv2.rectangle(
            colour, (4, 4), (colour.shape[1] - 4, 26),
            background, cv2.FILLED,
        )
        cv2.putText(
            colour, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, (255, 255, 255), thickness, cv2.LINE_AA,
        )
    else:
        cv2.putText(
            colour, "Wrist Camera (PBVS)", (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

    if safety_status in (
        SafetyStatus.TRACKING,
        SafetyStatus.JOINT_LIMIT,
        SafetyStatus.SAFETY_STOP,
    ):
        if servo_telemetry:
            distance = servo_telemetry.get("distance", 0.0)
            error = servo_telemetry.get("standoff_error", 0.0)
            cv2.putText(
                colour,
                f"Dist: {distance:.3f}m | "
                f"Standoff: 0.220m | Err: {error:+.3f}m",
                (8, 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.36, (0, 255, 255), 1, cv2.LINE_AA,
            )

        if wrist_kf and wrist_kf.tracking:
            pos = np.round(wrist_kf.position, 2)
            vel = np.round(wrist_kf.velocity, 2)
            cv2.putText(
                colour, f"Wrist KF: Pos {pos} Vel {vel}",
                (8, 56), cv2.FONT_HERSHEY_SIMPLEX,
                0.34, (255, 200, 100), 1, cv2.LINE_AA,
            )
    else:
        cv2.putText(
            colour, wrist_status, (8, colour.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35, (255, 255, 255), 1, cv2.LINE_AA,
        )

    cv2.putText(
        wrist_panel, "Wrist Depth: near = warm", (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (255, 255, 255), 1, cv2.LINE_AA,
    )
    wrist_row = np.hstack((colour, wrist_panel))

    safety_annotated = safety_camera.annotate_view(
        safety_rgb, safety_depth, obstacle_active,
    )
    safety_panel = depth_to_colormap(safety_depth)
    cv2.putText(
        safety_panel, "Overhead Safety Depth", (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (255, 255, 255), 1, cv2.LINE_AA,
    )
    safety_row = np.hstack((safety_annotated, safety_panel))
    return np.vstack((wrist_row, safety_row))


def create_scene(seed):
    """Initialise the scene without passing spawn positions to the task."""
    model = mujoco.MjModel.from_xml_path(str(COMPLETE))
    model.vis.quality.offsamples = 0
    data = mujoco.MjData(model)

    joints = np.array([
        get_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in JOINT_NAMES
    ])
    q_addresses = model.jnt_qposadr[joints]
    v_addresses = model.jnt_dofadr[joints]

    motors = np.array([
        get_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in MOTOR_NAMES
    ])

    gripper = get_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_fingers_actuator",
    )
    camera = get_id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera",
    )
    marker = get_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "placement_square",
    )

    home = np.deg2rad(HOME_DEG)
    data.qpos[q_addresses] = home

    rng = np.random.default_rng(seed)
    positions = []

    for index in range(3):
        geom = get_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"cube_{index}_geom",
        )
        half_size = float(model.geom_size[geom, 0])

        for _ in range(1000):
            xy = rng.uniform([0.22, 0.08], [0.38, 0.24])
            if all(
                np.linalg.norm(xy - previous) > 0.085
                for previous in positions
            ):
                break
        else:
            raise RuntimeError("Could not space cubes apart.")

        positions.append(xy)

        joint = get_id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"cube_{index}_joint",
        )
        address = model.jnt_qposadr[joint]
        yaw = rng.uniform(-np.pi, np.pi)

        data.qpos[address:address + 3] = [
            xy[0], xy[1], TABLE_TOP + half_size + 0.002,
        ]
        data.qpos[address + 3:address + 7] = [
            np.cos(yaw / 2), 0, 0, np.sin(yaw / 2),
        ]

    marker_xy = rng.uniform([0.26, -0.18], [0.34, -0.12])
    model.geom_pos[marker] = [
        marker_xy[0], marker_xy[1], TABLE_TOP + 0.0002,
    ]
    mujoco.mj_forward(model, data)

    return (
        model, data, q_addresses, v_addresses,
        motors, gripper, camera, home,
    )


def render_rgbd(renderer, data, options, camera_name="wrist_camera"):
    renderer.disable_depth_rendering()
    renderer.update_scene(
        data, camera=camera_name, scene_option=options,
    )
    rgb = renderer.render().copy()

    renderer.enable_depth_rendering()
    renderer.update_scene(
        data, camera=camera_name, scene_option=options,
    )
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    return rgb, depth


def run_scene(seed, sim_speed: float = 1.0):
    (
        model, data, q_addresses, v_addresses,
        motors, gripper, camera, home,
    ) = create_scene(seed)

    arm_controller = None
    if USE_LQR_TORQUE:
        arm_controller = ComputedTorqueController(
            model, q_addresses, v_addresses, motors,
        )

    detector = CubeDetector(
        CAMERA_WIDTH, CAMERA_HEIGHT, model.cam_fovy[camera],
    )
    frames = CameraFrames(model, JOINT_NAMES)
    frames.update(home)

    def read_arm_state():
        if observer.filter.last_time is None:
            return home.copy(), np.zeros(6)
        return observer.filter.q, observer.filter.qd

    scanner = WorkspaceScanner(
        model, data, q_addresses, v_addresses, camera,
        state_reader=read_arm_state,
    )
    search = scanner.run()
    task = None

    histories = {
        name: deque(maxlen=STABLE_FRAMES)
        for name in CUBE_NAMES
    }
    marker_history = deque(maxlen=STABLE_FRAMES)
    records = {}
    placement = None

    feedback = {
        "frames": frames,
        "state_reader": read_arm_state,
        "mode": "idle",
        "name": None,
        "collecting": False,
        "result": None,
        "servo": None,
        "history": deque(maxlen=STABLE_FRAMES),
    }

    desired_q = home.copy()
    desired_qd = np.zeros(6)
    grip = 0.0
    frozen = False
    next_report = 0.0

    options = mujoco.MjvOption()
    options.geomgroup[3] = 0

    dt = model.opt.timestep
    steps_per_frame = max(1, round(1 / (CAMERA_FPS * dt)))
    window = "Wrist camera: RGB-D and PBVS"

    log_directory = ROOT / "logs"
    log_directory.mkdir(exist_ok=True)

    log_path = log_directory / f"pbvs_seed_{seed}_{time.time_ns()}.csv"
    observer_path = log_path.with_name(
        log_path.name.replace("pbvs_", "observer_", 1)
    )
    safety_path = log_path.with_name(
        log_path.name.replace("pbvs_", "safety_", 1)
    )

    print("Searching for cubes and purple using RGB-D.")
    print(
        "PBVS controls pickup and placement alignment; "
        "placement lowering uses a trajectory."
    )
    print("Press Q in the camera window to exit.")
    print("PBVS log:", log_path)

    safety_camera = StationarySafetyCamera(
        model, camera_name="safety_camera",
        width=CAMERA_WIDTH, height=CAMERA_HEIGHT,
    )

    # Known, fixed overhead-camera calibration.
    overhead_detector = CubeDetector(
        CAMERA_WIDTH, CAMERA_HEIGHT,
        model.cam_fovy[safety_camera.camera_id],
    )
    overhead_position_world, overhead_rotation_world = (
        safety_camera.get_camera_extrinsics(data)
    )

    base_rotation_world = frames.kinematics.xmat[
        frames.base_id
    ].reshape(3, 3).copy()
    base_position_world = frames.kinematics.xpos[
        frames.base_id
    ].copy()

    overhead_rotation_base = (
        base_rotation_world.T
        @ overhead_rotation_world
        @ np.diag([1.0, -1.0, -1.0])
    )
    overhead_position_base = (
        base_rotation_world.T
        @ (overhead_position_world - base_position_world)
    )
    overhead_frames = SimpleNamespace(
        points_to_base=lambda points: (
            points @ overhead_rotation_base.T
            + overhead_position_base
        )
    )

    obstacle = InteractiveObstacle(model, data, name="obstacle")
    carry_supervisor = CubeCarrySafetySupervisor(
        config=safety_camera.config,
    )
    pinch_site = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "gripper_pinch",
    )

    safety_status = SafetyStatus.NORMAL
    last_safety_status = SafetyStatus.NORMAL

    wrist_kf = ObjectTrackingKalmanFilter(
        position_std=0.004, acceleration_std=2.0,
    )
    servo_controller = VisualServoingAvoidanceController(
        model, q_addresses, v_addresses,
        camera_name="wrist_camera",
        config=safety_camera.config,
    )

    servo_rates = np.zeros(6)
    servo_telemetry = {}
    wrist_intruder_det = None

    last_task_q = home.copy()
    last_task_grip = 0.0
    resume_target_q = home.copy()

    carry_hold = False
    carry_hold_q = None

    try:
        with (
            log_path.open("w", newline="") as log_file,
            ObserverMonitor(
                observer_path, JOINT_NAMES, seed,
            ) as observer,
            SafetyLogger(safety_path) as safety_log,
        ):
            estimation_plotter = EstimationPlotter(JOINT_NAMES)
            observer.plotter = estimation_plotter

            writer = csv.writer(log_file)
            writer.writerow([
                "time_s", "target", "phase", "valid",
                "translation_error_m", "rotation_error_rad",
                "max_command_rad_s", "converged",
            ])

            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                window, 2 * CAMERA_WIDTH, 2 * CAMERA_HEIGHT,
            )
            cv2.moveWindow(window, 30, 60)

            cv2.namedWindow(
                estimation_plotter.window_name, cv2.WINDOW_AUTOSIZE,
            )
            cv2.moveWindow(estimation_plotter.window_name, 690, 60)
            estimation_plotter.sim_speed = sim_speed

            with mujoco.Renderer(
                model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH,
            ) as renderer:
                with mujoco.viewer.launch_passive(model, data) as viewer:
                    with viewer.lock():
                        viewer.opt.geomgroup[3] = 0
                        viewer.cam.distance = 1.4
                        viewer.cam.lookat[:] = [0.2, 0, 0.15]

                    while viewer.is_running():
                        wall_start = time.perf_counter()

                        if not frozen:
                            for _ in range(steps_per_frame):
                                if carry_hold:
                                    desired_q = carry_hold_q.copy()
                                    desired_qd = np.zeros(6)

                                elif safety_status == SafetyStatus.SAFETY_STOP:
                                    d_rate = -desired_qd
                                    desired_qd += np.clip(
                                        d_rate, -2.0 * dt, 2.0 * dt,
                                    )
                                    desired_q += desired_qd * dt

                                elif safety_status in (
                                    SafetyStatus.TRACKING,
                                    SafetyStatus.JOINT_LIMIT,
                                ):
                                    d_rate = servo_rates - desired_qd
                                    desired_qd += np.clip(
                                        d_rate, -1.5 * dt, 1.5 * dt,
                                    )
                                    desired_q += desired_qd * dt
                                    desired_q = np.clip(
                                        desired_q,
                                        servo_controller.lower_limits,
                                        servo_controller.upper_limits,
                                    )

                                elif safety_status == SafetyStatus.RECOVERING:
                                    d_rate = -desired_qd
                                    desired_qd += np.clip(
                                        d_rate, -2.0 * dt, 2.0 * dt,
                                    )
                                    desired_q += desired_qd * dt

                                elif safety_status == SafetyStatus.RESUMING:
                                    error = resume_target_q - desired_q
                                    distance = float(np.linalg.norm(error))

                                    if distance > 0.002:
                                        max_step = np.deg2rad(20.0) * dt
                                        step = min(distance, max_step)
                                        desired_qd = (
                                            error / distance
                                        ) * (step / dt)
                                        desired_q += desired_qd * dt
                                    else:
                                        desired_qd = np.zeros(6)
                                        desired_q = resume_target_q.copy()
                                        safety_status = SafetyStatus.NORMAL

                                        if feedback.get("servo") is not None:
                                            feedback["servo"].last_seen = data.time

                                        print(
                                            "[Safety Net] Smooth return "
                                            "complete: Task resumed."
                                        )

                                elif safety_status == SafetyStatus.RETREAT:
                                    desired_qd = np.zeros(6)
                                    error_to_home = home - desired_q
                                    distance = np.linalg.norm(error_to_home)

                                    if distance > 1e-4:
                                        desired_q += (
                                            error_to_home / distance
                                        ) * min(0.003, distance)

                                elif safety_status == SafetyStatus.INTERRUPT:
                                    desired_qd = np.zeros(6)

                                else:
                                    active = task if task is not None else search

                                    if active is not None:
                                        try:
                                            desired_q, desired_qd, grip = next(active)
                                            last_task_q = desired_q.copy()
                                            last_task_grip = grip

                                        except StopIteration:
                                            desired_qd = np.zeros(6)
                                            if task is not None:
                                                frozen = True
                                                break
                                            search = None

                                        except (
                                            RuntimeError,
                                            ValueError,
                                            np.linalg.LinAlgError,
                                        ) as error:
                                            message = str(error)

                                            if (
                                                safety_status != SafetyStatus.NORMAL
                                                or "tracking margin" in message
                                                or "joint-limit" in message
                                                or "target measurement lost" in message
                                            ):
                                                print(
                                                    "[Safety Net] Task tracking "
                                                    f"divergence caught ({error}) -> "
                                                    "Engaging Overhead Camera Recovery!"
                                                )
                                                desired_q = observer.filter.q.copy()
                                                desired_qd = np.zeros(6)

                                                if (
                                                    arm_controller is not None
                                                    and hasattr(
                                                        arm_controller, "feedback"
                                                    )
                                                ):
                                                    arm_controller.feedback.integral_error[:] = 0.0

                                                safety_status = SafetyStatus.RECOVERING
                                                break
                                            else:
                                                frozen = True
                                                print("STOPPED:", error)
                                                break

                                try:
                                    if arm_controller is not None:
                                        if (
                                            data.time + 1e-10
                                            >= arm_controller.next_update
                                        ):
                                            mujoco.mj_sensorPos(model, data)

                                            observer.sample(
                                                data.time,
                                                data.sensordata[
                                                    frames.encoder_addresses
                                                ],
                                                data.qpos[q_addresses],
                                                data.qvel[v_addresses],
                                                acceleration=(
                                                    arm_controller
                                                    .acceleration_command.copy()
                                                ),
                                            )

                                        data.ctrl[motors] = arm_controller.commands(
                                            data, desired_q, desired_qd,
                                            estimated_q=observer.filter.q,
                                            estimated_qd=observer.filter.qd,
                                        )
                                    else:
                                        # Existing development fallback.
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
                                        data.ctrl[motors] = np.clip(
                                            torque,
                                            model.actuator_ctrlrange[motors, 0],
                                            model.actuator_ctrlrange[motors, 1],
                                        )

                                except (
                                    RuntimeError,
                                    ValueError,
                                    np.linalg.LinAlgError,
                                ) as error:
                                    frozen = True
                                    print("STOPPED: arm controller:", error)
                                    break

                                data.ctrl[gripper] = grip
                                mujoco.mj_step(model, data)

                                error_deg = np.rad2deg(
                                    desired_q - observer.filter.q
                                )

                                if (
                                    not np.all(np.isfinite(data.qpos))
                                    or not np.all(np.isfinite(data.qvel))
                                    or not np.all(np.isfinite(error_deg))
                                ):
                                    frozen = True
                                    print(
                                        "STOPPED: invalid numerical state (NaN/Inf)."
                                    )
                                    break

                                elif np.max(np.abs(error_deg)) > 8.0:
                                    print(
                                        "[Safety Net] Joint divergence "
                                        f"({np.max(np.abs(error_deg)):.1f} deg) "
                                        "detected -> Engaging Overhead "
                                        "Camera Recovery!"
                                    )
                                    desired_q = observer.filter.q.copy()
                                    desired_qd = np.zeros(6)

                                    if (
                                        arm_controller is not None
                                        and hasattr(arm_controller, "feedback")
                                    ):
                                        arm_controller.feedback.integral_error[:] = 0.0

                                    safety_status = SafetyStatus.RECOVERING
                                    servo_rates = np.zeros(6)
                                    break

                        mujoco.mj_forward(model, data)

                        if arm_controller is None:
                            observer.sample(
                                data.time,
                                data.sensordata[frames.encoder_addresses],
                                data.qpos[q_addresses],
                                data.qvel[v_addresses],
                            )

                        viewer.sync()

                        rgb, depth = render_rgbd(
                            renderer, data, options,
                            camera_name="wrist_camera",
                        )
                        annotated, detections = detector.detect(rgb, depth)

                        safety_rgb, safety_depth = render_rgbd(
                            renderer, data, options,
                            camera_name="safety_camera",
                        )

                        frames.update(observer.filter.q)

                        estimated_pinch_pos = (
                            frames.kinematics.site_xpos[pinch_site].copy()
                            if pinch_site >= 0 else None
                        )
                        pinch_pos = (
                            data.site_xpos[pinch_site].copy()
                            if pinch_site >= 0 else None
                        )

                        safety_camera.update(
                            safety_rgb, safety_depth, data,
                            now=data.time,
                            robot_pinch_pos=estimated_pinch_pos,
                        )

                        wrist_intruder_det = detect_cyan_intruder_wrist(
                            rgb, depth,
                            fovy=float(model.cam_fovy[camera]),
                        )

                        if wrist_intruder_det is not None:
                            pt_cam = wrist_intruder_det["point_camera"]
                            pt_base = frames.points_to_base(
                                pt_cam.reshape(1, 3)
                            )[0]
                            wrist_kf.update(pt_base, data.time)

                        elif (
                            safety_camera.filter.tracking
                            and safety_camera.filter.confirmed_frames >= 1
                        ):
                            if not wrist_kf.tracking:
                                wrist_kf.update(
                                    safety_camera.filter.position, data.time,
                                )
                            else:
                                wrist_kf.predict(1.0 / CAMERA_FPS)
                        else:
                            wrist_kf.mark_missed()

                        servo = feedback.get("servo")
                        is_holding = (
                            grip > 0.01
                            or (
                                servo is not None
                                and getattr(servo, "placing", False)
                            )
                        )
                        held_size = (
                            servo.held_size
                            if (
                                is_holding
                                and servo is not None
                                and getattr(servo, "held_size", None) is not None
                            )
                            else 0.04
                        )

                        carry_supervisor.update(
                            safety_camera, is_holding,
                            observer.filter.q.copy(),
                        )

                        if carry_supervisor.paused and not carry_hold:
                            carry_hold = True
                            carry_hold_q = carry_supervisor.frozen_q.copy()
                            desired_q = carry_hold_q.copy()
                            desired_qd = np.zeros(6)
                            servo_rates = np.zeros(6)

                            print(
                                "[Safety Net] Carrying cube: obstacle "
                                "detected, holding joint pose."
                            )

                        if carry_hold and carry_supervisor.ready_to_resume:
                            carry_hold = False
                            carry_hold_q = None
                            carry_supervisor.finish_resume()

                            desired_q = observer.filter.q.copy()
                            desired_qd = np.zeros(6)
                            resume_target_q = last_task_q.copy()
                            safety_status = SafetyStatus.RESUMING
                            servo_rates = np.zeros(6)

                            print(
                                "[Safety Net] Carry hold cleared: "
                                "smooth return to task."
                            )

                        if carry_hold:
                            safety_status = SafetyStatus.HOLDING_SAFETY_STOP
                            safety_camera.status = SafetyStatus.HOLDING_SAFETY_STOP

                        elif safety_status == SafetyStatus.RECOVERING:
                            desired_q = observer.filter.q.copy()
                            desired_qd = np.zeros(6)

                            if (
                                safety_camera.filter.tracking
                                and safety_camera.filter.confirmed_frames >= 1
                            ):
                                target_pos_base = safety_camera.filter.position
                                target_vel_base = safety_camera.filter.velocity

                                (
                                    servo_status, rates, servo_telemetry,
                                ) = servo_controller.compute_command(
                                    data, frames, target_pos_base,
                                    desired_q, target_vel_base,
                                    holding_object=is_holding,
                                    held_size=held_size,
                                )

                                servo_rates = np.clip(rates, -0.3, 0.3)
                                safety_status = SafetyStatus.TRACKING

                                print(
                                    "[Safety Net] Overhead camera successfully "
                                    "re-acquired obstacle and restored visual tracking!"
                                )
                            else:
                                safety_status = SafetyStatus.RESUMING
                                resume_target_q = last_task_q.copy()
                                servo_rates = np.zeros(6)

                                print(
                                    "[Safety Net] Overhead camera confirmed "
                                    "workspace clear: Resuming task!"
                                )

                        elif (
                            safety_camera.filter.tracking
                            and safety_camera.filter.confirmed_frames
                            >= safety_camera.config.persistence_frames
                        ):
                            if (
                                wrist_kf.tracking
                                and wrist_kf.confirmed_frames >= 1
                            ):
                                target_pos_base = wrist_kf.position
                                target_vel_base = wrist_kf.velocity
                            else:
                                target_pos_base = safety_camera.filter.position
                                target_vel_base = safety_camera.filter.velocity

                            if safety_status in (
                                SafetyStatus.NORMAL,
                                SafetyStatus.RESUMING,
                                SafetyStatus.WARNING,
                            ):
                                desired_q = observer.filter.q.copy()
                                desired_qd = np.zeros(6)

                            (
                                servo_status, rates, servo_telemetry,
                            ) = servo_controller.compute_command(
                                data, frames, target_pos_base,
                                desired_q, target_vel_base,
                                holding_object=is_holding,
                                held_size=held_size,
                            )

                            safety_status = servo_status
                            servo_rates = rates

                        elif safety_status in (
                            SafetyStatus.TRACKING,
                            SafetyStatus.JOINT_LIMIT,
                            SafetyStatus.SAFETY_STOP,
                        ):
                            safety_status = SafetyStatus.RESUMING
                            resume_target_q = last_task_q.copy()
                            desired_q = observer.filter.q.copy()
                            desired_qd = np.zeros(6)

                            safety_camera.filter.reset()
                            wrist_kf.reset()
                            servo_rates = np.zeros(6)
                            servo_telemetry = {}

                            print(
                                "[Safety Net] Obstacle cleared: Initiating smooth "
                                "return trajectory to task waypoint..."
                            )

                        if safety_status != last_safety_status:
                            print(
                                "[Safety Net] Status transition: "
                                f"{last_safety_status.value.upper()} -> "
                                f"{safety_status.value.upper()}"
                            )
                            last_safety_status = safety_status

                        # Evaluation only.
                        safety_log.sample(
                            now=data.time,
                            camera=safety_camera,
                            obstacle=obstacle,
                            status=safety_status,
                            carrying=is_holding,
                            carry_hold=carry_hold,
                            actual_q=data.qpos[q_addresses],
                            actual_qd=data.qvel[v_addresses],
                            pinch_position=pinch_pos,
                        )

                        # Fresh visual measurements requested by stacking.
                        verification = feedback.get("verification")

                        if verification is not None:
                            _, overhead_detections = overhead_detector.detect(
                                safety_rgb, safety_depth
                            )

                            check_name = verification["name"]
                            check_size = verification["size"]

                            surface = estimate_support_square(
                                overhead_detections.get(check_name),
                                overhead_frames,
                                check_size,
                            )
                            source = "overhead"

                            if surface is None:
                                surface = estimate_support_square(
                                    detections.get(check_name),
                                    frames,
                                    check_size,
                                )
                                source = "wrist"

                            if surface is None:
                                verification["samples"].clear()
                                verification["reason"] = (
                                    "Neither camera has an accepted "
                                    "top-face measurement."
                                )
                            else:
                                centre_base = np.asarray(
                                    surface["centre_base"], dtype=float
                                ).copy()
                                centre_base[2] -= check_size / 2
                                centre_world = frames.base_to_world(centre_base)

                                if np.all(np.isfinite(centre_world)):
                                    verification["samples"].append(
                                        (float(data.time), centre_world.copy())
                                    )
                                    verification["source"] = source
                                    verification["reason"] = (
                                        "Waiting for stable visual samples."
                                    )
                                else:
                                    verification["samples"].clear()
                                    verification["reason"] = (
                                        "Non-finite visual measurement."
                                    )

                        measurements = {}

                        for name in CUBE_NAMES:
                            if name not in detections:
                                continue

                            measurement = estimate_table_cube(
                                detections[name], frames, TABLE_TOP,
                            )
                            if measurement is not None:
                                measurements[name] = measurement

                        purple = None
                        if "purple" in detections:
                            purple = estimate_placement_square(
                                detections["purple"], frames, TABLE_TOP,
                            )

                        if not frozen and feedback["mode"] == "measure":
                            name = feedback["name"]

                            if (
                                feedback["collecting"]
                                and feedback["result"] is None
                            ):
                                measurement = measurements.get(name)

                                if measurement is None:
                                    feedback["history"].clear()
                                else:
                                    feedback["history"].append(measurement)
                                    fresh = stable_cube(feedback["history"])

                                    if fresh is not None:
                                        feedback["result"] = world_target(
                                            name, fresh, frames,
                                        )
                                        centre = np.round(
                                            fresh["centre_base"], 3,
                                        )
                                        print(f"Fresh {name}: {centre} m")

                            elif not feedback["collecting"]:
                                feedback["history"].clear()

                        if (
                            not frozen
                            and feedback["mode"] in ("pbvs", "transport")
                        ):
                            servo = feedback["servo"]

                            try:
                                if servo.placing:
                                    measurement = estimate_support_square(
                                        detections.get(servo.name),
                                        frames,
                                        servo.size,
                                    )
                                else:
                                    measurement = measurements.get(servo.name)

                                servo.update(measurement, data.time)

                                if feedback["mode"] == "transport":
                                    feedback["transport_valid_frames"] = (
                                        feedback.get("transport_valid_frames", 0) + 1
                                        if servo.valid else 0
                                    )
                                    servo.command[:] = 0.0

                            except (
                                RuntimeError, ValueError, np.linalg.LinAlgError,
                            ) as error:
                                frozen = True
                                print("STOPPED:", error)

                            writer.writerow([
                                data.time, servo.name, servo.phase, servo.valid,
                                servo.translation_error, servo.rotation_error,
                                np.max(np.abs(servo.command)), servo.converged,
                            ])

                            if data.time >= next_report:
                                next_report = data.time + 1.0
                                position_mm = servo.translation_error * 1000
                                angle_deg = np.rad2deg(servo.rotation_error)

                                print(
                                    f"PBVS {servo.phase} ({servo.name}): "
                                    f"valid={servo.valid}, "
                                    f"position={position_mm:.1f} mm, "
                                    f"rotation={angle_deg:.1f} deg"
                                )

                        if task is None and not frozen:
                            if scanner.collecting and data.time >= 2.0:
                                for name in CUBE_NAMES:
                                    if name in records:
                                        continue

                                    measurement = measurements.get(name)
                                    if measurement is None:
                                        histories[name].clear()
                                        continue

                                    histories[name].append(measurement)
                                    record = stable_cube(histories[name])

                                    if record is not None:
                                        records[name] = record
                                        size_mm = record["size"] * 1000
                                        print(
                                            f"Recorded {name}: {size_mm:.1f} mm"
                                        )

                                if placement is None:
                                    if purple is None:
                                        marker_history.clear()
                                    else:
                                        marker_history.append(purple)
                                        placement = stable_marker(marker_history)

                                        if placement is not None:
                                            centre = np.round(
                                                placement["centre_base"], 3,
                                            )
                                            print("Recorded purple:", centre, "m")
                            else:
                                for history in histories.values():
                                    history.clear()
                                marker_history.clear()

                            ready = (
                                len(records) == 3
                                and placement is not None
                                and scanner.collecting
                            )

                            if ready:
                                catalog = sorted(
                                    [
                                        world_target(name, records[name], frames)
                                        for name in CUBE_NAMES
                                    ],
                                    key=lambda cube: cube["size"],
                                    reverse=True,
                                )

                                ambiguous = any(
                                    larger["size"] - smaller["size"] < 0.003
                                    for larger, smaller in zip(
                                        catalog, catalog[1:],
                                    )
                                )

                                if ambiguous:
                                    frozen = True
                                    print("STOPPED: size ordering is ambiguous.")
                                else:
                                    destination = frames.base_to_world(
                                        placement["centre_base"]
                                    )[:2]
                                    print(
                                        "Order:",
                                        " -> ".join(cube["name"] for cube in catalog),
                                    )
                                    print(
                                        "Camera-derived destination:",
                                        np.round(destination, 3),
                                    )

                                    feedback["placement"] = dict(
                                        placement, name="purple", size=0.11,
                                        yaw=0.0, surface=True,
                                    )
                                    task = stack_cubes(
                                        model, data, q_addresses, v_addresses,
                                        desired_q.copy(), catalog, feedback,
                                    )
                                    search = None
                                    scanner.collecting = False

                            elif search is None:
                                frozen = True
                                missing = [
                                    name for name in CUBE_NAMES
                                    if name not in records
                                ]
                                if placement is None:
                                    missing.append("purple")
                                print("SEARCH INCOMPLETE:", ", ".join(missing))

                            if data.time >= next_report:
                                next_report = data.time + 2.0
                                print(
                                    scanner.status,
                                    "; recorded:", list(records),
                                    "; purple:", placement is not None,
                                )

                        if safety_status == SafetyStatus.HOLDING_SAFETY_STOP:
                            display_status = "CUBE HELD - ARM PAUSED"
                        elif safety_status == SafetyStatus.SAFETY_STOP:
                            display_status = "SAFETY STOP (< 0.06m)"
                        elif safety_status == SafetyStatus.JOINT_LIMIT:
                            display_status = "JOINT LIMIT REACHED"
                        elif safety_status == SafetyStatus.TRACKING:
                            display_status = "TRACKING & AVOIDING"
                        elif safety_status == SafetyStatus.RESUMING:
                            display_status = "RESUMING TASK"
                        elif safety_status == SafetyStatus.RECOVERING:
                            display_status = "OVERHEAD RECOVERY"
                        elif safety_status in (
                            SafetyStatus.INTERRUPT, SafetyStatus.RETREAT,
                        ):
                            display_status = (
                                f"INTERRUPTED ({safety_status.value.upper()})"
                            )
                        elif frozen:
                            display_status = "Paused"
                        elif task is None:
                            display_status = scanner.status
                        elif feedback["servo"] is not None:
                            display_status = feedback["servo"].phase
                        else:
                            display_status = feedback["mode"]

                        cv2.imshow(
                            window,
                            display_dashboard(
                                annotated, depth, display_status,
                                safety_rgb, safety_depth, safety_camera,
                                obstacle.active,
                                wrist_det=wrist_intruder_det,
                                servo_telemetry=servo_telemetry,
                                safety_status=safety_status,
                                wrist_kf=wrist_kf,
                            ),
                        )
                        cv2.imshow(
                            estimation_plotter.window_name,
                            estimation_plotter.render(),
                        )

                        key = cv2.waitKeyEx(1)
                        if key != -1:
                            raw_key = key & 255

                            if raw_key == ord("q"):
                                break

                            if raw_key in (ord("+"), ord("=")):
                                sim_speed = (
                                    1.0 if sim_speed == 0
                                    else min(5.0, round(sim_speed + 0.5, 1))
                                )
                                estimation_plotter.sim_speed = sim_speed
                                print(
                                    f"[Sim Speed] Increased to {sim_speed:.1f}x"
                                )

                            elif raw_key in (ord("-"), ord("_")):
                                sim_speed = (
                                    3.0 if sim_speed == 0
                                    else max(0.5, round(sim_speed - 0.5, 1))
                                )
                                estimation_plotter.sim_speed = sim_speed
                                print(
                                    f"[Sim Speed] Decreased to {sim_speed:.1f}x"
                                )

                            elif raw_key in (ord("f"), ord("F")):
                                sim_speed = 0.0 if sim_speed > 0 else 1.0
                                estimation_plotter.sim_speed = sim_speed
                                print(
                                    "[Sim Speed] Fast mode: "
                                    f"{'UNLIMITED' if sim_speed == 0 else '1.0x'}"
                                )

                            obstacle.handle_key(key)
                            estimation_plotter.handle_key(key)
                            safety_log.obstacle_changed(
                                obstacle.active, data.time,
                            )

                        if cv2.getWindowProperty(
                            window, cv2.WND_PROP_VISIBLE,
                        ) < 1:
                            break

                        if sim_speed > 0:
                            remaining = (
                                steps_per_frame * dt / sim_speed
                                - (time.perf_counter() - wall_start)
                            )
                            if remaining > 0:
                                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\nSimulation stopped.")

    finally:
        cv2.destroyAllWindows()