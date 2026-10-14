"""Runtime: scene setup, physics, RGB-D, perception and logging."""
from pathlib import Path
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
from torque_controller import ComputedTorqueController
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
COMPLETE = ROOT / 'models' / 'ur3_complete.xml'
HOME_DEG = [0, -90, 90, -90, -90, 0]
KP = np.array([80, 100, 80, 3, 3, 2], dtype=float)
KD = np.array([20, 25, 20, 0.15, 0.15, 0.08], dtype=float)
USE_LQR_TORQUE = True
CUBE_NAMES = ('red', 'green', 'blue')
TABLE_TOP = -0.01
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 15

def display(rgb, depth, status):
    colour = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    valid = np.isfinite(depth) & (depth > 0)
    safe_depth = np.where(valid, depth, 1.0)
    scaled = np.clip((safe_depth - 0.05) / 0.95, 0.0, 1.0)
    panel = cv2.applyColorMap(((1 - scaled) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    panel[~valid] = 0
    for image, title in ((colour, 'RGB'), (panel, 'Depth: near = warm')):
        cv2.putText(
            image,
            title,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        colour,
        status,
        (8, colour.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.hstack((colour, panel))

def create_scene(seed):
    """Initialise only; spawn positions are not passed to the task."""
    model = mujoco.MjModel.from_xml_path(str(COMPLETE))
    data = mujoco.MjData(model)
    joints = np.array([get_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES])
    q_addresses = model.jnt_qposadr[joints]
    v_addresses = model.jnt_dofadr[joints]
    motors = np.array([get_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in MOTOR_NAMES])
    gripper = get_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'gripper_fingers_actuator')
    camera = get_id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'wrist_camera')
    marker = get_id(model, mujoco.mjtObj.mjOBJ_GEOM, 'placement_square')
    home = np.deg2rad(HOME_DEG)
    data.qpos[q_addresses] = home
    rng = np.random.default_rng(seed)
    positions = []
    for index in range(3):
        geom = get_id(model, mujoco.mjtObj.mjOBJ_GEOM, f'cube_{index}_geom')
        half_size = float(model.geom_size[geom, 0])
        for _ in range(1000):
            xy = rng.uniform([0.22, 0.08], [0.38, 0.24])
            if all((np.linalg.norm(xy - previous) > 0.085 for previous in positions)):
                break
        else:
            raise RuntimeError('Could not space cubes apart.')
        positions.append(xy)
        joint = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, f'cube_{index}_joint')
        address = model.jnt_qposadr[joint]
        yaw = rng.uniform(-np.pi, np.pi)
        data.qpos[address:address + 3] = [xy[0], xy[1], TABLE_TOP + half_size + 0.002]
        data.qpos[address + 3:address + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    marker_xy = rng.uniform([0.26, -0.18], [0.34, -0.12])
    model.geom_pos[marker] = [marker_xy[0], marker_xy[1], TABLE_TOP + 0.0002]
    mujoco.mj_forward(model, data)
    return (model, data, q_addresses, v_addresses, motors, gripper, camera, home)

def render_rgbd(renderer, data, options):
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera='wrist_camera', scene_option=options)
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera='wrist_camera', scene_option=options)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    return (rgb, depth)

def run_scene(seed):
    model, data, q_addresses, v_addresses, motors, gripper, camera, home = create_scene(seed)
    arm_controller = None
    if USE_LQR_TORQUE:
        arm_controller = ComputedTorqueController(
            model,
            q_addresses,
            v_addresses,
            motors,
        )
    detector = CubeDetector(CAMERA_WIDTH, CAMERA_HEIGHT, model.cam_fovy[camera])
    frames = CameraFrames(model, JOINT_NAMES)
    frames.update(data.sensordata)
    scanner = WorkspaceScanner(model, data, q_addresses, v_addresses, camera)
    search = scanner.run()
    task = None
    histories = {name: deque(maxlen=STABLE_FRAMES) for name in CUBE_NAMES}
    marker_history = deque(maxlen=STABLE_FRAMES)
    records = {}
    placement = None
    feedback = {
        'frames': frames,
        'mode': 'idle',
        'name': None,
        'collecting': False,
        'result': None,
        'servo': None,
        'history': deque(maxlen=STABLE_FRAMES),
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
    window = 'Wrist camera: RGB-D and PBVS'
    log_directory = ROOT / 'logs'
    log_directory.mkdir(exist_ok=True)
    log_path = log_directory / f'pbvs_seed_{seed}_{time.time_ns()}.csv'
    observer_path = log_path.with_name(
        log_path.name.replace("pbvs_", "observer_", 1)
    )
    print('Searching for cubes and purple using RGB-D.')
    print('PBVS controls pickup and placement alignment; placement lowering uses a trajectory.')
    print('Press Q in the camera window to exit.')
    print('PBVS log:', log_path)
    try:
        with log_path.open("w", newline="") as log_file, ObserverMonitor(
            observer_path, JOINT_NAMES, seed
        ) as observer:
            writer = csv.writer(log_file)
            writer.writerow([
                'time_s',
                'target',
                'phase',
                'valid',
                'translation_error_m',
                'rotation_error_rad',
                'max_command_rad_s',
                'converged',
            ])
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 2 * CAMERA_WIDTH, CAMERA_HEIGHT)
            with mujoco.Renderer(model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH) as renderer:
                with mujoco.viewer.launch_passive(model, data) as viewer:
                    with viewer.lock():
                        viewer.opt.geomgroup[3] = 0
                        viewer.cam.distance = 1.4
                        viewer.cam.lookat[:] = [0.2, 0, 0.15]
                    while viewer.is_running():
                        wall_start = time.perf_counter()
                        if not frozen:
                            for _ in range(steps_per_frame):
                                active = (
                                    task
                                    if task is not None
                                    else search
                                )

                                if active is not None:
                                    try:
                                        desired_q, desired_qd, grip = next(active)

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
                                        frozen = True
                                        print("STOPPED:", error)
                                        break

                                # Arm control: LQR computed torque or
                                # the existing PD controller.
                                try:
                                    if arm_controller is not None:
                                        data.ctrl[motors] = (
                                            arm_controller.commands(
                                                data,
                                                desired_q,
                                                desired_qd,
                                            )
                                        )

                                    else:
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
                                            model.actuator_ctrlrange[
                                                motors, 0
                                            ],
                                            model.actuator_ctrlrange[
                                                motors, 1
                                            ],
                                        )

                                except (
                                    RuntimeError,
                                    ValueError,
                                    np.linalg.LinAlgError,
                                ) as error:
                                    frozen = True
                                    print("STOPPED: arm controller:", error)
                                    break

                                # Gripper control and physics.
                                data.ctrl[gripper] = grip
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
                                        "STOPPED: tracking error "
                                        "or invalid state."
                                    )
                                    break

                        # Refresh sensors at the current state.
                        mujoco.mj_forward(model, data)

                        # Observer evaluation remains separate
                        # from the controller for this first test.
                        observer.sample(
                            data.time,
                            data.sensordata[
                                frames.encoder_addresses
                            ],
                            data.qpos[q_addresses],
                            data.qvel[v_addresses],
                        )

                        viewer.sync()
                        rgb, depth = render_rgbd(renderer, data, options)
                        annotated, detections = detector.detect(rgb, depth)
                        frames.update(data.sensordata)
                        measurements = {}
                        for name in CUBE_NAMES:
                            if name not in detections:
                                continue
                            measurement = estimate_table_cube(detections[name], frames, TABLE_TOP)
                            if measurement is not None:
                                measurements[name] = measurement
                        purple = None
                        if 'purple' in detections:
                            purple = estimate_placement_square(
                                detections['purple'],
                                frames,
                                TABLE_TOP,
                            )
                        if not frozen and feedback['mode'] == 'measure':
                            name = feedback['name']
                            if feedback['collecting'] and feedback['result'] is None:
                                measurement = measurements.get(name)
                                if measurement is None:
                                    feedback['history'].clear()
                                else:
                                    feedback['history'].append(measurement)
                                    fresh = stable_cube(feedback['history'])
                                    if fresh is not None:
                                        feedback['result'] = world_target(name, fresh, frames)
                                        centre = np.round(fresh['centre_base'], 3)
                                        print(f'Fresh {name}: {centre} m')
                            elif not feedback['collecting']:
                                feedback['history'].clear()
                        if not frozen and feedback['mode'] in ('pbvs', 'transport'):
                            servo = feedback['servo']
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
                                if feedback['mode'] == 'transport':
                                    # Observe first; the trajectory still owns motion.
                                    feedback['transport_valid_frames'] = (
                                        feedback.get('transport_valid_frames', 0) + 1
                                        if servo.valid else 0
                                    )
                                    servo.command[:] = 0.0
                            except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                                frozen = True
                                print('STOPPED:', error)
                            writer.writerow([
                                data.time,
                                servo.name,
                                servo.phase,
                                servo.valid,
                                servo.translation_error,
                                servo.rotation_error,
                                np.max(np.abs(servo.command)),
                                servo.converged,
                            ])
                            if data.time >= next_report:
                                next_report = data.time + 1.0
                                position_mm = servo.translation_error * 1000
                                angle_deg = np.rad2deg(servo.rotation_error)
                                print(f'PBVS {servo.phase} ({servo.name}): valid={servo.valid}, position={position_mm:.1f} mm, rotation={angle_deg:.1f} deg')
                        if task is None and (not frozen):
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
                                        size_mm = record['size'] * 1000
                                        print(f'Recorded {name}: {size_mm:.1f} mm')
                                if placement is None:
                                    if purple is None:
                                        marker_history.clear()
                                    else:
                                        marker_history.append(purple)
                                        placement = stable_marker(marker_history)
                                        if placement is not None:
                                            centre = np.round(placement['centre_base'], 3)
                                            print('Recorded purple:', centre, 'm')
                            else:
                                for history in histories.values():
                                    history.clear()
                                marker_history.clear()
                            ready = len(records) == 3 and placement is not None and scanner.collecting
                            if ready:
                                catalog = sorted(
                                    [world_target(
                                        name,
                                        records[name],
                                        frames,
                                    ) for name in CUBE_NAMES],
                                    key=lambda cube: cube['size'],
                                    reverse=True,
                                )
                                ambiguous = any((larger['size'] - smaller['size'] < 0.003 for larger, smaller in zip(
                                    catalog,
                                    catalog[1:],
                                )))
                                if ambiguous:
                                    frozen = True
                                    print('STOPPED: size ordering is ambiguous.')
                                else:
                                    destination = frames.base_to_world(placement['centre_base'])[:2]
                                    print('Order:', ' -> '.join((cube['name'] for cube in catalog)))
                                    print('Camera-derived destination:', np.round(destination, 3))
                                    feedback['placement'] = dict(
                                        placement,
                                        name='purple',
                                        size=0.11,
                                        yaw=0.0,
                                        surface=True,
                                    )
                                    task = stack_cubes(
                                        model,
                                        data,
                                        q_addresses,
                                        v_addresses,
                                        desired_q.copy(),
                                        catalog,
                                        feedback,
                                    )
                                    search = None
                                    scanner.collecting = False
                            elif search is None:
                                frozen = True
                                missing = [name for name in CUBE_NAMES if name not in records]
                                if placement is None:
                                    missing.append('purple')
                                print('SEARCH INCOMPLETE:', ', '.join(missing))
                            if data.time >= next_report:
                                next_report = data.time + 2.0
                                print(
                                    scanner.status,
                                    '; recorded:',
                                    list(records),
                                    '; purple:',
                                    placement is not None,
                                )
                        if frozen:
                            status = 'Paused'
                        elif task is None:
                            status = scanner.status
                        elif feedback['servo'] is not None:
                            status = feedback['servo'].phase
                        else:
                            status = feedback['mode']
                        cv2.imshow(window, display(annotated, depth, status))
                        key = cv2.waitKey(1) & 255
                        if key == ord('q'):
                            break
                        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                            break
                        remaining = steps_per_frame * dt - (time.perf_counter() - wall_start)
                        if remaining > 0:
                            time.sleep(remaining)
    
    except KeyboardInterrupt:
        print('\nSimulation stopped.')
    finally:
        cv2.destroyAllWindows()
