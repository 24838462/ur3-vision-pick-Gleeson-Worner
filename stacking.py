import mujoco
import numpy as np
from roboticstoolbox import jtraj
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from robot_utils import get_id, joint_limits, solve_pose
from pbvs import GRASP_OFFSET_Z
from observation import WorkspaceScanner
from pbvs import PoseServo, LOST_TIMEOUT
TRAVEL_Z = 0.25
INWARD_RADIUS = 0.27
LIFT_CLEARANCE = 0.04
MAX_JOINT_SPEED_DEG = 25.0
VERTICAL_SPEED = 0.05
PLACEMENT_SPEED = 0.015
COLOUR_TO_INDEX = {'red': 0, 'green': 1, 'blue': 2}

def stack_cubes(model, data, q_addresses, v_addresses, home, catalog, feedback):
    """PBVS pickup and placement alignment with trajectory lowering."""
    dt = model.opt.timestep
    camera_id = get_id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'wrist_camera')
    pinch_id = get_id(model, mujoco.mjtObj.mjOBJ_SITE, 'gripper_pinch')
    table_id = get_id(model, mujoco.mjtObj.mjOBJ_GEOM, 'table_top')
    lower, upper = joint_limits(model, q_addresses)
    ik_data = mujoco.MjData(model)
    goal_q = np.asarray(home, dtype=float).copy()
    gripper_command = 0.0

    def hold(seconds):
        steps = max(1, int(np.ceil(seconds / dt)))
        for _ in range(steps):
            yield (goal_q.copy(), np.zeros(6), gripper_command)

    def settle():
        stable_steps = 0
        required_steps = max(1, int(np.ceil(0.15 / dt)))
        for _ in range(int(np.ceil(3.0 / dt))):
            yield (goal_q.copy(), np.zeros(6), gripper_command)
            error = np.max(np.abs(goal_q - data.qpos[q_addresses]))
            speed = np.max(np.abs(data.qvel[v_addresses]))
            if error < np.deg2rad(1) and speed < 0.05:
                stable_steps += 1
            else:
                stable_steps = 0
            if stable_steps >= required_steps:
                return
        raise RuntimeError('Arm did not settle at the requested pose.')

    def solve_ik(position, rotation, initial_q=None):
        position = np.asarray(position, dtype=float)
        ik_data.qpos[:] = data.qpos
        if initial_q is None:
            initial_q = data.qpos[q_addresses].copy()
        current_q = np.clip(initial_q, lower, upper)

        def residual(q):
            ik_data.qpos[q_addresses] = q
            mujoco.mj_kinematics(model, ik_data)
            actual_rotation = ik_data.site_xmat[pinch_id].reshape(3, 3)
            position_error = ik_data.site_xpos[pinch_id] - position
            rotation_error = Rotation.from_matrix(rotation @ actual_rotation.T).as_rotvec()
            return np.r_[position_error, 0.15 * rotation_error]
        solution = solve_pose(residual, current_q, lower, upper, 200)
        if solution is not None:
            return solution
        raise RuntimeError(f'IK failed for position {np.round(position, 3)} metres.')

    def move_to(position, rotation, label=None):
        nonlocal goal_q
        if label:
            print(label)
        start_q = data.qpos[q_addresses].copy()
        goal_q = solve_ik(position, rotation)
        largest_change_deg = np.max(np.abs(np.rad2deg(goal_q - start_q)))
        duration = max(2.0, 1.875 * largest_change_deg / MAX_JOINT_SPEED_DEG)
        times = np.arange(int(np.ceil(duration / dt)) + 1) * dt
        trajectory = jtraj(start_q, goal_q, times)
        for q, qd in zip(trajectory.q, trajectory.qd):
            yield (q.copy(), qd.copy(), gripper_command)
        yield from settle()

    def move_line(start_position, end_position, rotation, label, speed=VERTICAL_SPEED):
        """Approximate a Cartesian line using IK and a joint spline."""
        nonlocal goal_q
        print(label)
        start_position = np.asarray(start_position, dtype=float)
        end_position = np.asarray(end_position, dtype=float)
        path = np.linspace(0.0, 1.0, 15)
        joint_points = [data.qpos[q_addresses].copy()]
        for fraction in path[1:]:
            position = start_position + fraction * (end_position - start_position)
            solution = solve_ik(position, rotation, initial_q=joint_points[-1])
            if np.max(np.abs(solution - joint_points[-1])) > np.deg2rad(45):
                raise RuntimeError('Clearance movement encountered an IK configuration jump.')
            joint_points.append(solution)
        curve = CubicSpline(path, np.asarray(joint_points), axis=0, bc_type='natural')
        samples = np.linspace(0.0, 1.0, 200)
        peak_joint_slope = np.max(np.abs(curve(samples, 1)))
        duration = max(
            1.2,
            1.875 * np.linalg.norm(end_position - start_position) / speed,
            1.875 * peak_joint_slope / np.deg2rad(MAX_JOINT_SPEED_DEG),
        )
        steps = int(np.ceil(duration / dt))
        duration = steps * dt
        for step in range(steps + 1):
            u = step / steps
            progress = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
            progress_rate = (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4) / duration
            yield (curve(progress), curve(progress, 1) * progress_rate, gripper_command)
        goal_q = joint_points[-1].copy()
        yield from settle()

    def set_gripper(target):
        nonlocal gripper_command
        start_command = gripper_command
        steps = max(1, int(np.ceil(1.0 / dt)))
        for index in range(steps):
            fraction = (index + 1) / steps
            gripper_command = start_command + fraction * (target - start_command)
            yield (goal_q.copy(), np.zeros(6), gripper_command)
        yield from hold(1.0)

    def refresh_target(name):
        nonlocal goal_q
        print(f'{name}: use recorded position to reacquire target.')
        recorded = next((cube for cube in catalog if cube['name'] == name))
        recorded_position = np.asarray(recorded['centre_world'], dtype=float)
        xy = recorded_position[:2].copy()
        inward_xy = xy.copy()
        radius = np.linalg.norm(inward_xy)
        if radius > 0.3:
            inward_xy *= 0.3 / radius
        if radius > 1e-06:
            tangent = np.array([-xy[1], xy[0]]) / radius
        else:
            tangent = np.array([0.0, 1.0])
        viewpoints = [
            [xy[0], xy[1], 0.35],
            [inward_xy[0], inward_xy[1], 0.35],
            [inward_xy[0] + 0.04 * tangent[0], inward_xy[1] + 0.04 * tangent[1], 0.35],
            [inward_xy[0] - 0.04 * tangent[0], inward_xy[1] - 0.04 * tangent[1], 0.35],
            [xy[0], xy[1], 0.3],
        ]
        feedback.update(mode='measure', name=name, collecting=False, result=None, servo=None)
        feedback['history'].clear()
        attempted = set()
        try:
            for position in viewpoints:
                key = tuple(np.round(position, 4))
                if key in attempted:
                    continue
                attempted.add(key)
                feedback['collecting'] = False
                feedback['history'].clear()
                print(f'{name}: trying camera view {np.round(position, 3)} m')
                observer = WorkspaceScanner(
                    model,
                    data,
                    q_addresses,
                    v_addresses,
                    camera_id,
                    positions=[position],
                )
                try:
                    for q, qd, command in observer.run():
                        goal_q = q.copy()
                        feedback['collecting'] = observer.collecting
                        yield (q, qd, command)
                        if observer.collecting and feedback['result'] is not None:
                            return dict(feedback['result'])
                except RuntimeError as error:
                    if str(error).startswith('Camera IK failed'):
                        print(f'{name}: camera view skipped:', error)
                        continue
                    raise
            raise RuntimeError(f'No stable fresh measurement for {name} after trying recorded-location views.')
        finally:
            feedback.update(mode='idle', collecting=False)

    def run_servo(target, phase, held_size=None, servo=None):
        nonlocal goal_q
        if servo is None:
            servo = PoseServo(model, feedback['frames'], v_addresses, target,
                              phase=phase, held_size=held_size)
        feedback.update(mode='pbvs', name=target['name'], servo=servo, collecting=False)
        print(f"PBVS {phase}: tracking {target['name']}.")
        start_time = data.time
        goal_q = data.qpos[q_addresses].copy()
        try:
            while data.time - start_time < 45.0:
                if servo.last_seen is None:
                    if data.time - start_time > 1.0:
                        raise RuntimeError(f'PBVS {phase}: target not acquired.')
                elif data.time - servo.last_seen > LOST_TIMEOUT:
                    raise RuntimeError(f'PBVS {phase}: target measurement lost; motion paused.')
                if servo.converged:
                    print(f'PBVS {phase} converged: translation={servo.translation_error * 1000:.1f} mm, rotation={np.rad2deg(servo.rotation_error):.1f} deg')
                    return servo
                rates = servo.command.copy() if servo.valid else np.zeros(6)
                goal_q = goal_q + rates * dt
                if np.any(goal_q <= lower + 0.01) or np.any(goal_q >= upper - 0.01):
                    raise RuntimeError('PBVS: joint-limit margin reached.')
                measured_q = feedback['frames'].kinematics.qpos[q_addresses]
                if np.max(np.abs(goal_q - measured_q)) > np.deg2rad(5):
                    raise RuntimeError('PBVS: commanded position exceeds encoder tracking margin.')
                yield (goal_q.copy(), rates, gripper_command)
            raise RuntimeError(f'PBVS {phase}: timed out without convergence.')
        finally:
            feedback.update(mode='idle', servo=None)
    def transport_to_support(support, size, height, rotation, name):
        """Use recorded vision first, then hand motion to live PBVS."""
        nonlocal goal_q
        frames = feedback["frames"]
        support_world = frames.base_to_world(support["centre_base"])
        destination = np.array([support_world[0], support_world[1], height])
        kin = frames.kinematics
        rb = kin.xmat[frames.base_id].reshape(3, 3)
        transport_height_base = float(
            (rb.T @ (destination - kin.xpos[frames.base_id]))[2]
        )
        transport_target = dict(support)
        if support["name"] == "purple":
            transport_target["placement_rotation_base"] = rb.T @ rotation
        servo = PoseServo(
            model, frames, v_addresses, transport_target,
            phase="place_transport", held_size=size,
            transport_height=transport_height_base,
        )
        feedback.update(mode="transport", name=support["name"], servo=servo,
                        collecting=False, transport_valid_frames=0)
        try:
            # Check an already visible target before planning a coarse move.
            goal_q = data.qpos[q_addresses].copy()
            look_start = data.time
            while data.time - look_start < 0.25:
                yield goal_q.copy(), np.zeros(6), gripper_command
                if feedback["transport_valid_frames"] >= 3:
                    print(f"{name}: target visible; transport handed to PBVS.")
                    return (yield from run_servo(
                        support, "place_transport", held_size=size, servo=servo,
                    ))
            movement = move_to(
                destination, rotation,
                f"{name}: transport towards recorded placement region.",
            )
            for q, qd, command in movement:
                goal_q = q.copy()
                yield q, qd, command
                if feedback["transport_valid_frames"] >= 3:
                    print(f"{name}: target visible; transport handed to PBVS.")
                    # Abandon the unexecuted trajectory. Only PBVS commands
                    # are integrated from the current arm position now.
                    movement.close()
                    return (yield from run_servo(
                        support, "place_transport", held_size=size, servo=servo,
                    ))
            servo.last_seen = None
            servo.valid = servo.converged = False
            servo.stable_frames = 0
            servo.command[:] = 0.0
            print(f"{name}: at recorded region; acquiring target for PBVS.")
            return (yield from run_servo(
                support, "place_transport", held_size=size, servo=servo,
            ))
        finally:
            feedback.update(mode="idle", servo=None, transport_valid_frames=0)

    table_z = data.geom_xpos[table_id, 2] + model.geom_size[table_id, 2]
    support = dict(feedback['placement'])
    ordered = sorted(catalog, key=lambda cube: cube['size'], reverse=True)
    largest_size = max((cube['size'] for cube in ordered))
    print('Stacking using PBVS pickup and placement alignment with trajectory lowering.')
    for cube in ordered:
        name = cube['name']
        fresh = (yield from refresh_target(name))
        if abs(fresh['size'] - cube['size']) > 0.003:
            raise RuntimeError(f'{name}: fresh size inconsistent with recorded size.')
        size = float(fresh['size'])
        half_size = size / 2
        body_id = get_id(model, mujoco.mjtObj.mjOBJ_BODY, f'cube_{COLOUR_TO_INDEX[name]}')
        yield from set_gripper(0.0)
        approach = (yield from run_servo(fresh, 'pick_approach'))
        pick = (yield from run_servo(dict(approach.last_target), 'pick_descend'))
        grasp_rotation = pick.grasp_rotation_world.copy()
        cube_position = feedback['frames'].base_to_world(pick.last_target['centre_base'])
        print(f'{name}: close gripper.')
        yield from set_gripper(255.0)
        current = data.site_xpos[pinch_id].copy()
        clear_z = max(
            current[2] + 0.08,
            table_z + largest_size + LIFT_CLEARANCE + half_size + GRASP_OFFSET_Z,
        )
        lifted = np.array([current[0], current[1], clear_z])
        yield from move_line(current, lifted, grasp_rotation, f'{name}: lift clear of cubes.')
        yield from hold(0.5)
        if data.xpos[body_id, 2] < cube_position[2] + 0.04:
            raise RuntimeError(f'{name} did not lift. Check grasp and finger contacts.')
        inward_xy = lifted[:2].copy()
        radius = np.linalg.norm(inward_xy)
        if radius > INWARD_RADIUS:
            inward_xy *= INWARD_RADIUS / radius
        inward = np.array([inward_xy[0], inward_xy[1], clear_z])
        if np.linalg.norm(inward - lifted) > 0.001:
            yield from move_line(data.site_xpos[pinch_id].copy(), inward, grasp_rotation, f'{name}: move inward before raising.')
        transport_z = max(TRAVEL_Z, clear_z)
        transport = np.array([inward_xy[0], inward_xy[1], transport_z])
        yield from move_line(data.site_xpos[pinch_id].copy(), transport, grasp_rotation, f'{name}: raise at inward position.')
        transport_servo = yield from transport_to_support(
            support, size, transport_z, grasp_rotation, name,
        )
        support = dict(transport_servo.last_target)
        if data.xpos[body_id, 2] < table_z + half_size + 0.04:
            raise RuntimeError(f'{name} slipped during transport.')
        placement_approach = (yield from run_servo(support, 'place_approach', held_size=size))
        release_position = placement_approach.grasp_world.copy()
        release_rotation = placement_approach.grasp_rotation_world.copy()
        measured_support = dict(placement_approach.last_target)
        measured_support_world = feedback['frames'].base_to_world(measured_support['centre_base'])
        stack_xy = measured_support_world[:2].copy()
        current = data.site_xpos[pinch_id].copy()
        if np.linalg.norm(current[:2] - release_position[:2]) > 0.005:
            raise RuntimeError('Placement alignment exceeds 5 mm XY margin.')
        release_position[:2] = current[:2]
        if release_position[2] >= current[2]:
            raise RuntimeError('Measured release height is not below alignment pose.')
        yield from move_line(current, release_position, release_rotation, f'{name}: trajectory lowering after PBVS alignment.', speed=PLACEMENT_SPEED)
        print(f'{name}: release after trajectory lowering.')
        yield from set_gripper(0.0)
        current = data.site_xpos[pinch_id].copy()
        yield from move_line(current, current + np.array([0.0, 0.0, 0.08]), release_rotation, f'{name}: withdraw.')
        yield from hold(1.5)
        actual_position = data.xpos[body_id].copy()
        expected_z = measured_support_world[2] + half_size
        xy_error = np.linalg.norm(actual_position[:2] - stack_xy)
        z_error = abs(actual_position[2] - expected_z)
        if xy_error > 0.025 or z_error > 0.02:
            raise RuntimeError(f'{name} placement failed: XY error={xy_error * 1000:.1f} mm, Z error={z_error * 1000:.1f} mm.')
        print(f'{name}: placement confirmed.')
        next_top = measured_support['centre_base'].copy()
        next_top[2] += size
        frames = feedback['frames']
        base_rotation_world = frames.kinematics.xmat[frames.base_id].reshape(3, 3)
        release_rotation_base = base_rotation_world.T @ release_rotation
        placed_yaw = float(np.arctan2(release_rotation_base[1, 0], release_rotation_base[0, 0]))
        support = {
            'name': name,
            'size': size,
            'yaw': placed_yaw,
            'centre_base': next_top,
            'surface': True,
        }
    print('Returning to observation after stacking.')
    observer = WorkspaceScanner(
        model,
        data,
        q_addresses,
        v_addresses,
        camera_id,
        positions=[[0.3, 0.16, 0.35]],
    )
    for q, qd, command in observer.run():
        goal_q = q.copy()
        yield (q, qd, command)
    print('Stack sequence completed. Holding final pose.')
    while True:
        yield (goal_q.copy(), np.zeros(6), gripper_command)
