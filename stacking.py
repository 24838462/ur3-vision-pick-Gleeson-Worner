"""Stacking with estimated robot states and visual cube checks."""

from collections import deque

import mujoco
import numpy as np
from roboticstoolbox import jtraj
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from robot_utils import get_id, joint_limits, solve_pose
from observation import WorkspaceScanner
from pbvs import GRASP_OFFSET_Z, PoseServo, LOST_TIMEOUT


TRAVEL_Z = 0.25
INWARD_RADIUS = 0.27
LIFT_CLEARANCE = 0.04
MAX_JOINT_SPEED_DEG = 45.0
TRANSPORT_JOINT_SPEED_DEG = 35.0

PLACEMENT_JOINT_ACCELERATION = 0.40
PLACEMENT_COMMAND_TIME_CONSTANT = 0.12

PICK_APPROACH_ATTEMPTS = 3
VERTICAL_SPEED = 0.08
PLACEMENT_SPEED = 0.030

COLOUR_TO_INDEX = {"red": 0, "green": 1, "blue": 2}


def stack_cubes(model, data, q_addresses, v_addresses, home, catalog, feedback):
    """PBVS pickup and placement alignment with trajectory lowering."""
    dt = model.opt.timestep
    camera_id = get_id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    pinch_id = get_id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_pinch")
    table_id = get_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")

    lower, upper = joint_limits(model, q_addresses)
    ik_data = mujoco.MjData(model)
    goal_q = np.asarray(home, dtype=float).copy()
    gripper_command = 0.0

    def estimated_pinch_position():
        frames = feedback["frames"]
        measured_q, _ = feedback["state_reader"]()
        frames.update(measured_q)
        return frames.kinematics.site_xpos[pinch_id].copy()

    def hold(seconds):
        steps = max(1, int(np.ceil(seconds / dt)))
        for _ in range(steps):
            yield goal_q.copy(), np.zeros(6), gripper_command

    def verify_cube_position(name, size, timeout=3.0):
        """Hold position while collecting fresh visual cube measurements."""
        request = {
            "name": name,
            "size": float(size),
            "samples": deque(maxlen=5),
            "source": None,
            "reason": "Waiting for a camera measurement.",
        }
        feedback["verification"] = request
        started = float(data.time)

        try:
            for _ in range(int(np.ceil(timeout / dt))):
                yield goal_q.copy(), np.zeros(6), gripper_command

                samples = list(request["samples"])
                if len(samples) < 5:
                    continue

                if data.time - samples[-1][0] > 0.2:
                    request["reason"] = "Visual measurements are stale."
                    continue

                if samples[0][0] <= started:
                    continue

                positions = np.asarray([
                    position for _, position in samples
                ])
                centre = np.median(positions, axis=0)
                spread = np.max(
                    np.linalg.norm(positions - centre, axis=1)
                )

                if spread > 0.005:
                    request["reason"] = (
                        "Visual positions vary by more than 5 mm."
                    )
                    continue

                print(
                    f"{name}: visual verification via "
                    f"{request['source']} camera."
                )
                return centre.copy()

            raise RuntimeError(
                f"{name}: cube position could not be verified from RGB-D. "
                f"{request['reason']}"
            )

        finally:
            if feedback.get("verification") is request:
                feedback["verification"] = None

    def settle():
        window_steps = max(1, int(np.ceil(0.15 / dt)))
        position_errors = deque(maxlen=window_steps)
        squared_velocities = deque(maxlen=window_steps)
        rms_velocity = np.full(6, np.nan)

        for _ in range(int(np.ceil(3.0 / dt))):
            yield goal_q.copy(), np.zeros(6), gripper_command
            measured_q, measured_qd = feedback["state_reader"]()

            if (
                not np.all(np.isfinite(measured_q))
                or not np.all(np.isfinite(measured_qd))
            ):
                raise RuntimeError(
                    "Stacking received invalid KF estimates."
                )

            position_errors.append(
                float(np.max(np.abs(goal_q - measured_q)))
            )
            squared_velocities.append(np.square(measured_qd))

            if len(position_errors) < window_steps:
                continue

            rms_velocity = np.sqrt(
                np.mean(np.asarray(squared_velocities), axis=0)
            )
            position_ok = max(position_errors) < np.deg2rad(1)
            velocity_ok = np.max(rms_velocity) < 0.05

            if position_ok and velocity_ok:
                return

        measured_q, measured_qd = feedback["state_reader"]()
        print(
            "Estimated settling errors (deg):",
            np.round(np.rad2deg(goal_q - measured_q), 3),
        )
        print(
            "Window RMS velocities (rad/s):",
            np.round(rms_velocity, 4),
        )

        raise RuntimeError(
            "Arm did not settle: "
            f"maximum window position error="
            f"{np.rad2deg(max(position_errors)):.3f} deg, "
            f"maximum window RMS speed="
            f"{np.max(rms_velocity):.4f} rad/s."
        )

    def solve_ik(position, rotation, initial_q=None):
        position = np.asarray(position, dtype=float)
        ik_data.qpos[:] = feedback["frames"].kinematics.qpos

        if initial_q is None:
            initial_q = feedback["state_reader"]()[0].copy()

        current_q = np.clip(initial_q, lower, upper)

        def residual(q):
            ik_data.qpos[q_addresses] = q
            mujoco.mj_kinematics(model, ik_data)

            actual_rotation = ik_data.site_xmat[pinch_id].reshape(3, 3)
            position_error = ik_data.site_xpos[pinch_id] - position
            rotation_error = Rotation.from_matrix(
                rotation @ actual_rotation.T
            ).as_rotvec()

            return np.r_[position_error, 0.15 * rotation_error]

        solution = solve_pose(
            residual, current_q, lower, upper, 200,
        )
        if solution is not None:
            return solution

        raise RuntimeError(
            f"IK failed for position {np.round(position, 3)} metres."
        )

    def move_to(position, rotation, label=None, speed_deg=MAX_JOINT_SPEED_DEG):
        nonlocal goal_q

        if label:
            print(label)

        start_q = feedback["state_reader"]()[0].copy()
        goal_q = solve_ik(position, rotation)

        largest_change_deg = np.max(
            np.abs(np.rad2deg(goal_q - start_q))
        )
        duration = max(2.0, 1.875 * largest_change_deg / speed_deg)
        times = np.arange(int(np.ceil(duration / dt)) + 1) * dt
        trajectory = jtraj(start_q, goal_q, times)

        for q, qd in zip(trajectory.q, trajectory.qd):
            yield q.copy(), qd.copy(), gripper_command

        yield from settle()

    def move_line(
        start_position, end_position, rotation, label, speed=VERTICAL_SPEED,
    ):
        """Cartesian line with joint speed/acceleration timing."""
        nonlocal goal_q
        print(label)

        if speed <= 0:
            raise ValueError("Movement speed must be positive.")

        start_position = np.asarray(start_position, dtype=float)
        end_position = np.asarray(end_position, dtype=float)

        path = np.linspace(0.0, 1.0, 15)
        joint_points = [feedback["state_reader"]()[0].copy()]

        for fraction in path[1:]:
            position = start_position + fraction * (
                end_position - start_position
            )
            solution = solve_ik(
                position, rotation, initial_q=joint_points[-1],
            )

            if np.max(np.abs(solution - joint_points[-1])) > np.deg2rad(45):
                raise RuntimeError(
                    "Clearance movement encountered an IK configuration jump."
                )

            joint_points.append(solution)

        curve = CubicSpline(
            path, np.asarray(joint_points), axis=0, bc_type="natural",
        )

        u = np.linspace(0.0, 1.0, 1001)
        progress = 10 * u**3 - 15 * u**4 + 6 * u**5
        progress_rate = 30 * u**2 - 60 * u**3 + 30 * u**4
        progress_acceleration = 60 * u - 180 * u**2 + 120 * u**3

        unit_velocity = curve(progress, 1) * progress_rate[:, None]
        unit_acceleration = (
            curve(progress, 2) * progress_rate[:, None]**2
            + curve(progress, 1) * progress_acceleration[:, None]
        )

        peak_velocity = float(np.max(np.abs(unit_velocity)))
        peak_acceleration = float(np.max(np.abs(unit_acceleration)))
        acceleration_limit = 0.5

        duration = max(
            1.2,
            1.875 * np.linalg.norm(end_position - start_position) / speed,
            peak_velocity / np.deg2rad(MAX_JOINT_SPEED_DEG),
            np.sqrt(1.1 * peak_acceleration / acceleration_limit),
        )

        steps = max(1, int(np.ceil(duration / dt)))
        duration = steps * dt
        print(f"Trajectory duration: {duration:.2f} s")

        for step in range(steps + 1):
            fraction = step / steps
            progress = (
                10 * fraction**3 - 15 * fraction**4 + 6 * fraction**5
            )
            progress_rate = (
                30 * fraction**2 - 60 * fraction**3 + 30 * fraction**4
            ) / duration

            yield (
                curve(progress),
                curve(progress, 1) * progress_rate,
                gripper_command,
            )

        goal_q = joint_points[-1].copy()
        yield from settle()

    def set_gripper(target):
        nonlocal gripper_command
        start_command = gripper_command
        steps = max(1, int(np.ceil(1.0 / dt)))

        for index in range(steps):
            fraction = (index + 1) / steps
            gripper_command = (
                start_command + fraction * (target - start_command)
            )
            yield goal_q.copy(), np.zeros(6), gripper_command

        yield from hold(1.0)

    def refresh_target(name, excluded_views=()):
        nonlocal goal_q
        print(f"{name}: use recorded position to reacquire target.")

        recorded = next(cube for cube in catalog if cube["name"] == name)
        recorded_position = np.asarray(recorded["centre_world"], dtype=float)
        xy = recorded_position[:2].copy()
        inward_xy = xy.copy()
        radius = np.linalg.norm(inward_xy)

        if radius > 0.30:
            inward_xy *= 0.30 / radius

        tangent = (
            np.array([-xy[1], xy[0]]) / radius
            if radius > 1e-6 else np.array([0.0, 1.0])
        )

        viewpoints = [
            [xy[0], xy[1], 0.35],
            [inward_xy[0], inward_xy[1], 0.35],
            [
                inward_xy[0] + 0.04 * tangent[0],
                inward_xy[1] + 0.04 * tangent[1], 0.35,
            ],
            [
                inward_xy[0] - 0.04 * tangent[0],
                inward_xy[1] - 0.04 * tangent[1], 0.35,
            ],
            [xy[0], xy[1], 0.30],
            [inward_xy[0], inward_xy[1], 0.30],
        ]

        feedback.update(
            mode="measure", name=name, collecting=False,
            result=None, servo=None,
        )
        feedback["history"].clear()
        attempted = set()

        try:
            for position in viewpoints:
                key = tuple(np.round(position, 4))
                if key in attempted or key in excluded_views:
                    continue

                attempted.add(key)
                feedback["collecting"] = False
                feedback["history"].clear()
                print(
                    f"{name}: trying camera view {np.round(position, 3)} m"
                )

                observer = WorkspaceScanner(
                    model, data, q_addresses, v_addresses, camera_id,
                    positions=[position],
                    state_reader=feedback["state_reader"],
                )

                try:
                    for q, qd, command in observer.run():
                        goal_q = q.copy()
                        feedback["collecting"] = observer.collecting
                        yield q, qd, command

                        if (
                            observer.collecting
                            and feedback["result"] is not None
                        ):
                            feedback["pickup_view"] = key
                            return dict(feedback["result"])

                except RuntimeError as error:
                    if str(error).startswith("Camera IK failed"):
                        print(f"{name}: camera view skipped:", error)
                        continue
                    raise

            raise RuntimeError(
                f"No stable fresh measurement for {name} "
                "after trying recorded-location views."
            )
        finally:
            feedback.update(mode="idle", collecting=False)

    def run_servo(target, phase, held_size=None, servo=None):
        nonlocal goal_q

        if servo is None:
            servo = PoseServo(
                model, feedback["frames"], v_addresses, target,
                phase=phase, held_size=held_size,
            )

        feedback.update(
            mode="pbvs", name=target["name"],
            servo=servo, collecting=False,
        )
        print(f"PBVS {phase}: tracking {target['name']}.")
        start_time = data.time

        if phase != "place_transport":
            goal_q = feedback["state_reader"]()[0].copy()

        smoothed_rates = (
            feedback["state_reader"]()[1].copy()
            if servo.placing else np.zeros(6)
        )
        smoothing_alpha = (
            1.0 - np.exp(-dt / PLACEMENT_COMMAND_TIME_CONSTANT)
        )

        try:
            while data.time - start_time < (90.0 if servo.placing else 45.0):
                if servo.last_seen is None:
                    if data.time - start_time > 1.0:
                        raise RuntimeError(
                            f"PBVS {phase}: target not acquired."
                        )
                elif data.time - servo.last_seen > LOST_TIMEOUT:
                    reason = servo.rejection_reason or "No recent valid frame."
                    raise RuntimeError(
                        f"PBVS {phase}: target measurement lost; "
                        f"motion paused. {reason}"
                    )

                if servo.converged and (
                    not servo.placing
                    or np.max(np.abs(smoothed_rates)) < 0.002
                ):
                    print(
                        f"PBVS {phase} converged: "
                        f"translation={servo.translation_error * 1000:.1f} mm, "
                        f"rotation={np.rad2deg(servo.rotation_error):.1f} deg"
                    )
                    return servo

                rates = servo.command.copy() if servo.valid else np.zeros(6)

                if servo.placing:
                    if not servo.valid:
                        smoothed_rates[:] = 0.0
                    else:
                        change = smoothing_alpha * (rates - smoothed_rates)
                        smoothed_rates += np.clip(
                            change,
                            -PLACEMENT_JOINT_ACCELERATION * dt,
                            PLACEMENT_JOINT_ACCELERATION * dt,
                        )
                    rates = smoothed_rates.copy()

                goal_q = goal_q + rates * dt

                if (
                    np.any(goal_q <= lower + 0.01)
                    or np.any(goal_q >= upper - 0.01)
                ):
                    raise RuntimeError("PBVS: joint-limit margin reached.")

                measured_q = feedback["frames"].kinematics.qpos[q_addresses]
                if np.max(np.abs(goal_q - measured_q)) > np.deg2rad(5):
                    raise RuntimeError(
                        "PBVS: commanded position exceeds encoder tracking margin."
                    )

                yield goal_q.copy(), rates, gripper_command

            raise RuntimeError(
                f"PBVS {phase}: timed out without convergence."
            )
        finally:
            feedback.update(mode="idle", servo=None)

    def transport_to_support(support, size, height, rotation, name):
        """Use recorded vision first, then hand motion to live PBVS."""
        nonlocal goal_q
        frames = feedback["frames"]
        support_world = frames.base_to_world(support["centre_base"])
        destination = np.array([
            support_world[0], support_world[1], height,
        ])

        kin = frames.kinematics
        rb = kin.xmat[frames.base_id].reshape(3, 3)
        transport_height_base = float(
            (rb.T @ (destination - kin.xpos[frames.base_id]))[2]
        )

        servo = PoseServo(
            model, frames, v_addresses, dict(support),
            phase="place_transport", held_size=size,
            transport_height=transport_height_base,
        )
        feedback.update(
            mode="transport", name=support["name"], servo=servo,
            collecting=False, transport_valid_frames=0,
        )

        try:
            goal_q = feedback["state_reader"]()[0].copy()
            look_start = data.time

            while data.time - look_start < 0.25:
                yield goal_q.copy(), np.zeros(6), gripper_command

                if feedback["transport_valid_frames"] >= 3:
                    print(
                        f"{name}: target visible; transport handed to PBVS."
                    )
                    return (yield from run_servo(
                        support, "place_transport",
                        held_size=size, servo=servo,
                    ))

            movement = move_to(
                destination, rotation,
                f"{name}: transport towards recorded placement region.",
                speed_deg=TRANSPORT_JOINT_SPEED_DEG,
            )

            for q, qd, command in movement:
                goal_q = q.copy()
                yield q, qd, command

                if feedback["transport_valid_frames"] >= 3:
                    print(
                        f"{name}: target visible; transport handed to PBVS."
                    )
                    movement.close()
                    return (yield from run_servo(
                        support, "place_transport",
                        held_size=size, servo=servo,
                    ))

            servo.last_seen = None
            servo.valid = False
            servo.converged = False
            servo.stable_frames = 0
            servo.command[:] = 0.0

            print(
                f"{name}: at recorded region; acquiring target for PBVS."
            )
            return (yield from run_servo(
                support, "place_transport", held_size=size, servo=servo,
            ))
        finally:
            feedback.update(
                mode="idle", servo=None, transport_valid_frames=0,
            )

    table_z = data.geom_xpos[table_id, 2] + model.geom_size[table_id, 2]
    support = dict(feedback["placement"])
    ordered = sorted(catalog, key=lambda cube: cube["size"], reverse=True)
    largest_size = max(cube["size"] for cube in ordered)

    print(
        "Stacking using PBVS pickup and placement "
        "alignment with trajectory lowering."
    )

    for cube in ordered:
        name = cube["name"]
        fresh = yield from refresh_target(name)

        if abs(fresh["size"] - cube["size"]) > 0.003:
            raise RuntimeError(
                f"{name}: fresh size inconsistent with recorded size."
            )

        size = float(fresh["size"])
        half_size = size / 2
        body_id = get_id(
            model, mujoco.mjtObj.mjOBJ_BODY,
            f"cube_{COLOUR_TO_INDEX[name]}",
        )

        yield from set_gripper(0.0)
        failed_views = set()

        for attempt in range(PICK_APPROACH_ATTEMPTS):
            try:
                approach = yield from run_servo(fresh, "pick_approach")
                break
            except RuntimeError as error:
                if not str(error).startswith(
                    "PBVS pick_approach: target measurement lost"
                ):
                    raise

                if attempt + 1 >= PICK_APPROACH_ATTEMPTS:
                    raise RuntimeError(
                        f"{name}: pickup approach lost vision after "
                        f"{PICK_APPROACH_ATTEMPTS} attempts. {error}"
                    ) from error

                print(f"{name}: {error}")
                print(
                    f"{name}: holding, then reobserving from another view "
                    f"(attempt {attempt + 2}/{PICK_APPROACH_ATTEMPTS})."
                )
                failed_views.add(feedback["pickup_view"])
                goal_q = feedback["state_reader"]()[0].copy()
                yield from hold(0.3)

                fresh = yield from refresh_target(
                    name, excluded_views=failed_views,
                )
                if abs(fresh["size"] - size) > 0.003:
                    raise RuntimeError(
                        f"{name}: recovery size inconsistent with recorded size."
                    )

        pick = yield from run_servo(
            dict(approach.last_target), "pick_descend",
        )
        grasp_rotation = pick.grasp_rotation_world.copy()
        cube_position = feedback["frames"].base_to_world(
            pick.last_target["centre_base"]
        )

        print(f"{name}: close gripper.")
        yield from set_gripper(255.0)

        current = estimated_pinch_position()
        clear_z = max(
            current[2] + 0.08,
            table_z + largest_size + LIFT_CLEARANCE
            + half_size + GRASP_OFFSET_Z,
        )
        lifted = np.array([current[0], current[1], clear_z])

        yield from move_line(
            current, lifted, grasp_rotation,
            f"{name}: lift clear of cubes.",
        )
        yield from hold(0.5)

        observed_cube = yield from verify_cube_position(name, size)

        if observed_cube[2] < cube_position[2] + 0.04:
            raise RuntimeError(
                f"{name}: RGB-D measurement shows insufficient lift."
            )

        print(f"{name}: lift visually confirmed.")

        inward_xy = lifted[:2].copy()
        radius = np.linalg.norm(inward_xy)
        if radius > INWARD_RADIUS:
            inward_xy *= INWARD_RADIUS / radius

        inward = np.array([inward_xy[0], inward_xy[1], clear_z])

        if np.linalg.norm(inward - lifted) > 0.001:
            yield from move_line(
                estimated_pinch_position(), inward, grasp_rotation,
                f"{name}: move inward before raising.",
            )

        transport_z = max(TRAVEL_Z, clear_z)
        transport = np.array([
            inward_xy[0], inward_xy[1], transport_z,
        ])

        yield from move_line(
            estimated_pinch_position(), transport, grasp_rotation,
            f"{name}: raise at inward position.",
        )

        transport_servo = yield from transport_to_support(
            support, size, transport_z, grasp_rotation, name,
        )
        support = dict(transport_servo.last_target)

        # Fresh visual check after transport.
        transported_cube = yield from verify_cube_position(name, size)

        if transported_cube[2] < table_z + half_size + 0.04:
            raise RuntimeError(
                f"{name}: RGB-D measurement shows insufficient "
                "cube clearance after transport."
            )

        print(
            f"{name}: cube clearance visually confirmed after transport."
        )

        placement_approach = yield from run_servo(
            support, "place_approach", held_size=size,
        )
        release_position = placement_approach.grasp_world.copy()
        release_rotation = placement_approach.grasp_rotation_world.copy()
        measured_support = dict(placement_approach.last_target)
        measured_support_world = feedback["frames"].base_to_world(
            measured_support["centre_base"]
        )
        stack_xy = measured_support_world[:2].copy()

        current = estimated_pinch_position()

        if np.linalg.norm(current[:2] - release_position[:2]) > 0.005:
            raise RuntimeError(
                "Placement alignment exceeds 5 mm XY margin."
            )

        release_position[:2] = current[:2]

        if release_position[2] >= current[2]:
            raise RuntimeError(
                "Measured release height is not below alignment pose."
            )

        yield from move_line(
            current, release_position, release_rotation,
            f"{name}: trajectory lowering after PBVS alignment.",
            speed=PLACEMENT_SPEED,
        )

        print(f"{name}: release after trajectory lowering.")
        yield from set_gripper(0.0)

        current = estimated_pinch_position()
        yield from move_line(
            current, current + np.array([0.0, 0.0, 0.08]),
            release_rotation, f"{name}: withdraw.",
        )
        yield from hold(1.5)

        # Existing final-placement check, retained for the next step.
        observed_position = yield from verify_cube_position(name, size)

        expected_z = measured_support_world[2] + half_size
        xy_error = np.linalg.norm(observed_position[:2] - stack_xy)
        z_error = abs(observed_position[2] - expected_z)

        if xy_error > 0.025 or z_error > 0.02:
            raise RuntimeError(
                f"{name}: visual placement check failed: "
                f"XY error={xy_error * 1000:.1f} mm, "
                f"Z error={z_error * 1000:.1f} mm."
            )

        print(
            f"{name}: placement visually confirmed: "
            f"XY error={xy_error * 1000:.1f} mm, "
            f"Z error={z_error * 1000:.1f} mm."
        )

        next_top = measured_support["centre_base"].copy()
        next_top[2] += size

        frames = feedback["frames"]
        base_rotation_world = frames.kinematics.xmat[
            frames.base_id
        ].reshape(3, 3)
        release_rotation_base = base_rotation_world.T @ release_rotation
        placed_yaw = float(np.arctan2(
            release_rotation_base[1, 0], release_rotation_base[0, 0],
        ))

        support = {
            "name": name,
            "size": size,
            "yaw": placed_yaw,
            "centre_base": next_top,
            "surface": True,
        }

    print("Returning to observation after stacking.")

    observer = WorkspaceScanner(
        model, data, q_addresses, v_addresses, camera_id,
        positions=[[0.30, 0.16, 0.35]],
        state_reader=feedback["state_reader"],
    )

    for q, qd, command in observer.run():
        goal_q = q.copy()
        yield q, qd, command

    print("Stack sequence completed. Holding final pose.")

    while True:
        yield goal_q.copy(), np.zeros(6), gripper_command