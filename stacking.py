import mujoco
import numpy as np

from roboticstoolbox import jtraj
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

STACK_XY = np.array([0.30, -0.15])
TRAVEL_Z = 0.25
GRASP_OFFSET_Z = 0.008
MAX_JOINT_SPEED_DEG = 50.0

# Largest cube at the bottom: blue, green, red.
STACK_ORDER = (2, 1, 0)


def get_id(model, object_type, name):
    index = mujoco.mj_name2id(model, object_type, name)

    if index < 0:
        raise ValueError(f"Missing model element: {name}")

    return index


def stack_cubes(model, data, q_addresses, v_addresses, home):
    """
    Yield arm position, arm velocity and gripper references.

    main.py applies these references and advances the physics.
    """
    dt = model.opt.timestep

    pinch_id = get_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "gripper_pinch",
    )

    table_id = get_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "table_top",
    )

    arm_joint_ids = [
        get_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in JOINT_NAMES
    ]

    # Bounds for numerical inverse kinematics.
    lower = np.full(6, -2 * np.pi)
    upper = np.full(6, 2 * np.pi)

    for index, joint_id in enumerate(arm_joint_ids):
        if model.jnt_limited[joint_id]:
            lower[index], upper[index] = (
                model.jnt_range[joint_id]
            )

    lower += 0.001
    upper -= 0.001

    # Separate data for IK: solving never changes the live robot.
    ik_data = mujoco.MjData(model)

    goal_q = np.asarray(home, dtype=float).copy()
    gripper_command = 0.0

    def hold(seconds):
        steps = max(1, int(np.ceil(seconds / dt)))

        for _ in range(steps):
            yield (
                goal_q.copy(),
                np.zeros(6),
                gripper_command,
            )

    def solve_ik(position, rotation):
        position = np.asarray(position, dtype=float)
        ik_data.qpos[:] = data.qpos

        current_q = np.clip(
            data.qpos[q_addresses].copy(),
            lower,
            upper,
        )

        def residual(q):
            ik_data.qpos[q_addresses] = q

            # Update poses only, avoiding collision/dynamics
            # calculations during the IK search.
            mujoco.mj_kinematics(model, ik_data)

            actual_rotation = (
                ik_data.site_xmat[pinch_id].reshape(3, 3)
            )

            position_error = (
                ik_data.site_xpos[pinch_id] - position
            )

            rotation_error = Rotation.from_matrix(
                rotation @ actual_rotation.T
            ).as_rotvec()

            return np.concatenate([
                position_error,
                0.15 * rotation_error,
            ])

        seeds = [current_q]

        for elbow_deg in (90, -90):
            alternative = current_q.copy()
            alternative[1:3] = np.deg2rad(
                [-60, elbow_deg]
            )
            seeds.append(
                np.clip(alternative, lower, upper)
            )

        for seed in seeds:
            result = least_squares(
                residual,
                seed,
                bounds=(lower, upper),
                max_nfev=200,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )

            error = residual(result.x)

            position_error = np.linalg.norm(error[:3])
            rotation_error = (
                np.linalg.norm(error[3:]) / 0.15
            )

            if (
                position_error < 0.002
                and rotation_error < np.deg2rad(2)
            ):
                return result.x.copy()

        raise RuntimeError(
            "IK failed for position "
            f"{np.round(position, 3)} metres."
        )

    def move_to(position, rotation, label=None):
        nonlocal goal_q

        if label:
            print(label)

        start_q = data.qpos[q_addresses].copy()
        goal_q = solve_ik(position, rotation)

        largest_change_deg = np.max(
            np.abs(np.rad2deg(goal_q - start_q))
        )

        duration = max(
            2.0,
            1.875 * largest_change_deg
            / MAX_JOINT_SPEED_DEG,
        )

        steps = int(np.ceil(duration / dt))
        times = np.arange(steps + 1) * dt
        trajectory = jtraj(start_q, goal_q, times)

        for q, qd in zip(trajectory.q, trajectory.qd):
            yield (
                q.copy(),
                qd.copy(),
                gripper_command,
            )

        # Wait for the actual robot to reach the reference.
        settled_steps = 0
        required_steps = max(1, int(0.15 / dt))

        for _ in range(int(np.ceil(3.0 / dt))):
            yield (
                goal_q.copy(),
                np.zeros(6),
                gripper_command,
            )

            error = np.max(np.abs(
                goal_q - data.qpos[q_addresses]
            ))
            speed = np.max(np.abs(
                data.qvel[v_addresses]
            ))

            if error < np.deg2rad(1) and speed < 0.05:
                settled_steps += 1
            else:
                settled_steps = 0

            if settled_steps >= required_steps:
                return

        raise RuntimeError(
            "Arm did not settle at the requested pose."
        )

    def move_vertical(xy, start_z, end_z, rotation, label):
        nonlocal goal_q

        print(label)

        # Solve positions along the descent before starting movement.
        path = np.linspace(0.0, 1.0, 15)
        joint_points = [data.qpos[q_addresses].copy()]

        for fraction in path[1:]:
            height = start_z + fraction * (end_z - start_z)
            position = np.array([xy[0], xy[1], height])

            solution = solve_ik(position, rotation)

            # Catch a sudden change of IK configuration.
            if np.max(np.abs(solution - joint_points[-1])) > np.deg2rad(45):
                raise RuntimeError(
                    "Vertical movement encountered an IK configuration jump."
                )

            joint_points.append(solution)

        curve = CubicSpline(
            path,
            np.asarray(joint_points),
            axis=0,
            bc_type="natural",
        )

        # Vertical speed in metres per second.
        vertical_speed = 0.1
        joint_speed = np.deg2rad(MAX_JOINT_SPEED_DEG)

        samples = np.linspace(0.0, 1.0, 200)
        peak_joint_slope = np.max(np.abs(curve(samples, 1)))

        duration = max(
            1.2,
            1.875 * abs(end_z - start_z) / vertical_speed,
            1.875 * peak_joint_slope / joint_speed,
        )

        steps = int(np.ceil(duration / dt))
        duration = steps * dt

        # Quintic timing: smooth start, continuous motion, smooth stop.
        for step in range(steps + 1):
            u = step / steps

            progress = 10 * u**3 - 15 * u**4 + 6 * u**5
            progress_rate = (
                30 * u**2 - 60 * u**3 + 30 * u**4
            ) / duration

            desired_q = curve(progress)
            desired_qd = curve(progress, 1) * progress_rate

            yield desired_q, desired_qd, gripper_command

        goal_q = joint_points[-1].copy()

        # Settle once, at the final height.
        stable_steps = 0
        required_steps = max(1, int(np.ceil(0.15 / dt)))

        for _ in range(int(np.ceil(3.0 / dt))):
            yield goal_q.copy(), np.zeros(6), gripper_command

            error = np.max(
                np.abs(goal_q - data.qpos[q_addresses])
            )
            speed = np.max(np.abs(data.qvel[v_addresses]))

            if error < np.deg2rad(1.0) and speed < 0.05:
                stable_steps += 1
                if stable_steps >= required_steps:
                    return
            else:
                stable_steps = 0

        raise RuntimeError("Arm did not settle after vertical movement.")

    def set_gripper(target):
        nonlocal gripper_command

        start_command = gripper_command
        steps = max(1, int(np.ceil(1.0 / dt)))

        for index in range(steps):
            fraction = (index + 1) / steps
            gripper_command = (
                start_command
                + fraction * (target - start_command)
            )

            yield (
                goal_q.copy(),
                np.zeros(6),
                gripper_command,
            )

        yield from hold(1.0)

    table_z = (
        data.geom_xpos[table_id, 2]
        + model.geom_size[table_id, 2]
    )
    stack_height = 0.0

    print("Stacking: settling the scene.")
    yield from hold(2.0)

    for cube_index in STACK_ORDER:
        body_id = get_id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"cube_{cube_index}",
        )
        geom_id = get_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"cube_{cube_index}_geom",
        )

        half_size = model.geom_size[geom_id, 0]
        cube_position = data.xpos[body_id].copy()

        cube_rotation = (
            data.xmat[body_id].reshape(3, 3)
        )
        yaw = np.arctan2(
            cube_rotation[1, 0],
            cube_rotation[0, 0],
        )

        # Pinch-frame Z points down.
        # Its X/Y axes align with the cube edges.
        grasp_rotation = (
            Rotation.from_euler("z", yaw).as_matrix()
            @ np.diag([1.0, -1.0, -1.0])
        )

        grasp_position = cube_position.copy()
        grasp_position[2] += GRASP_OFFSET_Z

        yield from set_gripper(0.0)

        yield from move_to(
            [
                grasp_position[0],
                grasp_position[1],
                TRAVEL_Z,
            ],
            grasp_rotation,
            f"Cube {cube_index}: approach.",
        )

        yield from move_vertical(
            grasp_position[:2],
            TRAVEL_Z,
            grasp_position[2],
            grasp_rotation,
            f"Cube {cube_index}: descend.",
        )

        print(f"Cube {cube_index}: close gripper.")
        yield from set_gripper(255.0)

        yield from move_to(
            [
                grasp_position[0],
                grasp_position[1],
                grasp_position[2] + 0.08,
            ],
            grasp_rotation,
            f"Cube {cube_index}: lift.",
        )
        yield from hold(0.5)

        if data.xpos[body_id, 2] < cube_position[2] + 0.04:
            raise RuntimeError(
                f"Cube {cube_index} did not lift. "
                "Check grasp height and finger contacts."
            )

        # Use the measured offset of the physically held cube.
        held_offset = (
            data.site_xpos[pinch_id]
            - data.xpos[body_id]
        ).copy()

        yield from move_to(
            [
                grasp_position[0],
                grasp_position[1],
                TRAVEL_Z,
            ],
            grasp_rotation,
            f"Cube {cube_index}: raise for transport.",
        )

        yield from move_to(
            [STACK_XY[0], STACK_XY[1], TRAVEL_Z],
            grasp_rotation,
            f"Cube {cube_index}: move above stack.",
        )

        if data.xpos[body_id, 2] < table_z + half_size + 0.04:
            raise RuntimeError(
                f"Cube {cube_index} slipped during transport."
            )

        desired_cube_position = np.array([
            STACK_XY[0],
            STACK_XY[1],
            table_z + stack_height + half_size + 0.003,
        ])

        release_position = (
            desired_cube_position + held_offset
        )

        yield from move_vertical(
            release_position[:2],
            TRAVEL_Z,
            release_position[2],
            grasp_rotation,
            f"Cube {cube_index}: lower onto stack.",
        )

        print(f"Cube {cube_index}: release.")
        yield from set_gripper(0.0)

        yield from move_to(
            [
                release_position[0],
                release_position[1],
                release_position[2] + 0.08,
            ],
            grasp_rotation,
            f"Cube {cube_index}: withdraw.",
        )
        yield from hold(1.5)

        actual_position = data.xpos[body_id].copy()
        expected_z = (
            table_z + stack_height + half_size
        )

        xy_error = np.linalg.norm(
            actual_position[:2] - STACK_XY
        )
        z_error = abs(actual_position[2] - expected_z)

        if xy_error > 0.025 or z_error > 0.02:
            raise RuntimeError(
                f"Cube {cube_index} placement failed: "
                f"XY error={xy_error * 1000:.1f} mm, "
                f"Z error={z_error * 1000:.1f} mm."
            )

        stack_height += 2 * half_size
        print(f"Cube {cube_index}: placement confirmed.")

    print("Stack sequence completed. Holding final pose.")

    while True:
        yield (
            goal_q.copy(),
            np.zeros(6),
            gripper_command,
        )