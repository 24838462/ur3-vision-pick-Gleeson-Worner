import mujoco
import numpy as np

from roboticstoolbox import jtraj
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# Camera positions covering the configured tabletop workspace.
# These are planned search locations, not object coordinates.
SEARCH_POSITIONS = [
    [0.30, 0.16, 0.35],
    [0.30, 0.00, 0.35],
    [0.30, -0.15, 0.35],
]

SEARCH_SPEED_DEG = 30.0
OBSERVATION_HOLD_SECONDS = 2.5


class WorkspaceScanner:
    """Move through observation poses and indicate when images are stable."""

    def __init__(
        self,
        model,
        data,
        q_addresses,
        v_addresses,
        camera_id,
    ):
        self.model = model
        self.data = data
        self.q_addresses = q_addresses
        self.v_addresses = v_addresses
        self.camera_id = camera_id

        self.dt = model.opt.timestep
        self.kinematics = mujoco.MjData(model)

        self.collecting = False
        self.status = "Starting workspace search"
        self.goal_q = data.qpos[q_addresses].copy()

        self.lower = np.full(6, -2 * np.pi)
        self.upper = np.full(6, 2 * np.pi)

        for index, address in enumerate(q_addresses):
            matching = np.flatnonzero(
                model.jnt_qposadr == address
            )
            joint_id = int(matching[0])

            if model.jnt_limited[joint_id]:
                self.lower[index], self.upper[index] = (
                    model.jnt_range[joint_id]
                )

        self.lower += 0.001
        self.upper -= 0.001

    def solve_camera_ik(self, position):
        position = np.asarray(position, dtype=float)

        # MuJoCo camera looks along -Z.
        # Identity orientation looks down along world -Z.
        desired_rotation = np.eye(3)

        current_q = np.clip(
            self.data.qpos[self.q_addresses].copy(),
            self.lower,
            self.upper,
        )

        def residual(q):
            self.kinematics.qpos[self.q_addresses] = q

            mujoco.mj_kinematics(
                self.model,
                self.kinematics,
            )
            mujoco.mj_camlight(
                self.model,
                self.kinematics,
            )

            position_error = (
                self.kinematics.cam_xpos[self.camera_id]
                - position
            )

            actual_rotation = (
                self.kinematics.cam_xmat[self.camera_id]
                .reshape(3, 3)
            )

            rotation_error = Rotation.from_matrix(
                desired_rotation @ actual_rotation.T
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
                np.clip(
                    alternative,
                    self.lower,
                    self.upper,
                )
            )

        for seed in seeds:
            result = least_squares(
                residual,
                seed,
                bounds=(self.lower, self.upper),
                max_nfev=300,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )

            error = residual(result.x)

            if (
                np.linalg.norm(error[:3]) < 0.002
                and np.linalg.norm(error[3:]) / 0.15
                < np.deg2rad(2)
            ):
                return result.x.copy()

        raise RuntimeError(
            "Camera IK failed at "
            f"{np.round(position, 3)} metres."
        )

    def run(self):
        for index, position in enumerate(SEARCH_POSITIONS):
            self.collecting = False
            self.status = f"Moving to search view {index + 1}"

            try:
                goal_q = self.solve_camera_ik(position)
            except RuntimeError as error:
                print("Search view skipped:", error)
                continue

            print(
                f"Search view {index + 1}: "
                f"camera XYZ={np.round(position, 3)}"
            )

            start_q = self.data.qpos[
                self.q_addresses
            ].copy()
            self.goal_q = goal_q.copy()

            largest_change = np.max(
                np.abs(np.rad2deg(goal_q - start_q))
            )
            duration = max(
                3.0,
                1.875 * largest_change / SEARCH_SPEED_DEG,
            )

            steps = int(np.ceil(duration / self.dt))
            times = np.arange(steps + 1) * self.dt
            trajectory = jtraj(start_q, goal_q, times)

            for q, qd in zip(
                trajectory.q,
                trajectory.qd,
            ):
                yield q.copy(), qd.copy(), 0.0

            self.status = f"Settling at search view {index + 1}"

            stable_steps = 0
            required_steps = max(
                1,
                int(np.ceil(0.3 / self.dt)),
            )
            settled = False

            for _ in range(
                int(np.ceil(4.0 / self.dt))
            ):
                yield goal_q.copy(), np.zeros(6), 0.0

                error = np.max(np.abs(
                    goal_q
                    - self.data.qpos[self.q_addresses]
                ))
                speed = np.max(np.abs(
                    self.data.qvel[self.v_addresses]
                ))

                if error < np.deg2rad(1) and speed < 0.05:
                    stable_steps += 1
                else:
                    stable_steps = 0

                if stable_steps >= required_steps:
                    settled = True
                    break

            if not settled:
                raise RuntimeError(
                    "Arm did not settle during workspace search."
                )

            self.collecting = True
            self.status = f"Observing search view {index + 1}"
            print(self.status)

            for _ in range(int(np.ceil(
                OBSERVATION_HOLD_SECONDS / self.dt
            ))):
                yield goal_q.copy(), np.zeros(6), 0.0

            self.collecting = False

        self.status = "Workspace search completed"
        print(self.status)