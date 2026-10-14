import mujoco
import numpy as np

from roboticstoolbox import jtraj
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# Camera positions are in world coordinates, in metres.
SEARCH_HEIGHT = 0.35

# Square containing the cube spawn region.
# Camera travel area; its field of view covers the wider cube region.
SEARCH_X_MIN = 0.22
SEARCH_X_MAX = 0.30
SEARCH_Y_MIN = 0.08
SEARCH_Y_MAX = 0.16

SEARCH_HEIGHT = 0.35
SEARCH_ROWS = 3

# Continue the same sweep towards the purple placement region.
PURPLE_VIEW = np.array([0.30, -0.15, SEARCH_HEIGHT])

# Slow enough to collect several consistent RGB-D measurements.
SCAN_SPEED = 0.025              # Camera travel: metres/second.
MAX_JOINT_SPEED = np.deg2rad(25)
PATH_SAMPLE_SPACING = 0.01      # IK samples every 10 mm.


def make_search_path():
    """Zigzag over the cube area, then observe the purple region."""
    points = []

    rows = np.linspace(
        SEARCH_Y_MAX,
        SEARCH_Y_MIN,
        SEARCH_ROWS,
    )

    for index, y in enumerate(rows):
        if index % 2 == 0:
            xs = [SEARCH_X_MIN, SEARCH_X_MAX]
        else:
            xs = [SEARCH_X_MAX, SEARCH_X_MIN]

        for x in xs:
            points.append([x, y, SEARCH_HEIGHT])

    # Connect to the placement region without returning home.
    points.append([0.30, SEARCH_Y_MIN, SEARCH_HEIGHT])
    points.append(PURPLE_VIEW.tolist())

    return np.asarray(points, dtype=float)


class WorkspaceScanner:
    def __init__(
        self,
        model,
        data,
        q_addresses,
        v_addresses,
        camera_id,
        positions=None,
    ):
        self.model = model
        self.data = data

        self.qa = np.asarray(q_addresses)
        self.va = np.asarray(v_addresses)
        self.cid = camera_id

        # Explicit positions retain the original stationary-view behaviour.
        self.positions = positions

        self.collecting = False
        self.status = "Starting workspace search"

        self.kin = mujoco.MjData(model)

        self.lower = np.full(6, -2.0 * np.pi)
        self.upper = np.full(6, 2.0 * np.pi)

        for index, address in enumerate(self.qa):
            joint_id = int(
                np.flatnonzero(
                    model.jnt_qposadr == address
                )[0]
            )

            if model.jnt_limited[joint_id]:
                self.lower[index], self.upper[index] = (
                    model.jnt_range[joint_id]
                )

        self.lower += 0.001
        self.upper -= 0.001

    def solve(self, position, seed=None):
        """Find an arm pose with the camera looking vertically down."""
        position = np.asarray(position, dtype=float)

        if seed is None:
            seed = self.data.qpos[self.qa]

        current = np.clip(
            np.asarray(seed, dtype=float),
            self.lower,
            self.upper,
        )

        def residual(q):
            self.kin.qpos[self.qa] = q

            mujoco.mj_kinematics(self.model, self.kin)
            mujoco.mj_camlight(self.model, self.kin)

            position_error = (
                self.kin.cam_xpos[self.cid] - position
            )

            camera_rotation = (
                self.kin.cam_xmat[self.cid].reshape(3, 3)
            )

            # Identity MuJoCo camera orientation looks along world -Z.
            rotation_error = Rotation.from_matrix(
                camera_rotation.T
            ).as_rotvec()

            return np.concatenate(
                [position_error, 0.15 * rotation_error]
            )

        seeds = [current]

        for elbow in (90, -90):
            alternative = current.copy()
            alternative[1:3] = np.deg2rad([-60, elbow])

            seeds.append(
                np.clip(
                    alternative,
                    self.lower,
                    self.upper,
                )
            )

        for candidate in seeds:
            result = least_squares(
                residual,
                candidate,
                bounds=(self.lower, self.upper),
                max_nfev=300,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )

            error = residual(result.x)

            position_ok = np.linalg.norm(error[:3]) < 0.002
            rotation_ok = (
                np.linalg.norm(error[3:]) / 0.15
                < np.deg2rad(2)
            )

            if position_ok and rotation_ok:
                return result.x.copy()

        raise RuntimeError(
            f"Camera IK failed at {np.round(position, 3)} m"
        )

    def move_and_settle(self, goal):
        dt = self.model.opt.timestep

        self.collecting = False

        start = self.data.qpos[self.qa].copy()

        duration = max(
            3.0,
            1.875
            * np.max(np.abs(goal - start))
            / MAX_JOINT_SPEED,
        )

        steps = int(np.ceil(duration / dt))
        times = np.arange(steps + 1) * dt
        trajectory = jtraj(start, goal, times)

        for q, qd in zip(trajectory.q, trajectory.qd):
            yield q.copy(), qd.copy(), 0.0

        stable_seconds = 0.0

        for _ in range(int(np.ceil(4.0 / dt))):
            yield goal.copy(), np.zeros(6), 0.0

            position_ok = (
                np.max(
                    np.abs(
                        goal - self.data.qpos[self.qa]
                    )
                )
                < np.deg2rad(1)
            )

            velocity_ok = (
                np.max(
                    np.abs(self.data.qvel[self.va])
                )
                < 0.05
            )

            if position_ok and velocity_ok:
                stable_seconds += dt
            else:
                stable_seconds = 0.0

            if stable_seconds >= 0.3:
                return

        raise RuntimeError(
            "Observation movement did not settle"
        )

    def stationary_views(self):
        """Used to remeasure a cube before its PBVS approach."""
        dt = self.model.opt.timestep

        for index, position in enumerate(self.positions):
            self.status = (
                f"Moving to observation view {index + 1}"
            )
            print(self.status)

            goal = self.solve(position)

            yield from self.move_and_settle(goal)

            self.status = (
                f"Observing view {index + 1}"
            )
            self.collecting = True
            print(self.status)

            for _ in range(int(np.ceil(2.5 / dt))):
                yield goal.copy(), np.zeros(6), 0.0

            self.collecting = False

    def continuous_search(self):
        dt = self.model.opt.timestep
        corners = make_search_path()

        self.status = "Moving to start of zigzag search"
        print(self.status)

        first_goal = self.solve(corners[0])

        yield from self.move_and_settle(first_goal)

        # Densely sample the camera path and warm-start each IK solve.
        distances = [0.0]
        joint_samples = [first_goal]
        previous_q = first_goal

        self.status = "Preparing zigzag search path"
        print(self.status)

        for start, finish in zip(corners[:-1], corners[1:]):
            length = np.linalg.norm(finish - start)

            if length < 1e-9:
                continue

            sample_count = max(
                1,
                int(np.ceil(length / PATH_SAMPLE_SPACING)),
            )

            for fraction in np.linspace(
                0.0,
                1.0,
                sample_count + 1,
            )[1:]:
                position = (
                    start + fraction * (finish - start)
                )

                q = self.solve(position, seed=previous_q)

                # Stop rather than accepting a sudden IK branch change.
                if np.max(np.abs(q - previous_q)) > np.deg2rad(20):
                    raise RuntimeError(
                        "Zigzag IK changed branch. "
                        "Adjust the search region or height."
                    )

                distances.append(
                    distances[-1] + length / sample_count
                )
                joint_samples.append(q)

                previous_q = q

        distances = np.asarray(distances)
        joint_samples = np.asarray(joint_samples)

        path = CubicSpline(
            distances,
            joint_samples,
            axis=0,
            bc_type="natural",
        )

        total_distance = float(distances[-1])

        # Quintic progress gives gentle acceleration and deceleration
        # at the beginning and end of the complete sweep.
        duration = (
            1.875 * total_distance / SCAN_SPEED
        )

        sample_s = np.linspace(
            0.0,
            total_distance,
            max(1000, len(distances) * 10),
        )

        sampled_q = path(sample_s)

        if (
            np.any(sampled_q < self.lower)
            or np.any(sampled_q > self.upper)
        ):
            raise RuntimeError(
                "Interpolated search path exceeds joint limits."
            )

        # Also constrain the joint speeds along the interpolated path.
        maximum_derivative = np.max(
            np.abs(path(sample_s, 1))
        )

        duration = max(
            duration,
            1.875
            * total_distance
            * maximum_derivative
            / MAX_JOINT_SPEED,
        )

        steps = int(np.ceil(duration / dt))
        duration = steps * dt

        self.status = "Scanning cube area and purple region"
        self.collecting = True
        print(self.status)

        for index in range(steps + 1):
            progress = index / steps

            smooth_progress = (
                10.0 * progress**3
                - 15.0 * progress**4
                + 6.0 * progress**5
            )

            progress_rate = (
                30.0 * progress**2
                - 60.0 * progress**3
                + 30.0 * progress**4
            ) / duration

            distance = total_distance * smooth_progress
            distance_rate = total_distance * progress_rate

            q = path(distance)
            qd = path(distance, 1) * distance_rate

            yield q.copy(), qd.copy(), 0.0

        # Allow a final observation of the purple placement region.
        goal = joint_samples[-1]

        self.status = "Observing end of search"

        for _ in range(int(np.ceil(2.5 / dt))):
            yield goal.copy(), np.zeros(6), 0.0

        self.collecting = False

    def run(self):
        try:
            if self.positions is None:
                yield from self.continuous_search()
            else:
                yield from self.stationary_views()

            self.status = "Workspace search completed"

        finally:
            self.collecting = False