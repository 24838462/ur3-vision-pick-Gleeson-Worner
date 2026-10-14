import mujoco
import numpy as np
from roboticstoolbox import jtraj
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from robot_utils import joint_limits, solve_pose
SEARCH_HEIGHT = 0.35
SEARCH_X_MIN = 0.22
SEARCH_X_MAX = 0.3
SEARCH_Y_MIN = 0.08
SEARCH_Y_MAX = 0.16
SEARCH_ROWS = 3
PURPLE_VIEW = np.array([0.3, -0.15, SEARCH_HEIGHT])
SCAN_SPEED = 0.025
MAX_JOINT_SPEED = np.deg2rad(25)
PATH_SAMPLE_SPACING = 0.01

def make_search_path():
    """Zigzag over the cube area, then observe the purple region."""
    points = []
    rows = np.linspace(SEARCH_Y_MAX, SEARCH_Y_MIN, SEARCH_ROWS)
    for index, y in enumerate(rows):
        if index % 2 == 0:
            xs = [SEARCH_X_MIN, SEARCH_X_MAX]
        else:
            xs = [SEARCH_X_MAX, SEARCH_X_MIN]
        for x in xs:
            points.append([x, y, SEARCH_HEIGHT])
    points.append([0.3, SEARCH_Y_MIN, SEARCH_HEIGHT])
    points.append(PURPLE_VIEW.tolist())
    return np.asarray(points, dtype=float)

class WorkspaceScanner:

    def __init__(self, model, data, q_addresses, v_addresses, camera_id, positions=None):
        self.model = model
        self.data = data
        self.qa = np.asarray(q_addresses)
        self.va = np.asarray(v_addresses)
        self.cid = camera_id
        self.positions = positions
        self.collecting = False
        self.status = 'Starting workspace search'
        self.kin = mujoco.MjData(model)
        self.lower, self.upper = joint_limits(model, self.qa)

    def solve(self, position, seed=None):
        """Find an arm pose with the camera looking vertically down."""
        position = np.asarray(position, dtype=float)
        if seed is None:
            seed = self.data.qpos[self.qa]
        current = np.clip(np.asarray(seed, dtype=float), self.lower, self.upper)

        def residual(q):
            self.kin.qpos[self.qa] = q
            mujoco.mj_kinematics(self.model, self.kin)
            mujoco.mj_camlight(self.model, self.kin)
            position_error = self.kin.cam_xpos[self.cid] - position
            camera_rotation = self.kin.cam_xmat[self.cid].reshape(3, 3)
            rotation_error = Rotation.from_matrix(camera_rotation.T).as_rotvec()
            return np.concatenate([position_error, 0.15 * rotation_error])
        solution = solve_pose(residual, current, self.lower, self.upper, 300)
        if solution is not None:
            return solution
        raise RuntimeError(f'Camera IK failed at {np.round(position, 3)} m')

    def move_and_settle(self, goal):
        dt = self.model.opt.timestep
        self.collecting = False
        start = self.data.qpos[self.qa].copy()
        duration = max(3.0, 1.875 * np.max(np.abs(goal - start)) / MAX_JOINT_SPEED)
        steps = int(np.ceil(duration / dt))
        times = np.arange(steps + 1) * dt
        trajectory = jtraj(start, goal, times)
        for q, qd in zip(trajectory.q, trajectory.qd):
            yield (q.copy(), qd.copy(), 0.0)
        stable_seconds = 0.0
        for _ in range(int(np.ceil(4.0 / dt))):
            yield (goal.copy(), np.zeros(6), 0.0)
            position_ok = np.max(np.abs(goal - self.data.qpos[self.qa])) < np.deg2rad(1)
            velocity_ok = np.max(np.abs(self.data.qvel[self.va])) < 0.05
            if position_ok and velocity_ok:
                stable_seconds += dt
            else:
                stable_seconds = 0.0
            if stable_seconds >= 0.3:
                return
        raise RuntimeError('Observation movement did not settle')

    def stationary_views(self):
        """Used to remeasure a cube before its PBVS approach."""
        dt = self.model.opt.timestep
        for index, position in enumerate(self.positions):
            self.status = f'Moving to observation view {index + 1}'
            print(self.status)
            goal = self.solve(position)
            yield from self.move_and_settle(goal)
            self.status = f'Observing view {index + 1}'
            self.collecting = True
            print(self.status)
            for _ in range(int(np.ceil(2.5 / dt))):
                yield (goal.copy(), np.zeros(6), 0.0)
            self.collecting = False

    def continuous_search(self):
        dt = self.model.opt.timestep
        corners = make_search_path()
        self.status = 'Moving to start of zigzag search'
        print(self.status)
        first_goal = self.solve(corners[0])
        yield from self.move_and_settle(first_goal)
        distances = [0.0]
        joint_samples = [first_goal]
        previous_q = first_goal
        self.status = 'Preparing zigzag search path'
        print(self.status)
        for start, finish in zip(corners[:-1], corners[1:]):
            length = np.linalg.norm(finish - start)
            if length < 1e-09:
                continue
            sample_count = max(1, int(np.ceil(length / PATH_SAMPLE_SPACING)))
            for fraction in np.linspace(0.0, 1.0, sample_count + 1)[1:]:
                position = start + fraction * (finish - start)
                q = self.solve(position, seed=previous_q)
                if np.max(np.abs(q - previous_q)) > np.deg2rad(20):
                    raise RuntimeError('Zigzag IK changed branch. Adjust the search region or height.')
                distances.append(distances[-1] + length / sample_count)
                joint_samples.append(q)
                previous_q = q
        distances = np.asarray(distances)
        joint_samples = np.asarray(joint_samples)
        path = CubicSpline(distances, joint_samples, axis=0, bc_type='natural')
        total_distance = float(distances[-1])
        duration = 1.875 * total_distance / SCAN_SPEED
        sample_s = np.linspace(0.0, total_distance, max(1000, len(distances) * 10))
        sampled_q = path(sample_s)
        if np.any(sampled_q < self.lower) or np.any(sampled_q > self.upper):
            raise RuntimeError('Interpolated search path exceeds joint limits.')
        maximum_derivative = np.max(np.abs(path(sample_s, 1)))
        duration = max(duration, 1.875 * total_distance * maximum_derivative / MAX_JOINT_SPEED)
        steps = int(np.ceil(duration / dt))
        duration = steps * dt
        self.status = 'Scanning cube area and purple region'
        self.collecting = True
        print(self.status)
        for index in range(steps + 1):
            progress = index / steps
            smooth_progress = 10.0 * progress ** 3 - 15.0 * progress ** 4 + 6.0 * progress ** 5
            progress_rate = (30.0 * progress ** 2 - 60.0 * progress ** 3 + 30.0 * progress ** 4) / duration
            distance = total_distance * smooth_progress
            distance_rate = total_distance * progress_rate
            q = path(distance)
            qd = path(distance, 1) * distance_rate
            yield (q.copy(), qd.copy(), 0.0)
        goal = joint_samples[-1]
        self.status = 'Observing end of search'
        for _ in range(int(np.ceil(2.5 / dt))):
            yield (goal.copy(), np.zeros(6), 0.0)
        self.collecting = False

    def run(self):
        try:
            if self.positions is None:
                yield from self.continuous_search()
            else:
                yield from self.stationary_views()
            self.status = 'Workspace search completed'
        finally:
            self.collecting = False
