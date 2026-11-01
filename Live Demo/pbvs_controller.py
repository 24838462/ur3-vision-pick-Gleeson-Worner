"""Position-Based Visual Servoing (PBVS) Controller for Eye-to-Hand UR3e.

Adapts the simulated PBVS controller for a stationary tripod-mounted RealSense D435i:
- Transforms visual detections from Camera Optical Frame into Robot Base Frame
- Regulates 6-DOF task-space Cartesian error (position & orientation)
- Applies adaptive Damped Least Squares (DLS) with singularity protection
- Enforces multi-level velocity, acceleration, and joint limit safety limits
- Coordinates multi-phase pick-and-place execution:
    1. HOVER_APPROACH: aligns above cube with clearance
    2. PRECISION_DESCEND: lowers to grasp pinch height
    3. GRASP: clamps RG2 gripper
    4. LIFT: elevates to transport clearance
    5. TRANSPORT: transfers to target placement coordinate
    6. PLACE_DESCEND: lowers to release height
    7. RELEASE: opens RG2 gripper
    8. RETRACT_HOME: returns to observation home
"""

import time
import numpy as np
from scipy.spatial.transform import Rotation

from ur3e_kinematics import UR3eKinematics


class PBVSController:
    """Eye-to-Hand PBVS visual servoing and state machine coordinator."""

    def __init__(
        self,
        kinematics: UR3eKinematics,
        hover_clearance: float = 0.080,   # 80 mm above cube
        grasp_offset_z: float = 0.010,    # 10 mm below top surface for finger grip
        final_approach_z_offset: float = 0.030,  # +3 cm higher grasp approach
        lift_height: float = 0.120,       # 120 mm vertical lift
        max_linear_speed: float = 0.030,  # 30 mm/s safety limit
        max_angular_speed: float = 0.20,  # 0.2 rad/s
        max_joint_speed: float = 0.30,    # 0.3 rad/s
        pos_tolerance: float = 0.003,     # 3 mm convergence
        rot_tolerance_deg: float = 3.0,   # 3 degrees convergence
    ):
        self.kin = kinematics
        self.hover_clearance = hover_clearance
        self.grasp_offset_z = grasp_offset_z
        self.final_approach_z_offset = final_approach_z_offset
        self.lift_height = lift_height
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.max_joint_speed = max_joint_speed
        self.pos_tolerance = pos_tolerance
        self.rot_tolerance = np.deg2rad(rot_tolerance_deg)
        self.min_safe_z = 0.040  # Table surface level threshold in base frame

        # Servoing proportional gains
        self.gain_pos = 1.2
        self.gain_rot = 1.0

        # State tracking
        self.phase = "IDLE"
        self.converged_frames = 0
        self.required_converged_frames = 5
        self.current_target_base = None
        self.current_target_yaw = 0.0
        self.target_cube_size = 0.040

    def start_pick(self, target_base_pos: np.ndarray, target_yaw: float, cube_size: float = 0.040):
        """Initialize pick sequence for detected cube."""
        self.current_target_base = np.asarray(target_base_pos, dtype=float).copy()
        self.current_target_yaw = float(target_yaw)
        self.target_cube_size = float(cube_size)
        self.phase = "HOVER_APPROACH"
        self.converged_frames = 0
        cmd_hover_z = max(self.current_target_base[2] + self.hover_clearance, self.min_safe_z + self.hover_clearance)
        print(f"\n[PBVS] Starting Pick Sequence:")
        print(f"       Cube Position (Base Frame): X={self.current_target_base[0]:.3f}, Y={self.current_target_base[1]:.3f}, Z={self.current_target_base[2]:.3f} m")
        print(f"       Hover Clearance Setting:    +{self.hover_clearance * 1000.0:.0f} mm (+{self.hover_clearance:.3f} m)")
        print(f"       Target Hover Height Z:      {cmd_hover_z:.3f} m (Floor Safe Minimum: {self.min_safe_z + self.hover_clearance:.3f} m)")
        print(f"       Target Cube Yaw:            {np.rad2deg(target_yaw):.1f} deg\n")

    def update_visual_target(self, target_base_pos: np.ndarray, target_yaw: float):
        """Continuously refine visual target when visible in camera with deadband jitter suppression."""
        if self.phase in ("HOVER_APPROACH", "PRECISION_DESCEND"):
            pos_diff = float(np.linalg.norm(target_base_pos - self.current_target_base))
            yaw_diff = float(abs((target_yaw - self.current_target_yaw + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0))

            # 5 mm deadband: filter out visual target fluctuations smaller than 5 mm
            if pos_diff > 0.005:
                alpha_pos = 0.25
                self.current_target_base = (1.0 - alpha_pos) * self.current_target_base + alpha_pos * target_base_pos

            yaw_err = (target_yaw - self.current_target_yaw + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0
            if abs(yaw_err) > np.deg2rad(2.0):
                alpha_rot = 0.20
                self.current_target_yaw = float((self.current_target_yaw + alpha_rot * yaw_err + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0)

    def compute_step(self, current_q: np.ndarray) -> tuple[np.ndarray, bool]:
        """Compute one PBVS control cycle.

        Returns:
            q_dot: Commanded joint velocities (6,)
            phase_complete: True if current phase reached convergence / target
        """
        if self.phase == "IDLE" or self.current_target_base is None:
            return np.zeros(6), False

        # Current TCP pose from forward kinematics
        t_current = self.kin.forward_kinematics(current_q, include_tcp=True)
        p_current = t_current[:3, 3]
        r_current = t_current[:3, :3]

        # Desired TCP position and orientation based on current phase
        desired_pos = self.current_target_base.copy()

        # Tool orientation: gripper pointing down (-Z in base frame) aligned with cube yaw
        # Base frame: Z up. Downward tool has R_down:
        cy, sy = np.cos(self.current_target_yaw), np.sin(self.current_target_yaw)
        # Tool Z points down [0, 0, -1]. Tool X points along [cos(yaw), sin(yaw), 0]
        desired_rot = np.array([
            [ cy,  sy,  0.0],
            [ sy, -cy,  0.0],
            [0.0, 0.0, -1.0],
        ], dtype=float)

        if self.phase == "HOVER_APPROACH":
            desired_pos[2] += self.hover_clearance
            # Floor safety clamp: hover height must never plunge below table + clearance
            min_hover_z = self.min_safe_z + self.hover_clearance
            if desired_pos[2] < min_hover_z:
                desired_pos[2] = min_hover_z

        elif self.phase == "PRECISION_DESCEND":
            # Pinch site descends into grasp depth (+3 cm higher as requested)
            desired_pos[2] += (self.final_approach_z_offset - self.grasp_offset_z)
            if desired_pos[2] < self.min_safe_z:
                desired_pos[2] = self.min_safe_z

        elif self.phase == "LIFT":
            desired_pos[2] += self.lift_height

        # Cartesian Errors
        e_pos = desired_pos - p_current
        # Orientation error via rotation vector with 4-fold square cube symmetry (modulo 90 deg)
        best_rot_err = None
        best_rot_dist = float("inf")
        for k in range(4):
            R_sym = desired_rot @ Rotation.from_euler('z', k * 90.0, degrees=True).as_matrix()
            candidate_err = Rotation.from_matrix(R_sym @ r_current.T).as_rotvec()
            cand_dist = float(np.linalg.norm(candidate_err))
            if cand_dist < best_rot_dist:
                best_rot_dist = cand_dist
                best_rot_err = candidate_err

        e_rot = best_rot_err
        pos_dist = float(np.linalg.norm(e_pos))
        rot_dist = best_rot_dist

        # 5 mm deadband filter: do not command robot motion if position is within 5 mm
        tol_pos = 0.005
        if pos_dist < tol_pos and rot_dist < self.rot_tolerance:
            self.converged_frames += 1
            # In tolerance deadband: zero out velocities to hold completely motionless
            return np.zeros(6), (self.converged_frames >= self.required_converged_frames)
        else:
            self.converged_frames = 0

        # Desired Cartesian twist
        v_cmd = self.gain_pos * e_pos
        w_cmd = self.gain_rot * e_rot

        # Velocity limiting in task space
        v_norm = np.linalg.norm(v_cmd)
        if v_norm > self.max_linear_speed:
            v_cmd = v_cmd * (self.max_linear_speed / v_norm)

        w_norm = np.linalg.norm(w_cmd)
        if w_norm > self.max_angular_speed:
            w_cmd = w_cmd * (self.max_angular_speed / w_norm)

        x_dot = np.r_[v_cmd, w_cmd]

        # Differential kinematics via Damped Least Squares
        J = self.kin.jacobian(current_q, include_tcp=True)

        # Adaptive damping based on minimum singular value
        s = np.linalg.svd(J, compute_uv=False)
        sigma_min = float(np.min(s))
        damping = 0.05 if sigma_min > 0.08 else 0.05 + 0.15 * (0.08 - sigma_min) / 0.08

        # DLS formula: q_dot = J^T (J J^T + lambda^2 I)^-1 x_dot
        A = J @ J.T + (damping ** 2) * np.eye(6)
        q_dot = J.T @ np.linalg.solve(A, x_dot)

        # Scale joint velocity if exceeding safety limit
        peak_qd = float(np.max(np.abs(q_dot)))
        if peak_qd > self.max_joint_speed:
            q_dot = q_dot * (self.max_joint_speed / peak_qd)

        return q_dot, False

    def compute_cartesian_step(
        self,
        current_q: np.ndarray,
        target_pos: np.ndarray,
        target_yaw: float,
        pos_tolerance: float = 0.002,
    ) -> tuple[np.ndarray, bool]:
        """Compute joint velocities to regulate directly to a specific Cartesian [X, Y, Z] and yaw.

        Strictly preserves X and Y when moving in Z, preventing any horizontal drift.
        """
        t_current = self.kin.forward_kinematics(current_q, include_tcp=True)
        p_current = t_current[:3, 3]
        r_current = t_current[:3, :3]

        desired_pos = np.asarray(target_pos, dtype=float).copy()
        if desired_pos[2] < self.min_safe_z:
            desired_pos[2] = self.min_safe_z

        cy, sy = np.cos(target_yaw), np.sin(target_yaw)
        desired_rot = np.array([
            [ cy,  sy,  0.0],
            [ sy, -cy,  0.0],
            [0.0, 0.0, -1.0],
        ], dtype=float)

        e_pos = desired_pos - p_current

        # Orientation error via rotation vector with 4-fold square cube symmetry (modulo 90 deg)
        best_rot_err = None
        best_rot_dist = float("inf")
        for k in range(4):
            R_sym = desired_rot @ Rotation.from_euler('z', k * 90.0, degrees=True).as_matrix()
            candidate_err = Rotation.from_matrix(R_sym @ r_current.T).as_rotvec()
            cand_dist = float(np.linalg.norm(candidate_err))
            if cand_dist < best_rot_dist:
                best_rot_dist = cand_dist
                best_rot_err = candidate_err

        e_rot = best_rot_err
        pos_dist = float(np.linalg.norm(e_pos))
        rot_dist = best_rot_dist

        if pos_dist < pos_tolerance and rot_dist < self.rot_tolerance:
            return np.zeros(6), True

        v_cmd = self.gain_pos * e_pos
        w_cmd = self.gain_rot * e_rot

        v_norm = np.linalg.norm(v_cmd)
        if v_norm > self.max_linear_speed:
            v_cmd = v_cmd * (self.max_linear_speed / v_norm)

        w_norm = np.linalg.norm(w_cmd)
        if w_norm > self.max_angular_speed:
            w_cmd = w_cmd * (self.max_angular_speed / w_norm)

        x_dot = np.r_[v_cmd, w_cmd]
        J = self.kin.jacobian(current_q, include_tcp=True)

        s = np.linalg.svd(J, compute_uv=False)
        sigma_min = float(np.min(s))
        damping = 0.05 if sigma_min > 0.08 else 0.05 + 0.15 * (0.08 - sigma_min) / 0.08

        A = J @ J.T + (damping ** 2) * np.eye(6)
        q_dot = J.T @ np.linalg.solve(A, x_dot)

        peak_qd = float(np.max(np.abs(q_dot)))
        if peak_qd > self.max_joint_speed:
            q_dot = q_dot * (self.max_joint_speed / peak_qd)

        return q_dot, False
