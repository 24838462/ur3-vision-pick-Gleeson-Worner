"""UR3e Kinematics and Jacobian calculations for Cartesian PBVS.

Provides:
- Analytic DH parameters for Universal Robots UR3e
- Forward Kinematics (FK) for any joint configuration
- Geometric Jacobian (6x6) in Base frame
- Singularity and manipulability checks
"""

import numpy as np


class UR3eKinematics:
    """Analytical kinematics and Jacobian for UR3e manipulator.

    Standard DH Parameters for UR3e (all lengths in meters, angles in radians):
    Joint 1: d1 = 0.15185, a1 = 0.0,       alpha1 = pi/2
    Joint 2: d2 = 0.0,     a2 = -0.24355,  alpha2 = 0.0
    Joint 3: d3 = 0.0,     a3 = -0.21320,  alpha3 = 0.0
    Joint 4: d4 = 0.13105, a4 = 0.0,       alpha4 = pi/2
    Joint 5: d5 = 0.08535, a5 = 0.0,       alpha5 = -pi/2
    Joint 6: d6 = 0.09210, a6 = 0.0,       alpha6 = 0.0

    TCP (Tool Center Point) default offset along tool Z axis for OnRobot RG2 gripper:
    approx. 0.160 m to 0.195 m from tool0 flange to gripper fingertip center.
    """

    def __init__(self, tcp_offset_z: float = 0.180):
        # DH Parameters [d, a, alpha]
        self.d = np.array([0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.09210], dtype=float)
        self.a = np.array([0.0, -0.24355, -0.21320, 0.0, 0.0, 0.0], dtype=float)
        self.alpha = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0], dtype=float)

        # Tool offset from flange (tool0) to gripper TCP
        self.tcp_offset_z = float(tcp_offset_z)

        # Standard joint limits [-2*pi, 2*pi]
        self.q_min = np.array([-2 * np.pi] * 6, dtype=float)
        self.q_max = np.array([2 * np.pi] * 6, dtype=float)

    @staticmethod
    def _dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
        """Compute standard Denavit-Hartenberg transformation matrix."""
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0.0,      sa,       ca,      d],
            [0.0,     0.0,      0.0,    1.0]
        ], dtype=float)

    def forward_kinematics_all(self, q: np.ndarray) -> list[np.ndarray]:
        """Compute all intermediate transformation matrices from base to each link."""
        q = np.asarray(q, dtype=float)
        transforms = [np.eye(4, dtype=float)]
        t_curr = np.eye(4, dtype=float)

        for i in range(6):
            t_i = self._dh_transform(q[i], self.d[i], self.a[i], self.alpha[i])
            t_curr = t_curr @ t_i
            transforms.append(t_curr.copy())

        # If TCP offset is non-zero, append TCP frame
        if abs(self.tcp_offset_z) > 1e-6:
            t_tcp = np.eye(4, dtype=float)
            t_tcp[2, 3] = self.tcp_offset_z
            transforms.append(t_curr @ t_tcp)
        else:
            transforms.append(t_curr.copy())

        return transforms

    def forward_kinematics(self, q: np.ndarray, include_tcp: bool = True) -> np.ndarray:
        """Compute end-effector / TCP 4x4 pose in Base frame."""
        transforms = self.forward_kinematics_all(q)
        return transforms[-1] if include_tcp else transforms[6]

    def jacobian(self, q: np.ndarray, include_tcp: bool = True) -> np.ndarray:
        """Compute 6x6 geometric Jacobian in Base frame.

        J = [ J_v ]  (linear velocity: 3x6)
            [ J_w ]  (angular velocity: 3x6)
        such that v_base = J * q_dot
        """
        transforms = self.forward_kinematics_all(q)
        p_ee = transforms[-1][:3, 3] if include_tcp else transforms[6][:3, 3]

        J = np.zeros((6, 6), dtype=float)

        for i in range(6):
            # Rotation axis z_{i-1} in base frame
            z_prev = transforms[i][:3, 2]
            # Origin p_{i-1} in base frame
            p_prev = transforms[i][:3, 3]

            # Linear velocity contribution: z_{i-1} x (p_ee - p_{i-1})
            J[:3, i] = np.cross(z_prev, p_ee - p_prev)
            # Angular velocity contribution: z_{i-1}
            J[3:, i] = z_prev

        return J

    def manipulability(self, q: np.ndarray) -> tuple[float, float]:
        """Compute Yoshikawa manipulability index and minimum singular value."""
        J = self.jacobian(q)
        singular_values = np.linalg.svd(J, compute_uv=False)
        sigma_min = float(np.min(singular_values))
        yoshikawa = float(np.prod(singular_values))
        return sigma_min, yoshikawa
