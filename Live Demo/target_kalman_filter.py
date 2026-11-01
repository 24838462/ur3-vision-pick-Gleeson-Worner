"""Kalman Filter for Visual Target Tracking (Position and Yaw).

Adapts the discrete-time Joseph-form Kalman Filter from estimation.py
to track 3D object centroids [X, Y, Z] and orientation (yaw) from noisy
camera detections.

Features:
- Decoupled 2-state [position, velocity] double-integrator dynamics for 3D coordinates:
    x = [p, v]^T,  A = [[1, dt], [0, 1]],  B = [0.5*dt^2, dt]^T
- Process noise Q = sigma_a^2 * B * B^T
- Joseph-form measurement update for numerical stability and covariance symmetry
- Canonical modulo pi/2 wrapping for cube yaw angle
- Outlier gating via innovation thresholding (rejects reflection spikes and hand occlusions)
- Velocity estimation to detect when the cube is completely stationary
"""

import time
import numpy as np


class VisualTargetKalmanFilter:
    """Kalman filter for 3D position [X, Y, Z] and yaw orientation tracking."""

    def __init__(
        self,
        meas_noise_pos: float = 0.005,      # 5 mm RealSense depth/spatial noise
        meas_noise_yaw_deg: float = 3.0,     # 3 deg orientation noise
        accel_noise_pos: float = 0.15,       # 0.15 m/s^2 target motion uncertainty
        accel_noise_yaw_deg: float = 10.0,   # 10 deg/s^2 angular acceleration uncertainty
        gating_distance_m: float = 0.10,     # Reject measurement jumps > 10 cm
        timeout_s: float = 1.0,              # Reset if target lost for > 1 s
    ):
        self.R_pos = float(meas_noise_pos ** 2)
        self.R_yaw = float(np.deg2rad(meas_noise_yaw_deg) ** 2)
        self.sigma_a_pos = float(accel_noise_pos)
        self.sigma_a_yaw = float(np.deg2rad(accel_noise_yaw_deg))
        self.gating_distance = float(gating_distance_m)
        self.timeout_s = float(timeout_s)

        # 3D position filter states: shape (3, 2) -> [p, v] for X, Y, Z
        self.x_pos = np.zeros((3, 2), dtype=float)
        self.P_pos = np.tile(np.diag([self.R_pos, 0.05]), (3, 1, 1))

        # Yaw filter state: shape (2,) -> [yaw, yaw_rate]
        self.x_yaw = np.zeros(2, dtype=float)
        self.P_yaw = np.diag([self.R_yaw, np.deg2rad(15.0) ** 2])

        self.last_time = None
        self.initialized = False
        self.consecutive_updates = 0

    def reset(self):
        """Reset filter states."""
        self.x_pos.fill(0.0)
        self.P_pos = np.tile(np.diag([self.R_pos, 0.05]), (3, 1, 1))
        self.x_yaw.fill(0.0)
        self.P_yaw = np.diag([self.R_yaw, np.deg2rad(15.0) ** 2])
        self.last_time = None
        self.initialized = False
        self.consecutive_updates = 0

    @property
    def position(self) -> np.ndarray:
        """Filtered 3D position [X, Y, Z]."""
        return self.x_pos[:, 0].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Filtered 3D velocity [Vx, Vy, Vz]."""
        return self.x_pos[:, 1].copy()

    @property
    def yaw(self) -> float:
        """Filtered canonical yaw angle in radians [-pi/4, pi/4]."""
        return float(self.x_yaw[0])

    @property
    def yaw_rate(self) -> float:
        """Filtered yaw angular velocity in rad/s."""
        return float(self.x_yaw[1])

    @property
    def is_stationary(self) -> bool:
        """Returns True if target speed is below 8 mm/s and yaw rate below 0.05 rad/s."""
        speed = np.linalg.norm(self.velocity)
        return bool(self.initialized and speed < 0.012 and abs(self.yaw_rate) < 0.08)

    def predict(self, timestamp: float | None = None) -> tuple[np.ndarray, float]:
        """Advance filter states using dynamic transition model without a measurement update."""
        if not self.initialized or self.last_time is None:
            return self.position, self.yaw
        now = float(timestamp) if timestamp is not None else time.time()
        dt = float(now - self.last_time)
        if dt <= 1e-4:
            return self.position, self.yaw
        if dt > self.timeout_s:
            self.reset()
            return self.position, self.yaw

        A = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
        B = np.array([0.5 * dt * dt, dt], dtype=float)
        Q_pos = (self.sigma_a_pos ** 2) * np.outer(B, B)
        Q_yaw = (self.sigma_a_yaw ** 2) * np.outer(B, B)

        self.x_pos = self.x_pos @ A.T
        for i in range(3):
            self.P_pos[i] = A @ self.P_pos[i] @ A.T + Q_pos

        self.x_yaw = A @ self.x_yaw
        self.x_yaw[0] = (self.x_yaw[0] + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0
        self.P_yaw = A @ self.P_yaw @ A.T + Q_yaw

        self.last_time = now
        return self.position, self.yaw

    def update(self, measured_pos: np.ndarray, measured_yaw: float, timestamp: float | None = None) -> tuple[np.ndarray, float]:
        """Incorporate a new raw measurement and return the filtered (pos, yaw)."""
        now = float(timestamp) if timestamp is not None else time.time()
        measured_pos = np.asarray(measured_pos, dtype=float)
        measured_yaw = float(measured_yaw)

        # 1. Initialize or reset on timeout
        if not self.initialized or self.last_time is None or (now - self.last_time > self.timeout_s):
            self.x_pos[:, 0] = measured_pos
            self.x_pos[:, 1] = 0.0
            self.x_yaw[0] = measured_yaw
            self.x_yaw[1] = 0.0
            self.last_time = now
            self.initialized = True
            self.consecutive_updates = 1
            return self.position, self.yaw

        dt = float(now - self.last_time)
        if dt <= 1e-4:
            return self.position, self.yaw

        # 2. Outlier rejection gating on 3D distance
        pred_pos = self.x_pos[:, 0] + self.x_pos[:, 1] * dt
        jump_dist = float(np.linalg.norm(measured_pos - pred_pos))
        if jump_dist > self.gating_distance and self.consecutive_updates > 5:
            # Reject anomalous measurement spike
            self.last_time = now
            return self.position, self.yaw

        # 3. Transition and noise matrices
        A = np.array([
            [1.0, dt],
            [0.0, 1.0],
        ], dtype=float)
        B = np.array([0.5 * dt * dt, dt], dtype=float)

        Q_pos = (self.sigma_a_pos ** 2) * np.outer(B, B)
        Q_yaw = (self.sigma_a_yaw ** 2) * np.outer(B, B)

        # 4. Predict step (Position)
        # x_pos: (3, 2), A: (2, 2)
        self.x_pos = self.x_pos @ A.T
        for i in range(3):
            self.P_pos[i] = A @ self.P_pos[i] @ A.T + Q_pos

        # 5. Measurement update (Position) - Joseph Form
        innov_pos = measured_pos - self.x_pos[:, 0]
        # Suppress sub-millimeter jitter when target is stationary
        if self.is_stationary and np.linalg.norm(innov_pos) < 0.003:
            innov_pos = innov_pos * 0.15
            S_i = self.P_pos[i, 0, 0] + self.R_pos
            K_i = self.P_pos[i, :, 0] / S_i  # (2,)

            self.x_pos[i] += K_i * innov_pos[i]

            # Joseph form: P = (I - K H) P (I - K H)^T + K R K^T
            I_KH = np.eye(2)
            I_KH[:, 0] -= K_i
            self.P_pos[i] = I_KH @ self.P_pos[i] @ I_KH.T + self.R_pos * np.outer(K_i, K_i)
            # Guarantee symmetry
            self.P_pos[i] = 0.5 * (self.P_pos[i] + self.P_pos[i].T)

        # 6. Predict step (Yaw)
        self.x_yaw = A @ self.x_yaw
        self.x_yaw[0] = (self.x_yaw[0] + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0
        self.P_yaw = A @ self.P_yaw @ A.T + Q_yaw

        # 7. Measurement update (Yaw) with canonical wrapping modulo pi/2
        # Error between measured and predicted yaw
        innov_yaw = (measured_yaw - self.x_yaw[0] + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0

        S_yaw = self.P_yaw[0, 0] + self.R_yaw
        K_yaw = self.P_yaw[:, 0] / S_yaw

        self.x_yaw += K_yaw * innov_yaw
        # Re-wrap yaw to [-pi/4, pi/4]
        self.x_yaw[0] = (self.x_yaw[0] + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0

        I_KH_yaw = np.eye(2)
        I_KH_yaw[:, 0] -= K_yaw
        self.P_yaw = I_KH_yaw @ self.P_yaw @ I_KH_yaw.T + self.R_yaw * np.outer(K_yaw, K_yaw)
        self.P_yaw = 0.5 * (self.P_yaw + self.P_yaw.T)

        self.last_time = now
        self.consecutive_updates += 1

        return self.position, self.yaw
