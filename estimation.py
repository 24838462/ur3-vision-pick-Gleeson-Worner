"""Encoder-only joint observer and evaluation logging."""

import csv
import numpy as np


# Initial simulated noise assumptions; tune using logged results.
ENCODER_NOISE_DEG = 0.05
ACCELERATION_STD = 0.5  # rad/s²


class JointStateObserver:
    """Position/velocity Kalman filter for each arm joint."""

    def __init__(
        self,
        joints=6,
        encoder_noise_deg=ENCODER_NOISE_DEG,
        acceleration_std=ACCELERATION_STD,
    ):
        self.joints = joints
        self.encoder_std = np.deg2rad(encoder_noise_deg)
        self.R = max(self.encoder_std**2, 1e-12)
        self.acceleration_std = acceleration_std

        # Each joint has states [position, velocity].
        self.x = np.zeros((joints, 2))
        self.P = np.tile(
            np.diag([self.R, 1.0]),
            (joints, 1, 1),
        )

        self.last_time = None
        self.nis = np.full(joints, np.nan)

    @property
    def q(self):
        return self.x[:, 0].copy()

    @property
    def qd(self):
        return self.x[:, 1].copy()

    def update(self, measured_q, now):
        measured_q = np.asarray(measured_q, dtype=float)
        now = float(now)

        if (
            measured_q.shape != (self.joints,)
            or not np.all(np.isfinite(measured_q))
        ):
            raise ValueError(
                "Observer requires finite joint encoder readings."
            )

        if not np.isfinite(now):
            raise ValueError("Invalid observer timestamp.")

        if self.last_time is None:
            self.x[:, 0] = measured_q
            self.last_time = now
            return

        dt = now - self.last_time

        if dt <= 0:
            return

        # Constant-velocity prediction.
        A = np.array([
            [1.0, dt],
            [0.0, 1.0],
        ])

        # Allow velocity to change through process noise.
        G = np.array([0.5 * dt**2, dt])
        Q = self.acceleration_std**2 * np.outer(G, G)

        self.x = self.x @ A.T
        self.P = A @ self.P @ A.T + Q

        # Correct using position encoders only: C = [1, 0].
        innovation = measured_q - self.x[:, 0]
        S = self.P[:, 0, 0] + self.R
        K = self.P[:, :, 0] / S[:, None]

        self.x += K * innovation[:, None]

        # Joseph covariance update.
        correction = np.tile(
            np.eye(2),
            (self.joints, 1, 1),
        )
        correction[:, :, 0] -= K

        self.P = (
            correction
            @ self.P
            @ correction.transpose(0, 2, 1)
            + self.R * K[:, :, None] * K[:, None, :]
        )
        self.P = 0.5 * (
            self.P + self.P.transpose(0, 2, 1)
        )

        self.nis = innovation**2 / S
        self.last_time = now


class ObserverMonitor:
    """Run the observer and compare it with evaluation ground truth."""

    def __init__(self, path, joint_names, seed=None):
        self.path = path
        self.joint_names = list(joint_names)
        self.filter = JointStateObserver(
            joints=len(self.joint_names)
        )

        # Independent RNG: does not change cube spawn locations.
        self.rng = np.random.default_rng(
            np.random.SeedSequence(
                seed,
                spawn_key=(2026, 1),
            )
        )

        self.samples = 0
        self.error_sum = np.zeros(3)
        self.next_report = 0.0

    def __enter__(self):
        self.file = self.path.open("w", newline="")
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "time_s",
            "joint",
            "encoder_noisy_rad",
            "q_est_rad",
            "qd_est_rad_s",
            "q_true_rad",
            "qd_true_rad_s",
            "q_error_rad",
            "qd_error_rad_s",
            "q_std_rad",
            "qd_std_rad_s",
            "nis",
        ])

        print("Observer evaluation log:", self.path)
        return self

    def sample(self, now, encoder_q, true_q, true_qd):
        if (
            self.filter.last_time is not None
            and now <= self.filter.last_time
        ):
            return

        measured = np.asarray(encoder_q).copy()
        measured += self.rng.normal(
            0.0,
            self.filter.encoder_std,
            len(self.joint_names),
        )

        # Ground truth is not supplied to the filter.
        self.filter.update(measured, now)

        q = self.filter.q
        qd = self.filter.qd

        # Ground truth is used only below for evaluation.
        q_error = q - true_q
        qd_error = qd - true_qd

        uncertainty = np.sqrt(np.maximum(
            np.diagonal(
                self.filter.P,
                axis1=1,
                axis2=2,
            ),
            0.0,
        ))

        for i, name in enumerate(self.joint_names):
            self.writer.writerow([
                now,
                name,
                measured[i],
                q[i],
                qd[i],
                true_q[i],
                true_qd[i],
                q_error[i],
                qd_error[i],
                uncertainty[i, 0],
                uncertainty[i, 1],
                self.filter.nis[i],
            ])

        self.error_sum += [
            np.sum((measured - true_q)**2),
            np.sum(q_error**2),
            np.sum(qd_error**2),
        ]
        self.samples += len(self.joint_names)

        if now >= self.next_report:
            self.next_report = now + 2.0
            self.report()
            self.file.flush()

    def report(self):
        if not self.samples:
            return

        raw, position, velocity = np.sqrt(
            self.error_sum / self.samples
        )

        print(
            "Observer cumulative RMSE: "
            f"noisy encoder={np.rad2deg(raw):.3f} deg, "
            f"position={np.rad2deg(position):.3f} deg, "
            f"velocity={velocity:.4f} rad/s"
        )

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.report()
        finally:
            self.file.close()