"""Encoder-only joint estimation and evaluation logging."""

import csv

import numpy as np


ENCODER_NOISE_DEG = 0.20

# Uncertainty in acceleration beyond the nominal command.
# Increased for the faster filter and unmodelled contact loads.
ACCELERATION_STD = 2.0  # rad/s^2


class JointStateObserver:
    """Six position/velocity Kalman filters with known input."""

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

    def update(self, measured_q, now, acceleration=None):
        measured_q = np.asarray(measured_q, dtype=float)
        now = float(now)

        if (
            measured_q.shape != (self.joints,)
            or not np.all(np.isfinite(measured_q))
        ):
            raise ValueError(
                "Observer requires finite encoder readings."
            )

        if not np.isfinite(now):
            raise ValueError("Invalid observer timestamp.")

        if acceleration is None:
            acceleration = np.zeros(self.joints)
        else:
            acceleration = np.asarray(
                acceleration, dtype=float
            )

        if (
            acceleration.shape != (self.joints,)
            or not np.all(np.isfinite(acceleration))
        ):
            raise ValueError(
                "Observer requires finite acceleration inputs."
            )

        if self.last_time is None:
            # The simulation starts with the arm at rest.
            self.x[:, 0] = measured_q
            self.x[:, 1] = 0.0
            self.last_time = now
            return

        dt = now - self.last_time

        if dt <= 0:
            return

        A = np.array([
            [1.0, dt],
            [0.0, 1.0],
        ])

        B = np.array([0.5 * dt**2, dt])

        Q = (
            self.acceleration_std**2
            * np.outer(B, B)
        )

        # Predict using the command applied during the
        # interval since the previous encoder sample.
        self.x = (
            self.x @ A.T
            + acceleration[:, None] * B
        )
        self.P = A @ self.P @ A.T + Q

        innovation = measured_q - self.x[:, 0]
        S = self.P[:, 0, 0] + self.R
        K = self.P[:, :, 0] / S[:, None]

        self.x += K * innovation[:, None]

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
    """Supply estimates and log errors against evaluation truth."""

    def __init__(self, path, joint_names, seed=None):
        self.path = path
        self.joint_names = list(joint_names)
        self.filter = JointStateObserver(
            joints=len(self.joint_names)
        )

        self.rng = np.random.default_rng(
            np.random.SeedSequence(
                seed,
                spawn_key=(2026, 1),
            )
        )

        self.samples = 0
        self.error_sum = np.zeros(3)
        self.next_report = 0.0
        self.plotter = None

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
            "commanded_acceleration_rad_s2",
        ])

        print("Observer evaluation log:", self.path)
        print(
            "Observer assumptions: "
            f"encoder noise={ENCODER_NOISE_DEG:.3f} deg, "
            f"acceleration uncertainty="
            f"{ACCELERATION_STD:.2f} rad/s^2."
        )
        return self

    def sample(
        self,
        now,
        encoder_q,
        true_q,
        true_qd,
        acceleration=None,
    ):
        if (
            self.filter.last_time is not None
            and now <= self.filter.last_time
        ):
            return self.filter.q, self.filter.qd

        joints = len(self.joint_names)

        if acceleration is None:
            acceleration = np.zeros(joints)
        else:
            acceleration = np.asarray(
                acceleration, dtype=float
            ).copy()

        measured = np.asarray(
            encoder_q, dtype=float
        ).copy()

        measured += self.rng.normal(
            0.0,
            self.filter.encoder_std,
            joints,
        )

        # No ground truth enters the filter.
        self.filter.update(
            measured, now, acceleration
        )

        q = self.filter.q
        qd = self.filter.qd

        # Ground truth is used only for evaluation below.
        true_q = np.asarray(true_q)
        true_qd = np.asarray(true_qd)

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
                acceleration[i],
            ])

        self.error_sum += [
            np.sum((measured - true_q)**2),
            np.sum(q_error**2),
            np.sum(qd_error**2),
        ]
        self.samples += joints

        if self.plotter is not None:
            self.plotter.add_sample(
                now,
                measured,
                true_q,
                true_qd,
                q,
                qd,
                uncertainty[:, 0],
                uncertainty[:, 1],
                self.filter.nis,
            )

        if now >= self.next_report:
            self.next_report = now + 2.0
            self.report()
            self.file.flush()

        return q, qd

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