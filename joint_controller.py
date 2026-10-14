"""LQI feedback for the nominal acceleration-input joint model.

States:
    Six position errors, six velocity errors,
    and six integrated position errors.

Output:
    Joint acceleration in rad/s^2, not motor torque.
"""

import argparse

import numpy as np
from scipy.linalg import solve_discrete_are

from joint_model import build_joint_model


POSITION_ERROR_SCALE = np.deg2rad(5.0)
VELOCITY_ERROR_SCALE = 0.5
ACCELERATION_SCALE = 2.0

# Integrated error has units rad*s.
INTEGRAL_ERROR_SCALE = np.deg2rad(5.0) * 1.0
INTEGRAL_LIMIT = 0.15  # rad*s


class JointStateFeedback:
    def __init__(self, sample_time=0.01):
        self.sample_time = sample_time

        _, _, C, _, self.Ad, self.Bd = build_joint_model(
            sample_time
        )

        # Integral state:
        # integral_next = integral + Ts * position_error
        self.A_aug = np.block([
            [self.Ad, np.zeros((12, 6))],
            [sample_time * C, np.eye(6)],
        ])

        self.B_aug = np.vstack([
            self.Bd,
            np.zeros((6, 6)),
        ])

        self.Q = np.diag(
            [1.0 / POSITION_ERROR_SCALE**2] * 6
            + [1.0 / VELOCITY_ERROR_SCALE**2] * 6
            + [1.0 / INTEGRAL_ERROR_SCALE**2] * 6
        )

        self.R = np.eye(6) / ACCELERATION_SCALE**2

        self.P = solve_discrete_are(
            self.A_aug,
            self.B_aug,
            self.Q,
            self.R,
        )

        self.K = np.linalg.solve(
            self.R + self.B_aug.T @ self.P @ self.B_aug,
            self.B_aug.T @ self.P @ self.A_aug,
        )

        self.Kq = self.K[:, :6]
        self.Kv = self.K[:, 6:12]
        self.Ki = self.K[:, 12:]

        self.integral_error = np.zeros(6)

        self.closed_loop = (
            self.A_aug - self.B_aug @ self.K
        )
        self.poles = np.linalg.eigvals(
            self.closed_loop
        )

    @staticmethod
    def _vector(value, name):
        vector = np.asarray(value, dtype=float)

        if vector.shape != (6,):
            raise ValueError(
                f"{name} must contain six values."
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f"{name} contains invalid values."
            )

        return vector

    def acceleration(
        self,
        desired_q,
        desired_qd,
        estimated_q,
        estimated_qd,
        desired_qdd=None,
    ):
        desired_q = self._vector(
            desired_q, "desired_q"
        )
        desired_qd = self._vector(
            desired_qd, "desired_qd"
        )
        estimated_q = self._vector(
            estimated_q, "estimated_q"
        )
        estimated_qd = self._vector(
            estimated_qd, "estimated_qd"
        )

        if desired_qdd is None:
            desired_qdd = np.zeros(6)
        else:
            desired_qdd = self._vector(
                desired_qdd, "desired_qdd"
            )

        position_error = desired_q - estimated_q
        velocity_error = desired_qd - estimated_qd

        raw_command = (
            desired_qdd
            + self.Kq @ position_error
            + self.Kv @ velocity_error
            + self.Ki @ self.integral_error
        )

        command = np.clip(
            raw_command,
            -ACCELERATION_SCALE,
            ACCELERATION_SCALE,
        )

        # Update the integral for the next control sample.
        candidate_integral = np.clip(
            self.integral_error
            + self.sample_time * position_error,
            -INTEGRAL_LIMIT,
            INTEGRAL_LIMIT,
        )

        integral_change = (
            candidate_integral - self.integral_error
        )

        # The independent-joint model gives diagonal Ki.
        # Block integration that pushes an already clipped
        # acceleration command further into saturation.
        correction_change = (
            np.diag(self.Ki) * integral_change
        )

        blocked = (
            (raw_command >= ACCELERATION_SCALE)
            & (correction_change > 0)
        ) | (
            (raw_command <= -ACCELERATION_SCALE)
            & (correction_change < 0)
        )

        self.integral_error = np.where(
            blocked,
            self.integral_error,
            candidate_integral,
        )

        return command

    def report(self):
        print("Nominal joint LQI controller")
        print(
            f"Controller sample time: "
            f"{self.sample_time:.6f} seconds"
        )
        print("States: 18, including 6 integral states")
        print()

        for label, gains in (
            ("Position", self.Kq),
            ("Velocity", self.Kv),
            ("Integral", self.Ki),
        ):
            print(f"{label} feedback gains:")
            print(np.round(np.diag(gains), 4))

        print()
        print("Augmented closed-loop poles:")
        print(np.round(self.poles, 6))

        maximum = float(
            np.max(np.abs(self.poles))
        )

        print()
        print(
            f"Largest pole magnitude: {maximum:.6f}"
        )

        if maximum < 1.0:
            print(
                "PASS: the unsaturated nominal "
                "LQI model is stable."
            )
        else:
            print(
                "CHECK: nominal LQI stability failed."
            )

        print()
        print("Output: acceleration, not motor torque.")
        print("Pole analysis assumes a fixed reference")
        print("and excludes clipping and model errors.")
        print("Full robot performance requires testing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Design nominal joint LQI feedback."
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
    )

    args = parser.parse_args()
    JointStateFeedback(args.dt).report()