"""Safety evaluation logging. Does not control the robot."""

import csv
from pathlib import Path

import numpy as np


class SafetyLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.active = False
        self.episode = 0
        self.spawn_time = None
        self.clear_time = None
        self.detection_delay = np.nan
        self.response_delay = np.nan
        self.resume_delay = np.nan
        self.hold_reference = None
        self.next_flush = 0.0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="")
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "time_s",
            "episode",
            "obstacle_active",
            "carrying_commanded",
            "carry_hold",
            "safety_status",
            "confirmed_detection",
            "detection_delay_s",
            "response_delay_s",
            "return_to_normal_delay_s",
            "overhead_position_error_m",
            "pinch_to_intruder_centre_m",
            "hold_drift_deg",
            "max_joint_speed_rad_s",
            "intruder_true_x",
            "intruder_true_y",
            "intruder_true_z",
            "intruder_est_x",
            "intruder_est_y",
            "intruder_est_z",
        ])

        print("Safety evaluation log:", self.path)
        return self

    def obstacle_changed(self, active, now):
        """Record the simulation time of spawning or removal."""
        active = bool(active)

        if active == self.active:
            return

        self.active = active

        if active:
            self.episode += 1
            self.spawn_time = float(now)
            self.clear_time = None
            self.detection_delay = np.nan
            self.response_delay = np.nan
            self.resume_delay = np.nan
        else:
            self.clear_time = float(now)

    @staticmethod
    def point(value):
        if value is None:
            return np.full(3, np.nan)

        point = np.asarray(value, dtype=float)

        if point.shape != (3,) or not np.all(np.isfinite(point)):
            return np.full(3, np.nan)

        return point.copy()

    def sample(
        self,
        now,
        camera,
        obstacle,
        status,
        carrying,
        carry_hold,
        actual_q,
        actual_qd,
        pinch_position,
    ):
        now = float(now)
        self.obstacle_changed(obstacle.active, now)
        status_name = status.value

        confirmed = (
            camera.last_detection is not None
            and camera.filter.tracking
            and camera.filter.missed_frames == 0
            and camera.filter.confirmed_frames
            >= camera.config.persistence_frames
        )

        if self.active and self.spawn_time is not None:
            if confirmed and np.isnan(self.detection_delay):
                self.detection_delay = now - self.spawn_time

            responding = status_name in (
                "tracking",
                "joint_limit",
                "safety_stop",
                "holding_safety_stop",
                "interrupt",
                "retreat",
            )

            if responding and np.isnan(self.response_delay):
                self.response_delay = now - self.spawn_time

        if (
            not self.active
            and self.clear_time is not None
            and status_name == "normal"
            and np.isnan(self.resume_delay)
        ):
            self.resume_delay = now - self.clear_time

        q = np.asarray(actual_q, dtype=float)
        qd = np.asarray(actual_qd, dtype=float)

        if carry_hold:
            if self.hold_reference is None:
                self.hold_reference = q.copy()

            hold_drift = float(np.max(np.abs(
                np.rad2deg(q - self.hold_reference)
            )))
        else:
            self.hold_reference = None
            hold_drift = np.nan

        # Simulator truth is used here only for evaluation.
        truth = self.point(obstacle.position)
        estimate = self.point(
            camera.filter.position
            if camera.filter.tracking
            else None
        )
        pinch = self.point(pinch_position)

        position_error = float(np.linalg.norm(estimate - truth))
        pinch_distance = float(np.linalg.norm(pinch - truth))

        self.writer.writerow([
            now,
            self.episode,
            self.active,
            bool(carrying),
            bool(carry_hold),
            status_name,
            confirmed,
            self.detection_delay,
            self.response_delay,
            self.resume_delay,
            position_error,
            pinch_distance,
            hold_drift,
            float(np.max(np.abs(qd))),
            *truth,
            *estimate,
        ])

        if now >= self.next_flush:
            self.file.flush()
            self.next_flush = now + 1.0

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()