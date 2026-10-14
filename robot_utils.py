"""Shared robot model lookup and inverse-kinematics helpers."""

import mujoco
import numpy as np
from scipy.optimize import least_squares


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

MOTOR_NAMES = [
    name.replace("_joint", "_motor")
    for name in JOINT_NAMES
]


def get_id(model, kind, name):
    index = mujoco.mj_name2id(model, kind, name)

    if index < 0:
        raise ValueError(
            f"Missing model element: {name}"
        )

    return index


def joint_limits(model, q_addresses):
    lower = np.full(6, -2 * np.pi)
    upper = np.full(6, 2 * np.pi)

    for index, address in enumerate(q_addresses):
        joint_id = int(
            np.flatnonzero(
                model.jnt_qposadr == address
            )[0]
        )

        if model.jnt_limited[joint_id]:
            lower[index], upper[index] = (
                model.jnt_range[joint_id]
            )

    return lower + 0.001, upper - 0.001


def solve_pose(
    residual,
    current,
    lower,
    upper,
    max_nfev,
):
    """Try the existing IK seeds and acceptance thresholds."""
    seeds = [current]

    for elbow in (90, -90):
        alternative = current.copy()
        alternative[1:3] = np.deg2rad(
            [-60, elbow]
        )
        seeds.append(
            np.clip(alternative, lower, upper)
        )

    for seed in seeds:
        result = least_squares(
            residual,
            seed,
            bounds=(lower, upper),
            max_nfev=max_nfev,
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

    return None