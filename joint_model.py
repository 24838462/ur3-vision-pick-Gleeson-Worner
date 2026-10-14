"""Nominal six-joint state-space model.

State:
    x = [q1, ..., q6, qd1, ..., qd6]

Input:
    u = six joint accelerations, in rad/s^2

Measurement:
    y = six encoder angles, in radians

Assumptions:
    Independent joints with acceleration inputs.
    Exact zero-order-hold discretisation.

This is a starting model, not the complete torque-driven
MuJoCo robot dynamics. It does not modify the simulation.
"""

import argparse

import numpy as np


def build_joint_model(sample_time, joints=6):
    """Return continuous and discrete state-space matrices."""
    if not np.isfinite(sample_time) or sample_time <= 0:
        raise ValueError("Sample time must be finite and positive.")

    identity = np.eye(joints)
    zero = np.zeros((joints, joints))

    # Continuous model:
    # q_dot = qd
    # qd_dot = u
    A = np.block([
        [zero, identity],
        [zero, zero],
    ])

    B = np.vstack([
        zero,
        identity,
    ])

    C = np.hstack([
        identity,
        zero,
    ])

    D = zero.copy()

    # Exact discrete model for acceleration held constant
    # throughout each sample interval.
    #
    # q_next  = q + Ts * qd + 0.5 * Ts^2 * u
    # qd_next = qd + Ts * u
    Ts = sample_time

    Ad = np.block([
        [identity, Ts * identity],
        [zero, identity],
    ])

    Bd = np.vstack([
        0.5 * Ts**2 * identity,
        Ts * identity,
    ])

    return A, B, C, D, Ad, Bd


def controllability_matrix(A, B):
    """Build [B, AB, A^2 B, ...]."""
    blocks = []
    current = B.copy()

    for _ in range(A.shape[0]):
        blocks.append(current)
        current = A @ current

    return np.hstack(blocks)


def observability_matrix(A, C):
    """Build [C; CA; CA^2; ...]."""
    blocks = []
    current = C.copy()

    for _ in range(A.shape[0]):
        blocks.append(current)
        current = current @ A

    return np.vstack(blocks)


def analyse_model(sample_time):
    A, B, C, D, Ad, Bd = build_joint_model(sample_time)

    states = Ad.shape[0]

    control_rank = np.linalg.matrix_rank(
        controllability_matrix(Ad, Bd)
    )

    observation_rank = np.linalg.matrix_rank(
        observability_matrix(Ad, C)
    )

    print("Nominal six-joint acceleration-input model")
    print(f"Sample time: {sample_time:.6f} seconds")
    print(f"States: {states} (6 angles + 6 velocities)")
    print(f"Inputs: {B.shape[1]} joint accelerations")
    print(f"Measurements: {C.shape[0]} encoder angles")
    print()

    print(f"Controllability rank: {control_rank}/{states}")
    print(f"Observability rank: {observation_rank}/{states}")
    print()

    if control_rank == states:
        print("PASS: the nominal model is controllable.")
    else:
        print("CHECK: controllability rank is incomplete.")

    if observation_rank == states:
        print("PASS: the nominal model is observable.")
    else:
        print("CHECK: observability rank is incomplete.")

    print()
    print("Single-joint discrete matrices:")
    print("Ad:")
    print(np.array([
        [1.0, sample_time],
        [0.0, 1.0],
    ]))

    print("Bd:")
    print(np.array([
        [0.5 * sample_time**2],
        [sample_time],
    ]))

    print("C:")
    print(np.array([[1.0, 0.0]]))

    print()
    print("These checks apply to the nominal model.")
    print("Motor torque dynamics and closed-loop stability")
    print("still need to be modelled and evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse the nominal arm joint model."
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Proposed controller sample time in seconds.",
    )

    args = parser.parse_args()
    analyse_model(args.dt)