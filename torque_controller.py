"""Computed-torque LQI using estimated arm joint states.

Other model DOFs retain simulator states for this development
stage. Contact forces are not explicitly compensated.
"""

import mujoco
import numpy as np

from joint_controller import JointStateFeedback


class ComputedTorqueController:
    def __init__(
        self,
        model,
        q_addresses,
        v_addresses,
        motor_ids,
        sample_time=0.01,
    ):
        self.model = model
        self.qa = np.asarray(q_addresses, dtype=int)
        self.va = np.asarray(v_addresses, dtype=int)
        self.motors = np.asarray(motor_ids, dtype=int)

        for addresses in (self.qa, self.va, self.motors):
            if addresses.shape != (6,):
                raise ValueError(
                    "Expected six arm joints and motors."
                )

        for motor, q_address, v_address in zip(
            self.motors, self.qa, self.va
        ):
            if (
                model.actuator_trntype[motor]
                != mujoco.mjtTrn.mjTRN_JOINT
            ):
                raise ValueError(
                    "Arm motors must drive individual joints."
                )

            joint = model.actuator_trnid[motor, 0]

            if (
                model.jnt_qposadr[joint] != q_address
                or model.jnt_dofadr[joint] != v_address
            ):
                raise ValueError(
                    "Motor and joint ordering disagree."
                )

            if (
                not np.isclose(
                    model.actuator_gear[motor, 0], 1.0
                )
                or model.actuator_dyntype[motor]
                != mujoco.mjtDyn.mjDYN_NONE
                or model.actuator_gaintype[motor]
                != mujoco.mjtGain.mjGAIN_FIXED
                or model.actuator_biastype[motor]
                != mujoco.mjtBias.mjBIAS_NONE
                or not np.isclose(
                    model.actuator_gainprm[motor, 0], 1.0
                )
            ):
                raise ValueError(
                    "Expected direct unit-gear torque motors."
                )

        physics_dt = float(model.opt.timestep)
        self.control_steps = max(
            1, int(round(sample_time / physics_dt))
        )
        self.sample_time = (
            self.control_steps * physics_dt
        )

        self.feedback = JointStateFeedback(
            self.sample_time
        )

        self.dynamics = mujoco.MjData(model)
        self.full_acceleration = np.zeros(model.nv)
        self.inertial_force = np.zeros(model.nv)

        self.requested_torque = np.zeros(6)
        self.command = np.zeros(6)
        self.acceleration_command = np.zeros(6)

        self.next_update = 0.0
        self.next_warning = 0.0

        self.updates = 0
        self.saturated_updates = 0

        print(
            "Computed-torque LQI enabled; "
            f"control interval={self.sample_time:.4f} s."
        )
        print(
            "Arm feedback and compensation use "
            "observer estimates."
        )

    def commands(
        self,
        data,
        desired_q,
        desired_qd,
        *,
        estimated_q,
        estimated_qd,
        desired_qdd=None,
    ):
        if data.time + 1e-10 < self.next_update:
            return self.command.copy()

        self.next_update = (
            data.time + self.sample_time
        )

        q = np.asarray(estimated_q, dtype=float)
        qd = np.asarray(estimated_qd, dtype=float)

        if (
            q.shape != (6,)
            or qd.shape != (6,)
            or not np.all(np.isfinite(q))
            or not np.all(np.isfinite(qd))
        ):
            raise ValueError(
                "Invalid observer state supplied to control."
            )

        self.acceleration_command = (
            self.feedback.acceleration(
                desired_q,
                desired_qd,
                q,
                qd,
                desired_qdd,
            )
        )

        # Retain other model states, then replace the arm
        # state before calculating inertia and bias forces.
        self.dynamics.qpos[:] = data.qpos
        self.dynamics.qvel[:] = data.qvel

        self.dynamics.qpos[self.qa] = q
        self.dynamics.qvel[self.va] = qd
        self.dynamics.time = data.time

        mujoco.mj_fwdPosition(
            self.model, self.dynamics
        )
        mujoco.mj_fwdVelocity(
            self.model, self.dynamics
        )

        self.full_acceleration[:] = 0.0
        self.full_acceleration[self.va] = (
            self.acceleration_command
        )

        mujoco.mj_mulM(
            self.model,
            self.dynamics,
            self.inertial_force,
            self.full_acceleration,
        )

        self.requested_torque = (
            self.inertial_force[self.va]
            + self.dynamics.qfrc_bias[self.va]
            - self.dynamics.qfrc_passive[self.va]
        )

        if not np.all(np.isfinite(self.requested_torque)):
            raise RuntimeError(
                "Computed-torque command is invalid."
            )

        lower = np.full(6, -np.inf)
        upper = np.full(6, np.inf)

        limited = self.model.actuator_ctrllimited[
            self.motors
        ].astype(bool)
        ranges = self.model.actuator_ctrlrange[
            self.motors
        ]

        lower[limited] = ranges[limited, 0]
        upper[limited] = ranges[limited, 1]

        self.command = np.clip(
            self.requested_torque, lower, upper
        )

        saturated = np.any(
            np.abs(
                self.command - self.requested_torque
            ) > 1e-9
        )

        self.updates += 1
        self.saturated_updates += int(saturated)

        if saturated and data.time >= self.next_warning:
            self.next_warning = data.time + 2.0
            print(
                "LQI motor command clipped; "
                "check tracking and torque limits."
            )

        return self.command.copy()