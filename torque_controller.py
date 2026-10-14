"""Map nominal LQR joint accelerations to arm motor commands.

First-stage development controller:
- Uses simulator joint positions and velocities.
- Updates at approximately 100 Hz.
- Assumes six direct, unit-gear torque motors.
- Does not compensate contact forces.
- Assumes zero commanded acceleration for other model DOFs.

Observer feedback will be connected separately.
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

        # Check the transmission and actuator assumptions.
        for motor, q_address, v_address in zip(
            self.motors, self.qa, self.va
        ):
            if (
                model.actuator_trntype[motor]
                != mujoco.mjtTrn.mjTRN_JOINT
            ):
                raise ValueError(
                    "Arm actuators must drive individual joints."
                )

            joint = model.actuator_trnid[motor, 0]

            if (
                model.jnt_qposadr[joint] != q_address
                or model.jnt_dofadr[joint] != v_address
            ):
                raise ValueError(
                    "Motor and joint address ordering disagree."
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

        # Match the control interval to whole physics steps.
        physics_dt = float(model.opt.timestep)

        self.control_steps = max(
            1, int(round(sample_time / physics_dt))
        )
        self.sample_time = self.control_steps * physics_dt

        self.feedback = JointStateFeedback(
            self.sample_time
        )

        # Separate data prevents dynamics calculations from
        # changing the live simulation's state or sensors.
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
            "Feedback currently uses simulator joint state."
        )

    def commands(
        self,
        data,
        desired_q,
        desired_qd,
        desired_qdd=None,
    ):
        """Return motor commands, held between control updates."""
        if data.time + 1e-10 < self.next_update:
            return self.command.copy()

        self.next_update = data.time + self.sample_time

        q = data.qpos[self.qa]
        qd = data.qvel[self.va]

        self.acceleration_command = (
            self.feedback.acceleration(
                desired_q,
                desired_qd,
                q,
                qd,
                desired_qdd,
            )
        )

        # Refresh dynamics at the current state.
        self.dynamics.qpos[:] = data.qpos
        self.dynamics.qvel[:] = data.qvel
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

        # inertial_force = M(q) @ commanded_acceleration
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
                "LQR motor command clipped; "
                "check tracking and torque limits."
            )

        return self.command.copy()