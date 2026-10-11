from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
from roboticstoolbox import jtraj


# ---------- SETTINGS YOU CAN CHANGE ----------

# Shoulder pan, shoulder lift, elbow, wrist 1, wrist 2, wrist 3.
HOME_DEG = np.array([0, -90, 90, -90, -90, 0], dtype=float)

MOVE_SECONDS = 5.0
START_HOLD_SECONDS = 2.0

KP = np.array([80, 100, 80, 3, 3, 2], dtype=float)
KD = np.array([20, 25, 20, 0.15, 0.15, 0.08], dtype=float)

model_path = (
    Path(__file__).resolve().parent
    / "models"
    / "ur3_collision_fixed.xml"
)

model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

# This tester assumes the current six-joint, six-motor arm.
if model.nq != 6 or model.nv != 6 or model.nu != 6:
    raise ValueError("This tester requires six hinge joints and six motors.")

home_q = np.deg2rad(HOME_DEG)

TARGET_DEG = np.array([0, 0, 0, 90, 0, -90], dtype=float)
target_q = np.deg2rad(TARGET_DEG)

# Check that the base target is inside its configured limits.
if model.jnt_limited[0]:
    lower, upper = model.jnt_range[0]
    if not lower <= target_q[0] <= upper:
        raise ValueError(
            "The requested rotation exceeds the base joint limits. "
            "Change HOME_DEG[0] or ROTATION_DEG."
        )

# Set the starting pose once.
data.qpos[:] = home_q
mujoco.mj_forward(model, data)


# ---------- GENERATE THE TRAJECTORY ----------

dt = model.opt.timestep
number_of_steps = int(np.ceil(MOVE_SECONDS / dt))
trajectory_times = np.arange(number_of_steps + 1) * dt

trajectory = jtraj(home_q, target_q, trajectory_times)

print("Starting angles:", HOME_DEG)
print("Target angles:", np.rad2deg(target_q))
print("Close the viewer window to stop.")


# ---------- RUN THE PHYSICS AND CONTROLLER ----------

with mujoco.viewer.launch_passive(model, data) as viewer:

    with viewer.lock():
        viewer.opt.geomgroup[3] = 0
        viewer.opt.flags[
            mujoco.mjtVisFlag.mjVIS_CONVEXHULL
        ] = True

        viewer.opt.flags[
            mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
        ] = True

    viewer.cam.distance = 1.5
    viewer.cam.lookat[:] = [0, 0, 0.3]

    wall_start = time.perf_counter()
    next_display_time = 0.0
    reported_finished = False

    while viewer.is_running():
        movement_time = data.time - START_HOLD_SECONDS

        if movement_time < 0:
            desired_q = home_q
            desired_qd = np.zeros(6)

        else:
            index = min(
                int(movement_time / dt),
                len(trajectory.q) - 1,
            )

            desired_q = trajectory.q[index]
            desired_qd = trajectory.qd[index]

        # For this first movement test, use simulator state directly.
        # Later we can replace these with the observer estimates.
        position_error = desired_q - data.qpos
        velocity_error = desired_qd - data.qvel

        # Bias compensation plus position/velocity feedback.
        torque = (
            data.qfrc_bias
            + KP * position_error
            + KD * velocity_error
        )

        # Respect the motor command limits from the XML.
        data.ctrl[:] = np.clip(
            torque,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1],
        )

        mujoco.mj_step(model, data)

        # Refresh the viewer approximately 60 times per second.
        if data.time >= next_display_time:
            viewer.sync()
            next_display_time = data.time + 1.0 / 60.0

        if (
            movement_time >= trajectory_times[-1] + 3.0
            and not reported_finished
        ):
            print("Trajectory finished; holding final pose.")

            print(
                "Actual joint angles (deg):",
                np.round(np.rad2deg(data.qpos), 2),
            )
            print(
                "Target joint angles (deg):",
                np.round(np.rad2deg(target_q), 2),
            )

            print("Joint velocities (rad/s):", np.round(data.qvel, 3))
            print("Requested torques (Nm):", np.round(torque, 2))
            print("Clipped commands (Nm):", np.round(data.ctrl, 2))
            print("Number of contacts:", data.ncon)

            for i in range(data.ncon):
                contact = data.contact[i]

                name1 = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
                )
                name2 = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
                )

                print(
                    f"Contact: {name1 or contact.geom1} "
                    f"<-> {name2 or contact.geom2}, "
                    f"distance={contact.dist:.6f} m"
                )

            reported_finished = True

        # Pace the simulation approximately in real time.
        remaining = (
            wall_start + data.time - time.perf_counter()
        )
        if remaining > 0:
            time.sleep(remaining)