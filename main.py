from pathlib import Path

import mujoco

import numpy as np

home_deg = np.array([
    0, # Base
    -90, # Shoulder Lift
    90, # Elbow
    -90, # Wrist 1
    -90, # Wrist 2
    0, # Wrist 3
])

home_pos = np.deg2rad(home_deg)

model_path = (
    Path(__file__).resolve().parent
    / "models"
    / "ur3_controlled.xml"
)

model = mujoco.MjModel.from_xml_path(str(model_path))

data = mujoco.MjData(model)

data.qpos[:] = home_pos

mujoco.mj_forward(model, data)

encoder_angles = data.sensordata.copy()

print("Motors:", model.nu)
print("Sensors:", model.nsensor)
print("Encoder angles (rad):", np.round(encoder_angles, 4))
print("Encoder angles (deg):", np.round(
    np.rad2deg(encoder_angles), 1
))