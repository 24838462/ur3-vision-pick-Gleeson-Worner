from pathlib import Path
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np


model_path = (
    Path(__file__).resolve().parent
    / "models"
    / "ur3_pick_gripper_scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

# Set the six arm joints.
data.qpos[:6] = np.deg2rad([0, -90, 90, -90, -90, 0])

# Initialise all gripper linkage joints at zero (open). 225 closed
data.qpos[6:] = 0.0

mujoco.mj_forward(model, data)

# Camera rendering options: hide collision meshes.
camera_options = mujoco.MjvOption()
camera_options.geomgroup[3] = 0

window_name = "Wrist camera"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 640, 480)

print("Main viewer: external scene.")
print("Second window: wrist camera.")
print("Physics paused. Press Q in the camera window to exit.")

try:
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            with viewer.lock():
                viewer.opt.geomgroup[3] = 0
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.distance = 1.4
                viewer.cam.lookat[:] = [0.2, 0, 0.15]

            while viewer.is_running():
                viewer.sync()

                renderer.update_scene(
                    data,
                    camera="wrist_camera",
                    scene_option=camera_options,
                )
                rgb = renderer.render()

                # OpenCV displays colours in BGR order.
                cv2.imshow(
                    window_name,
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if cv2.getWindowProperty(
                    window_name, cv2.WND_PROP_VISIBLE
                ) < 1:
                    break

                time.sleep(1 / 30)

finally:
    cv2.destroyAllWindows()