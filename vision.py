import cv2
import numpy as np
import mujoco


class CubeDetector:
    def __init__(self, width, height, fovy_deg):
        # Intrinsics for the current ideal simulated pinhole camera.
        # These come from its configured field of view.
        focal = height / (
            2.0 * np.tan(np.deg2rad(fovy_deg) / 2.0)
        )

        self.fx = focal
        self.fy = focal
        self.cx = (width - 1) / 2.0
        self.cy = (height - 1) / 2.0

        self.ranges = {
            "red": [
                ((0, 100, 35), (10, 255, 255)),
                ((170, 100, 35), (179, 255, 255)),
            ],
            "green": [
                ((35, 100, 35), (85, 255, 255)),
            ],
            "blue": [
                ((90, 100, 35), (135, 255, 255)),
            ],
        }

    def detect(self, rgb, depth):
        """Return image detections and visible surface measurements."""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        annotated = rgb.copy()
        detections = {}

        kernel = np.ones((3, 3), dtype=np.uint8)

        for name, ranges in self.ranges.items():
            mask = np.zeros(depth.shape, dtype=np.uint8)

            for lower, upper in ranges:
                colour_mask = cv2.inRange(
                    hsv,
                    np.array(lower, dtype=np.uint8),
                    np.array(upper, dtype=np.uint8),
                )
                mask = cv2.bitwise_or(mask, colour_mask)

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                kernel,
            )

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)

            if cv2.contourArea(contour) < 40:
                continue

            # Restrict depth samples to this colour component.
            component = np.zeros_like(mask)
            cv2.drawContours(
                component,
                [contour],
                -1,
                255,
                thickness=cv2.FILLED,
            )
            component = cv2.bitwise_and(component, mask)

            # Remove edge pixels that can mix object/background depth.
            component = cv2.erode(component, kernel)

            valid = (
                (component > 0)
                & np.isfinite(depth)
                & (depth > 0.02)
                & (depth < 2.0)
            )

            rows, columns = np.nonzero(valid)

            if len(rows) < 15:
                continue

            z = depth[rows, columns]

            # Optical coordinates:
            # X right, Y down, Z forward.
            x = (columns - self.cx) * z / self.fx
            y = (rows - self.cy) * z / self.fy

            surface_point = np.median(
                np.column_stack((x, y, z)),
                axis=0,
            )

            u = int(np.median(columns))
            v = int(np.median(rows))

            detections[name] = {
                "pixel": (u, v),
                "surface_camera": surface_point,
                "depth": float(np.median(z)),
            }

            bx, by, bw, bh = cv2.boundingRect(contour)

            cv2.rectangle(
                annotated,
                (bx, by),
                (bx + bw, by + bh),
                (255, 255, 255),
                1,
            )

            cv2.circle(
                annotated,
                (u, v),
                3,
                (255, 255, 255),
                -1,
            )

            label = (
                f"{name}: "
                f"{detections[name]['depth']:.3f} m"
            )

            cv2.putText(
                annotated,
                label,
                (bx, max(15, by - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated, detections

class CameraFrames:
    """Calculate camera pose using encoder readings and model geometry."""

    def __init__(self, model, joint_names):
        self.model = model

        # Separate data for forward kinematics.
        # It does not step physics or read live cube positions.
        self.kinematics = mujoco.MjData(model)

        self.camera_id = self._get_id(
            mujoco.mjtObj.mjOBJ_CAMERA,
            "wrist_camera",
        )
        self.base_id = self._get_id(
            mujoco.mjtObj.mjOBJ_BODY,
            "base_link",
        )

        self.q_addresses = []
        self.encoder_addresses = []

        for joint_name in joint_names:
            joint_id = self._get_id(
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )

            encoder_name = (
                joint_name.removesuffix("_joint") + "_encoder"
            )
            encoder_id = self._get_id(
                mujoco.mjtObj.mjOBJ_SENSOR,
                encoder_name,
            )

            if model.sensor_dim[encoder_id] != 1:
                raise ValueError(
                    f"Expected a scalar encoder: {encoder_name}"
                )

            self.q_addresses.append(
                model.jnt_qposadr[joint_id]
            )
            self.encoder_addresses.append(
                model.sensor_adr[encoder_id]
            )

        self.q_addresses = np.array(self.q_addresses)
        self.encoder_addresses = np.array(
            self.encoder_addresses
        )

        # Optical frame: X right, Y down, Z forward.
        # MuJoCo camera frame: X right, Y up, Z backward.
        self.optical_to_mujoco = np.diag([1.0, -1.0, -1.0])

        self.rotation_base_camera = np.eye(3)
        self.position_base_camera = np.zeros(3)

    def _get_id(self, object_type, name):
        index = mujoco.mj_name2id(
            self.model,
            object_type,
            name,
        )

        if index < 0:
            raise ValueError(f"Missing model element: {name}")

        return index

    def update(self, sensor_data):
        # Read the arm's joint-position sensors.
        measured_q = sensor_data[self.encoder_addresses]

        self.kinematics.qpos[self.q_addresses] = measured_q

        # Calculate the body and camera transforms.
        mujoco.mj_kinematics(self.model, self.kinematics)
        mujoco.mj_camlight(self.model, self.kinematics)

        camera_position_world = (
            self.kinematics.cam_xpos[self.camera_id]
        )
        camera_rotation_world = (
            self.kinematics.cam_xmat[self.camera_id].reshape(3, 3)
        )

        base_position_world = (
            self.kinematics.xpos[self.base_id]
        )
        base_rotation_world = (
            self.kinematics.xmat[self.base_id].reshape(3, 3)
        )

        self.position_base_camera = (
            base_rotation_world.T
            @ (camera_position_world - base_position_world)
        )

        self.rotation_base_camera = (
            base_rotation_world.T
            @ camera_rotation_world
            @ self.optical_to_mujoco
        )

    def point_to_base(self, point_camera):
        return (
            self.rotation_base_camera @ point_camera
            + self.position_base_camera
        )