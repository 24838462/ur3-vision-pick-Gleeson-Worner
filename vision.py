import cv2
import mujoco
import numpy as np


class CubeDetector:
    def __init__(self, width, height, fovy_deg):
        # Intrinsics for the ideal simulated pinhole camera.
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
            "purple": [
                ((140, 90, 35), (165, 255, 255)),
            ],
        }

    def detect(self, rgb, depth):
        """Detect coloured objects and recover visible surface points."""
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

            component = np.zeros_like(mask)

            cv2.drawContours(
                component,
                [contour],
                -1,
                255,
                thickness=cv2.FILLED,
            )

            component = cv2.bitwise_and(component, mask)

            # Remove edge pixels that can mix foreground/background.
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

            # Optical coordinates: X right, Y down, Z forward.
            x = (columns - self.cx) * z / self.fx
            y = (rows - self.cy) * z / self.fy

            points_camera = np.column_stack((x, y, z))
            surface_point = np.median(points_camera, axis=0)

            u = int(np.median(columns))
            v = int(np.median(rows))

            bx, by, bw, bh = cv2.boundingRect(contour)

            # A cropped marker is not accepted for placement measurement.
            touches_border = (
                bx <= 0
                or by <= 0
                or bx + bw >= rgb.shape[1]
                or by + bh >= rgb.shape[0]
            )

            detections[name] = {
                "pixel": (u, v),
                "surface_camera": surface_point,
                "points_camera": points_camera,
                "depth": float(np.median(z)),
                "touches_border": touches_border,
            }

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

            cv2.putText(
                annotated,
                f"{name}: {detections[name]['depth']:.3f} m",
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

        # Optical: X right, Y down, Z forward.
        # MuJoCo camera: X right, Y up, Z backward.
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
        measured_q = sensor_data[self.encoder_addresses]
        self.kinematics.qpos[self.q_addresses] = measured_q

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

    def points_to_base(self, points_camera):
        return (
            points_camera @ self.rotation_base_camera.T
            + self.position_base_camera
        )


def estimate_table_cube(detection, camera_frames, table_top):
    """Estimate an upright cube resting directly on the table."""
    points_base = camera_frames.points_to_base(
        detection["points_camera"]
    )

    if len(points_base) < 30:
        return None

    top_reference = np.percentile(points_base[:, 2], 90)

    top_points = points_base[
        np.abs(points_base[:, 2] - top_reference) < 0.0015
    ]

    if len(top_points) < 25:
        return None

    centred = top_points - top_points.mean(axis=0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    normal = axes[-1]

    if abs(normal[2]) < np.cos(np.deg2rad(10)):
        return None

    top_z = float(np.median(top_points[:, 2]))

    # Cube assumption: edge length equals height above the table.
    size = top_z - table_top

    if not 0.015 <= size <= 0.075:
        return None

    xy = np.ascontiguousarray(
        top_points[:, :2],
        dtype=np.float32,
    )

    rectangle = cv2.minAreaRect(xy)
    (centre_x, centre_y), (width, length), _ = rectangle

    if min(width, length) < 0.75 * size:
        return None

    if max(width, length) > 1.15 * size:
        return None

    if min(width, length) / max(width, length) < 0.80:
        return None

    corners = cv2.boxPoints(rectangle)
    edge = corners[1] - corners[0]

    yaw = float(np.arctan2(edge[1], edge[0]))
    yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4

    return {
        "size": float(size),
        "centre_base": np.array([
            centre_x,
            centre_y,
            top_z - size / 2.0,
        ]),
        "top_centre_base": np.array([
            centre_x,
            centre_y,
            top_z,
        ]),
        "yaw": yaw,
        "outline_size": np.array([width, length]),
        "top_points": len(top_points),
    }


def estimate_placement_square(detection, camera_frames, table_top):
    """Measure the purple marker position from its RGB-D points."""
    if detection.get("touches_border", False):
        return None

    points_base = camera_frames.points_to_base(
        detection["points_camera"]
    )

    points = points_base[
        np.abs(points_base[:, 2] - table_top) < 0.005
    ]

    if len(points) < 30:
        return None

    xy = np.ascontiguousarray(
        points[:, :2],
        dtype=np.float32,
    )

    rectangle = cv2.minAreaRect(xy)
    (centre_x, centre_y), (width, length), _ = rectangle

    # Known 110 mm marker size is used only for quality checks.
    if min(width, length) < 0.09:
        return None

    if max(width, length) > 0.125:
        return None

    if min(width, length) / max(width, length) < 0.85:
        return None

    return {
        "centre_base": np.array([
            centre_x,
            centre_y,
            np.median(points[:, 2]),
        ]),
        "width": float(width),
        "length": float(length),
    }