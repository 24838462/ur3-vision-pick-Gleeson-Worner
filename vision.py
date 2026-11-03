import cv2
import mujoco
import numpy as np
from robot_utils import get_id
STABLE_FRAMES = 10

class CubeDetector:

    def __init__(self, width, height, fovy_deg):
        focal = height / (2.0 * np.tan(np.deg2rad(fovy_deg) / 2.0))
        self.fx = focal
        self.fy = focal
        self.cx = (width - 1) / 2.0
        self.cy = (height - 1) / 2.0
        self.ranges = {
            'red': [((0, 100, 35), (10, 255, 255)), ((170, 100, 35), (179, 255, 255))],
            'green': [((35, 100, 35), (85, 255, 255))],
            'blue': [((90, 100, 35), (135, 255, 255))],
            'purple': [((140, 90, 35), (165, 255, 255))],
        }

    def detect(self, rgb, depth):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        annotated = rgb.copy()
        detections = {}
        kernel = np.ones((3, 3), dtype=np.uint8)
        for name, ranges in self.ranges.items():
            mask = np.zeros(depth.shape, dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(
                    hsv,
                    np.array(lower, dtype=np.uint8),
                    np.array(upper, dtype=np.uint8),
                )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) < 40:
                continue
            component = np.zeros_like(mask)
            cv2.drawContours(component, [contour], -1, 255, cv2.FILLED)
            component &= mask
            component = cv2.erode(component, kernel)
            valid = (component > 0) & np.isfinite(depth) & (depth > 0.02) & (depth < 2.0)
            rows, columns = np.nonzero(valid)
            if len(rows) < 15:
                continue
            z = depth[rows, columns]
            x = (columns - self.cx) * z / self.fx
            y = (rows - self.cy) * z / self.fy
            points = np.column_stack((x, y, z))
            u = int(np.median(columns))
            v = int(np.median(rows))
            bx, by, bw, bh = cv2.boundingRect(contour)
            detections[name] = {
                'pixel': (u, v),
                'surface_camera': np.median(points, axis=0),
                'points_camera': points,
                'depth': float(np.median(z)),
                'touches_border': bx <= 0 or by <= 0 or bx + bw >= rgb.shape[1] or (by + bh >= rgb.shape[0]),
            }
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (255, 255, 255), 1)
            cv2.circle(annotated, (u, v), 3, (255, 255, 255), -1)
            cv2.putText(annotated, f'{name}: {np.median(z):.3f} m', (bx, max(15, by - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return (annotated, detections)

class CameraFrames:
    """Forward kinematics using joint encoder measurements."""

    def __init__(self, model, joint_names):
        self.model = model
        self.kinematics = mujoco.MjData(model)
        self.camera_id = self._id(mujoco.mjtObj.mjOBJ_CAMERA, 'wrist_camera')
        self.base_id = self._id(mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        self.q_addresses = []
        self.encoder_addresses = []
        for name in joint_names:
            joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, name)
            sensor_id = self._id(
                mujoco.mjtObj.mjOBJ_SENSOR,
                name.removesuffix('_joint') + '_encoder',
            )
            if model.sensor_dim[sensor_id] != 1:
                raise ValueError('Expected scalar joint encoder.')
            self.q_addresses.append(model.jnt_qposadr[joint_id])
            self.encoder_addresses.append(model.sensor_adr[sensor_id])
        self.q_addresses = np.array(self.q_addresses)
        self.encoder_addresses = np.array(self.encoder_addresses)
        self.optical_to_mujoco = np.diag([1.0, -1.0, -1.0])
        self.rotation_base_camera = np.eye(3)
        self.position_base_camera = np.zeros(3)

    def _id(self, kind, name):
        return get_id(self.model, kind, name)

    def update(self, estimated_q):
        estimated_q = np.asarray(estimated_q, dtype=float)

        if (
            estimated_q.shape != self.q_addresses.shape
            or not np.all(np.isfinite(estimated_q))
        ):
            raise ValueError(
                "CameraFrames requires finite estimated joint angles."
            )

        self.kinematics.qpos[self.q_addresses] = estimated_q

        mujoco.mj_kinematics(self.model, self.kinematics)
        mujoco.mj_comPos(self.model, self.kinematics)
        mujoco.mj_camlight(self.model, self.kinematics)

        data = self.kinematics
        base_rotation = data.xmat[self.base_id].reshape(3, 3)
        camera_rotation = data.cam_xmat[self.camera_id].reshape(3, 3)

        self.position_base_camera = (
            base_rotation.T
            @ (
                data.cam_xpos[self.camera_id]
                - data.xpos[self.base_id]
            )
        )

        self.rotation_base_camera = (
            base_rotation.T
            @ camera_rotation
            @ self.optical_to_mujoco
        )

    def points_to_base(self, points):
        return points @ self.rotation_base_camera.T + self.position_base_camera

    def base_to_world(self, point):
        data = self.kinematics
        return data.xmat[self.base_id].reshape(3, 3) @ point + data.xpos[self.base_id]

def estimate_table_cube(detection, frames, table_top):
    """Measure an upright cube resting on the table."""
    points = frames.points_to_base(detection['points_camera'])
    top = _top_points(points)
    if top is None:
        return None
    top_z = float(np.median(top[:, 2]))
    size = top_z - table_top
    if not 0.015 <= size <= 0.075:
        return None
    rectangle = _rectangle(top)
    (centre_x, centre_y), (width, length), _ = rectangle
    if min(
        width,
        length,
    ) < 0.75 * size or max(
        width,
        length,
    ) > 1.15 * size or min(width, length) / max(width, length) < 0.8:
        return None
    yaw = _square_yaw(rectangle)
    return {
        'size': float(size),
        'centre_base': np.array([centre_x, centre_y, top_z - size / 2]),
        'top_centre_base': np.array([centre_x, centre_y, top_z]),
        'yaw': float(yaw),
        'outline_size': np.array([width, length]),
        'top_points': len(top),
    }

def estimate_placement_square(detection, frames, table_top):
    """Locate the purple square during the initial search."""
    if detection.get('touches_border', False):
        return None
    points = frames.points_to_base(detection['points_camera'])
    points = points[np.abs(points[:, 2] - table_top) < 0.005]
    if len(points) < 30:
        return None
    rectangle = _rectangle(points)
    (centre_x, centre_y), (width, length), _ = rectangle
    if min(
        width,
        length,
    ) < 0.09 or max(width, length) > 0.125 or min(width, length) / max(width, length) < 0.85:
        return None
    return {
        'centre_base': np.array([centre_x, centre_y, np.median(points[:, 2])]),
        'width': float(width),
        'length': float(length),
    }

def estimate_support_square(detection, frames, expected_size):
    """Measure the top surface of purple or a placed cube.

    Expected edge size comes from the marker configuration or
    the cube size recorded earlier through vision. Surface height
    and position are measured afresh from RGB-D.
    """
    if detection is None or detection.get('touches_border', False):
        return None
    points = frames.points_to_base(detection['points_camera'])
    points = points[np.all(np.isfinite(points), axis=1)]
    top = _top_points(points)
    if top is None:
        return None
    rectangle = _rectangle(top)
    (x, y), (width, length), _ = rectangle
    if min(
        width,
        length,
    ) < 0.8 * expected_size or max(
        width,
        length,
    ) > 1.15 * expected_size or min(width, length) / max(width, length) < 0.85:
        return None
    yaw = _square_yaw(rectangle)
    return {
        'centre_base': np.array([x, y, np.median(top[:, 2])]),
        'size': float(expected_size),
        'yaw': float(yaw),
        'surface': True,
        'outline_size': np.array([width, length]),
    }

def stable_cube(history):
    if len(history) < STABLE_FRAMES:
        return None
    centres = np.array([item['centre_base'] for item in history])
    sizes = np.array([item['size'] for item in history])
    angles = np.array([item['yaw'] for item in history])
    if not all((np.all(np.isfinite(values)) for values in (centres, sizes, angles))):
        return None
    centre = np.median(centres, axis=0)
    size = float(np.median(sizes))
    orientation_mean = np.mean(np.exp(4j * angles))
    if abs(orientation_mean) < 0.95:
        return None
    yaw = float(np.angle(orientation_mean) / 4)
    angle_errors = (angles - yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    if np.max(np.linalg.norm(centres - centre, axis=1)) > 0.002 or np.ptp(sizes) > 0.002:
        return None
    if np.max(np.abs(angle_errors)) > np.deg2rad(5):
        return None
    return {'centre_base': centre.copy(), 'size': size, 'yaw': yaw}

def stable_marker(history):
    if len(history) < STABLE_FRAMES:
        return None
    centres = np.array([item['centre_base'] for item in history])
    centre = np.median(centres, axis=0)
    if not np.all(np.isfinite(centres)) or np.max(np.linalg.norm(centres - centre, axis=1)) > 0.002:
        return None
    return {'centre_base': centre.copy()}

def world_target(name, measurement, frames):
    result = dict(measurement)
    result.update(name=name, centre_world=frames.base_to_world(measurement['centre_base']))
    return result

def _top_points(points):
    if len(points) < 30:
        return None
    reference = np.percentile(points[:, 2], 90)
    top = points[np.abs(points[:, 2] - reference) < 0.0015]
    if len(top) < 25:
        return None
    _, _, axes = np.linalg.svd(top - top.mean(axis=0), full_matrices=False)
    if abs(axes[-1, 2]) < np.cos(np.deg2rad(10)):
        return None
    return top

def _rectangle(points):
    return cv2.minAreaRect(np.ascontiguousarray(points[:, :2], dtype=np.float32))

def _square_yaw(rectangle):
    corners = cv2.boxPoints(rectangle)
    edge = corners[1] - corners[0]
    return (np.arctan2(edge[1], edge[0]) + np.pi / 4) % (np.pi / 2) - np.pi / 4
