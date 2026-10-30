"""Overhead safety monitoring, intrusion tracking and carrying hold."""

from dataclasses import dataclass
from enum import Enum

import cv2
import mujoco
import numpy as np


def cyan_intruder_mask(rgb):
    """Detect solid cyan patches and reject thin colour-edge artifacts."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    mask = cv2.inRange(
        hsv,
        np.array([85, 120, 50], dtype=np.uint8),
        np.array([95, 255, 255], dtype=np.uint8),
    )

    channels = rgb.astype(np.float32)
    red = channels[..., 0]
    green = channels[..., 1]
    blue = channels[..., 2]

    cyan_channels = (
        (green > 1.5 * red)
        & (blue > 1.5 * red)
        & (green > 0.65 * blue)
        & (blue > 0.65 * green)
    )
    mask[~cyan_channels] = 0

    # Remove narrow seams and isolated pixels.
    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    # Do not combine scattered cyan pixels across the image.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned,
        connectivity=8,
    )

    accepted = np.zeros_like(mask)

    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < 30:
            continue

        component = np.zeros_like(mask)
        component[labels == label] = 255

        thickness = cv2.distanceTransform(
            component,
            cv2.DIST_L2,
            5,
        )

        # Require a solid interior rather than a thin edge.
        if float(thickness.max()) < 2.0:
            continue

        accepted[labels == label] = 255

    return accepted


class SafetyStatus(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    INTERRUPT = "interrupt"
    TRACKING = "tracking"
    JOINT_LIMIT = "joint_limit"
    SAFETY_STOP = "safety_stop"
    HOLDING_SAFETY_STOP = "holding_safety_stop"
    RETREAT = "retreat"
    RESUMING = "resuming"
    RECOVERING = "recovering"


@dataclass
class SafetyConfig:
    """Safety thresholds and workspace boundaries."""

    workspace_x: tuple[float, float] = (0.10, 0.52)
    workspace_y: tuple[float, float] = (-0.28, 0.28)
    workspace_z: tuple[float, float] = (-0.02, 0.45)

    standoff_distance: float = 0.22
    critical_distance: float = 0.06
    warning_distance: float = 0.16

    persistence_frames: int = 2
    holding_clearance_frames: int = 10

    safe_joints: tuple[float, ...] = (
        0.0,
        -1.5707963,
        1.5707963,
        -1.5707963,
        -1.5707963,
        0.0,
    )


class CubeCarrySafetySupervisor:
    """Latch a joint-position hold while carrying a cube."""

    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()
        self.paused = False
        self.frozen_q: np.ndarray | None = None
        self.clear_frames = 0
        self.ready_to_resume = False
        self.status = SafetyStatus.NORMAL

    def update(
        self,
        camera: "StationarySafetyCamera",
        holding_object: bool,
        current_q: np.ndarray,
    ) -> SafetyStatus:
        confirmed = (
            camera.filter.tracking
            and camera.filter.confirmed_frames
            >= self.config.persistence_frames
            and camera.status in (
                SafetyStatus.TRACKING,
                SafetyStatus.SAFETY_STOP,
                SafetyStatus.INTERRUPT,
                SafetyStatus.RETREAT,
            )
        )

        if not self.paused:
            self.ready_to_resume = False

            if holding_object and confirmed:
                self.paused = True
                self.frozen_q = np.asarray(
                    current_q, dtype=float
                ).copy()
                self.clear_frames = 0
                self.status = SafetyStatus.HOLDING_SAFETY_STOP
            else:
                self.status = camera.status

            return self.status

        if confirmed:
            self.clear_frames = 0
        else:
            self.clear_frames += 1

        if self.clear_frames >= self.config.holding_clearance_frames:
            self.ready_to_resume = True
            self.status = SafetyStatus.RESUMING
        else:
            self.status = SafetyStatus.HOLDING_SAFETY_STOP

        return self.status

    def finish_resume(self) -> None:
        if not self.ready_to_resume:
            raise RuntimeError(
                "Cannot resume until the camera confirms clearance"
            )

        self.paused = False
        self.frozen_q = None
        self.clear_frames = 0
        self.ready_to_resume = False
        self.status = SafetyStatus.NORMAL


class ObjectTrackingKalmanFilter:
    """Constant-velocity filter: [x, y, z, vx, vy, vz]."""

    def __init__(
        self,
        position_std=0.005,
        acceleration_std=2.0,
    ):
        self.R = position_std**2 * np.eye(3)
        self.accel_variance = acceleration_std**2

        self.x = np.zeros(6)
        self.P = np.diag(
            [0.01, 0.01, 0.01, 0.5, 0.5, 0.5]
        )

        self.last_time = None
        self.nis = float("nan")
        self.tracking = False
        self.confirmed_frames = 0
        self.missed_frames = 0

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:].copy()

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.x[3:]))

    def reset(self):
        self.x[:] = 0.0
        self.P = np.diag(
            [0.01, 0.01, 0.01, 0.5, 0.5, 0.5]
        )
        self.last_time = None
        self.nis = float("nan")
        self.tracking = False
        self.confirmed_frames = 0
        self.missed_frames = 0

    def predict(self, dt: float):
        if dt <= 0:
            return

        identity = np.eye(3)

        A = np.block([
            [identity, dt * identity],
            [np.zeros((3, 3)), identity],
        ])

        B = np.vstack([
            0.5 * dt**2 * identity,
            dt * identity,
        ])

        Q = self.accel_variance * (B @ B.T)

        self.x = A @ self.x
        self.P = A @ self.P @ A.T + Q

    def update(
        self,
        measured_xyz: np.ndarray,
        now: float,
    ) -> bool:
        now = float(now)
        measured = np.asarray(measured_xyz, dtype=float)

        if not self.tracking or self.last_time is None:
            self.x[:3] = measured
            self.x[3:] = 0.0
            self.last_time = now
            self.tracking = True
            self.confirmed_frames = 1
            self.missed_frames = 0
            return True

        dt = now - self.last_time

        if dt > 0:
            self.predict(dt)
            self.last_time = now

        C = np.hstack([
            np.eye(3),
            np.zeros((3, 3)),
        ])

        innovation = measured - C @ self.x
        S = C @ self.P @ C.T + self.R

        self.nis = float(
            innovation.T @ np.linalg.solve(S, innovation)
        )

        if self.nis > 25.0:
            self.missed_frames += 1
            return False

        K = self.P @ C.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation

        correction = np.eye(6) - K @ C

        self.P = (
            correction @ self.P @ correction.T
            + K @ self.R @ K.T
        )
        self.P = 0.5 * (self.P + self.P.T)

        self.confirmed_frames += 1
        self.missed_frames = 0
        return True

    def mark_missed(self, max_missed=3):
        self.missed_frames += 1

        if self.missed_frames > max_missed:
            self.reset()


class InteractiveObstacle:
    """Keyboard-controlled mocap test obstacle."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        name: str = "obstacle",
    ):
        self.model = model
        self.data = data
        self.name = name

        self.body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            name,
        )

        self.active = False
        self.step_size = 0.025

        self.mocap_id = (
            int(model.body_mocapid[self.body_id])
            if self.body_id >= 0
            else -1
        )

    def spawn(
        self,
        x: float = 0.35,
        y: float = 0.0,
        z: float = 0.08,
    ):
        if self.mocap_id >= 0:
            self.data.mocap_pos[self.mocap_id] = [x, y, z]
            self.data.mocap_quat[self.mocap_id] = [
                1.0, 0.0, 0.0, 0.0
            ]

        self.active = True

        print(
            f"[Obstacle] Spawned at "
            f"({x:.2f}, {y:.2f}, {z:.2f}) m."
        )
        print(
            "[Obstacle] Controls: "
            "[WASD / Arrows] Move X-Y | "
            "[E/C or PgUp/PgDn] Height Z | "
            "[Space/P] Despawn"
        )

    def despawn(self):
        if self.mocap_id >= 0:
            self.data.mocap_pos[self.mocap_id] = [
                10.0, 10.0, 10.0
            ]

        self.active = False
        print("[Obstacle] Despawned from workspace.")

    def move(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
    ):
        if self.mocap_id >= 0 and self.active:
            self.data.mocap_pos[self.mocap_id, 0] += dx
            self.data.mocap_pos[self.mocap_id, 1] += dy
            self.data.mocap_pos[self.mocap_id, 2] += dz

    @property
    def position(self) -> np.ndarray | None:
        if self.mocap_id >= 0 and self.active:
            return self.data.mocap_pos[self.mocap_id].copy()
        return None

    def handle_key(self, key: int) -> bool:
        if key in (32, ord("p"), ord("P")):
            if self.active:
                self.despawn()
            else:
                self.spawn()
            return True

        if not self.active:
            if key in (ord("s"), ord("S")):
                self.spawn()
                return True
            return False

        if key in (ord("r"), ord("R")):
            self.spawn(0.35, 0.0, 0.08)
            return True

        step = self.step_size

        if key in (
            ord("w"), ord("W"), ord("i"), ord("I"),
            0x260000, 2490368, 82, 38, 0,
        ):
            self.move(dx=step)
            return True

        elif key in (
            ord("s"), ord("S"), ord("k"), ord("K"),
            0x280000, 2621440, 84, 40, 1,
        ):
            self.move(dx=-step)
            return True

        elif key in (
            ord("a"), ord("A"), ord("j"), ord("J"),
            0x250000, 2424832, 81, 37, 2,
        ):
            self.move(dy=step)
            return True

        elif key in (
            ord("d"), ord("D"), ord("l"), ord("L"),
            0x270000, 2555904, 83, 39, 3,
        ):
            self.move(dy=-step)
            return True

        elif key in (
            ord("e"), ord("E"), ord("u"), ord("U"),
            ord("+"), ord("="),
            0x210000, 2162688, 33,
        ):
            self.move(dz=step)
            return True

        elif key in (
            ord("c"), ord("C"), ord("o"), ord("O"),
            ord("-"), ord("_"),
            0x220000, 2228224, 34,
        ):
            self.move(dz=-step)
            return True

        return False


class StationarySafetyCamera:
    """Overhead RGB-D detection and Kalman tracking."""

    def __init__(
        self,
        model: mujoco.MjModel,
        camera_name: str = "safety_camera",
        width: int = 320,
        height: int = 240,
        config: SafetyConfig | None = None,
    ):
        self.model = model
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.config = config or SafetyConfig()

        self.camera_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_name,
        )

        self.fovy = (
            float(model.cam_fovy[self.camera_id])
            if self.camera_id >= 0
            else 60.0
        )

        focal = height / (
            2.0 * np.tan(np.deg2rad(self.fovy) / 2.0)
        )

        self.fx = focal
        self.fy = focal
        self.cx = (width - 1) / 2.0
        self.cy = (height - 1) / 2.0

        self.filter = ObjectTrackingKalmanFilter()
        self.status = SafetyStatus.NORMAL
        self.last_detection = None
        self.detection_pixel = None
        self.rng = np.random.default_rng(2026)

    def render(
        self,
        renderer: mujoco.Renderer,
        data: mujoco.MjData,
        options: mujoco.MjvOption | None = None,
    ):
        renderer.disable_depth_rendering()
        renderer.update_scene(
            data,
            camera=self.camera_name,
            scene_option=options,
        )
        rgb = renderer.render().copy()

        renderer.enable_depth_rendering()
        renderer.update_scene(
            data,
            camera=self.camera_name,
            scene_option=options,
        )
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()

        return rgb, depth

    def get_camera_extrinsics(
        self,
        data: mujoco.MjData,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.camera_id < 0:
            return (
                np.array([0.3, 0.0, 0.85]),
                np.diag([1.0, -1.0, -1.0]),
            )

        return (
            data.cam_xpos[self.camera_id].copy(),
            data.cam_xmat[self.camera_id].reshape(3, 3).copy(),
        )

    def project_world_to_pixel(
        self,
        world_xyz: np.ndarray,
        data: mujoco.MjData,
    ) -> tuple[int, int] | None:
        cam_pos, cam_rot = self.get_camera_extrinsics(data)
        optical_to_mujoco = np.diag([1.0, -1.0, -1.0])

        pt_optical = (
            optical_to_mujoco @ (cam_rot.T @ (world_xyz - cam_pos))
        )
        z_c = pt_optical[2]

        if z_c <= 0.05:
            return None

        u = int(round(self.cx + self.fx * pt_optical[0] / z_c))
        v = int(round(self.cy + self.fy * pt_optical[1] / z_c))

        if 0 <= u < self.width and 0 <= v < self.height:
            return u, v
        return None

    def detect_intruder(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        data: mujoco.MjData,
    ) -> np.ndarray | None:
        self.detection_pixel = None

        if rgb is None or depth is None:
            return None

        mask = cyan_intruder_mask(rgb)

        valid = (
            (mask > 0)
            & np.isfinite(depth)
            & (depth > 0.1)
            & (depth < 1.5)
        )

        rows, cols = np.nonzero(valid)

        if len(rows) < 15:
            return None

        u = int(np.median(cols))
        v = int(np.median(rows))
        self.detection_pixel = (u, v)

        z = float(np.median(depth[rows, cols]))

        pt_cam = np.array([
            (u - self.cx) * z / self.fx,
            (v - self.cy) * z / self.fy,
            z,
        ])

        cam_pos, cam_rot = self.get_camera_extrinsics(data)
        optical_to_mujoco = np.diag([1.0, -1.0, -1.0])
        pt_world = cam_pos + cam_rot @ optical_to_mujoco @ pt_cam

        # Existing radius compensation for the test sphere.
        pt_world[2] -= 0.035
        return pt_world

    def update(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        data: mujoco.MjData,
        now: float,
        robot_pinch_pos: np.ndarray | None = None,
    ) -> SafetyStatus:
        raw_detection = self.detect_intruder(rgb, depth, data)
        self.last_detection = raw_detection

        if raw_detection is not None:
            self.filter.update(raw_detection, now)
        else:
            self.filter.mark_missed()

        if (
            not self.filter.tracking
            or self.filter.confirmed_frames
            < self.config.persistence_frames
        ):
            self.status = SafetyStatus.NORMAL
            return self.status

        intruder_pos = self.filter.position
        cfg = self.config

        if robot_pinch_pos is not None:
            distance = float(
                np.linalg.norm(intruder_pos - robot_pinch_pos)
            )

            if distance < cfg.critical_distance:
                self.status = SafetyStatus.SAFETY_STOP
                return self.status

            elif distance < cfg.warning_distance:
                self.status = SafetyStatus.TRACKING
                return self.status

        inside_envelope = (
            cfg.workspace_x[0] <= intruder_pos[0] <= cfg.workspace_x[1]
            and cfg.workspace_y[0] <= intruder_pos[1] <= cfg.workspace_y[1]
            and cfg.workspace_z[0] <= intruder_pos[2] <= cfg.workspace_z[1]
        )

        self.status = (
            SafetyStatus.TRACKING
            if inside_envelope
            else SafetyStatus.WARNING
        )
        return self.status

    def annotate_view(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        obstacle_active: bool,
    ) -> np.ndarray:
        annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        color_map = {
            SafetyStatus.NORMAL: (0, 200, 0),
            SafetyStatus.WARNING: (0, 220, 255),
            SafetyStatus.TRACKING: (255, 200, 0),
            SafetyStatus.JOINT_LIMIT: (0, 165, 255),
            SafetyStatus.SAFETY_STOP: (0, 0, 255),
            SafetyStatus.HOLDING_SAFETY_STOP: (0, 0, 255),
            SafetyStatus.INTERRUPT: (0, 0, 255),
            SafetyStatus.RETREAT: (255, 0, 255),
            SafetyStatus.RECOVERING: (255, 0, 255),
            SafetyStatus.RESUMING: (0, 165, 255),
        }

        cv2.rectangle(
            annotated, (6, 6), (250, 28),
            (30, 30, 30), cv2.FILLED,
        )
        cv2.putText(
            annotated,
            f"SAFETY: {self.status.value.upper()}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_map.get(self.status, (255, 255, 255)),
            2,
            cv2.LINE_AA,
        )

        if self.filter.tracking:
            pos = np.round(self.filter.position, 3)
            vel = np.round(self.filter.velocity, 2)
            speed = f"{self.filter.speed:.2f}"
            nis = (
                f"{self.filter.nis:.1f}"
                if np.isfinite(self.filter.nis)
                else "init"
            )

            lines = (
                (f"KF Pos: {pos} m", 44),
                (f"KF Vel: {vel} m/s (Speed {speed})", 58),
                (
                    f"NIS: {nis} | Confirmed: "
                    f"{self.filter.confirmed_frames}",
                    72,
                ),
            )

            for text, y in lines:
                cv2.putText(
                    annotated, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0, 255, 255), 1, cv2.LINE_AA,
                )

            if self.detection_pixel:
                cv2.circle(
                    annotated, self.detection_pixel,
                    8, (0, 0, 255), 2,
                )
                cv2.drawMarker(
                    annotated, self.detection_pixel,
                    (0, 255, 255), cv2.MARKER_CROSS, 14, 1,
                )

        prompt = (
            "[Space/P] Despawn"
            if obstacle_active
            else "[Space/P] Spawn"
        )
        cv2.putText(
            annotated,
            f"{prompt} | [WASD/Arrows] X-Y | [E/C or PgUp/Dn] Z",
            (8, annotated.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return annotated


def detect_cyan_intruder_wrist(
    rgb: np.ndarray,
    depth: np.ndarray,
    fovy: float = 60.0,
) -> dict | None:
    if rgb is None or depth is None:
        return None

    h, w = rgb.shape[:2]
    focal = h / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    mask = cyan_intruder_mask(rgb)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 20:
        return None

    bx, by, bw, bh = cv2.boundingRect(contour)
    component = np.zeros_like(mask)
    cv2.drawContours(
        component, [contour], -1, 255, cv2.FILLED
    )

    valid = (
        (component > 0)
        & np.isfinite(depth)
        & (depth > 0.04)
        & (depth < 2.0)
    )
    rows, cols = np.nonzero(valid)

    if len(rows) < 10:
        return None

    u = int(np.median(cols))
    v = int(np.median(rows))
    z = float(np.median(depth[rows, cols]))

    return {
        "pixel": (u, v),
        "point_camera": np.array([
            (u - cx) * z / focal,
            (v - cy) * z / focal,
            z,
        ]),
        "depth": z,
        "bbox": (bx, by, bw, bh),
        "area": float(cv2.contourArea(contour)),
    }


class VisualServoingAvoidanceController:
    """Camera gaze tracking and standoff-distance control."""

    def __init__(
        self,
        model: mujoco.MjModel,
        q_addresses: np.ndarray,
        v_addresses: np.ndarray,
        camera_name: str = "wrist_camera",
        config: SafetyConfig | None = None,
    ):
        self.model = model
        self.q_addresses = np.asarray(q_addresses)
        self.v_addresses = np.asarray(v_addresses)
        self.config = config or SafetyConfig()

        self.camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        self.camera_body_id = (
            int(model.cam_bodyid[self.camera_id])
            if self.camera_id >= 0
            else -1
        )
        self.pinch_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "gripper_pinch"
        )
        self.base_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )

        self.gain_rot = 2.0
        self.gain_dist = 1.2
        self.gain_lat = 0.8

        self.max_lin_speed = 0.10
        self.max_ang_speed = 0.6
        self.max_joint_speed = 0.45
        self.damping = 0.005
        self.rotation_weight = 0.15

        from robot_utils import joint_limits

        self.lower_limits, self.upper_limits = joint_limits(
            model, self.q_addresses
        )

        self.safe_lower = np.array([
            -2.60, -2.44, 0.52, -2.79, -2.79, -3.14
        ])
        self.safe_upper = np.array([
            2.60, -0.70, 2.62, -0.35, -0.35, 3.14
        ])
        self.preferred_q = np.array([
            0.0,
            -np.pi / 2,
            np.pi / 2,
            -np.pi / 2,
            -np.pi / 2,
            0.0,
        ])

        self.k_null_posture = 1.2
        self.k_null_repulse = 0.8
        self.last_rates = np.zeros(6)
        self.status = SafetyStatus.TRACKING
        self.telemetry = {}

    def compute_command(
        self,
        data: mujoco.MjData,
        frames,
        target_pos_base: np.ndarray,
        current_q: np.ndarray,
        target_vel_base: np.ndarray | None = None,
        holding_object: bool = False,
        held_size: float = 0.04,
    ) -> tuple[SafetyStatus, np.ndarray, dict]:
        self.telemetry = {}

        if (
            target_pos_base is None
            or not np.all(np.isfinite(target_pos_base))
        ):
            return SafetyStatus.NORMAL, np.zeros(6), {}

        cam_pos_base = frames.position_base_camera
        cam_rot_base = frames.rotation_base_camera
        base_rot_w = data.xmat[frames.base_id].reshape(3, 3)
        base_pos_w = data.xpos[frames.base_id]

        d_base = target_pos_base - cam_pos_base
        distance = float(np.linalg.norm(d_base))
        p_c = cam_rot_base.T @ d_base
        z_c = float(p_c[2])

        if self.pinch_site_id >= 0:
            pinch_pos_base = base_rot_w.T @ (
                data.site_xpos[self.pinch_site_id] - base_pos_w
            )
            dist_to_pinch = float(np.linalg.norm(
                target_pos_base - pinch_pos_base
            ))
        else:
            dist_to_pinch = distance

        effective_critical = (
            self.config.critical_distance
            + (held_size * 0.5 if holding_object else 0.0)
        )

        if (
            dist_to_pinch < effective_critical
            or distance < effective_critical
        ):
            self.status = SafetyStatus.SAFETY_STOP
            self.telemetry = {
                "distance": distance,
                "pinch_distance": dist_to_pinch,
                "standoff_error": z_c - self.config.standoff_distance,
                "orientation_error": 0.0,
                "joint_limit_breached": False,
                "rates": np.zeros(6),
                "holding_object": holding_object,
            }
            return self.status, np.zeros(6), self.telemetry

        u_dir = p_c / max(1e-4, distance)
        e_rot = np.array([-u_dir[1], u_dir[0], 0.0])
        ang_error = float(np.linalg.norm(e_rot))
        omega = self.gain_rot * e_rot

        dist_err = z_c - self.config.standoff_distance
        v_lin = np.array([
            self.gain_lat * p_c[0],
            self.gain_lat * p_c[1],
            self.gain_dist * dist_err,
        ])

        scale = max(
            1.0,
            np.linalg.norm(v_lin) / self.max_lin_speed,
            np.linalg.norm(omega) / self.max_ang_speed,
        )
        v_lin /= scale
        omega /= scale

        jac_p = np.zeros((3, self.model.nv))
        jac_r = np.zeros((3, self.model.nv))

        mujoco.mj_jac(
            self.model,
            data,
            jac_p,
            jac_r,
            data.cam_xpos[self.camera_id],
            self.camera_body_id,
        )

        w_to_opt = (base_rot_w @ cam_rot_base).T
        jac_lin = w_to_opt @ jac_p[:, self.v_addresses]
        jac_ang = w_to_opt @ jac_r[:, self.v_addresses]

        # Translation plus two gaze rotation axes.
        task_jac = np.vstack((
            jac_lin,
            self.rotation_weight * jac_ang[:2, :],
        ))
        task_twist = np.r_[
            v_lin,
            self.rotation_weight * omega[:2],
        ]

        weights = np.ones(6)
        buffer = np.deg2rad(15.0)

        for j in range(6):
            min_dist = min(
                current_q[j] - self.safe_lower[j],
                self.safe_upper[j] - current_q[j],
            )
            if min_dist < buffer:
                ratio = max(0.01, min_dist / buffer)
                weights[j] = 1.0 + 1.0 / ratio**2

        W_inv = np.diag(1.0 / weights)
        JWJt = (
            task_jac @ W_inv @ task_jac.T
            + self.damping**2 * np.eye(5)
        )
        task_pinv = (
            W_inv @ task_jac.T
            @ np.linalg.solve(JWJt, np.eye(5))
        )

        rates_task = task_pinv @ task_twist
        N_null = np.eye(6) - task_pinv @ task_jac

        pref_q = self.preferred_q.copy()
        pref_q[0] = np.clip(
            np.arctan2(target_pos_base[1], target_pos_base[0]),
            self.safe_lower[0],
            self.safe_upper[0],
        )

        q_null_posture = self.k_null_posture * (pref_q - current_q)
        q_null_repulse = np.zeros(6)

        for j in range(6):
            mid = 0.5 * (self.safe_lower[j] + self.safe_upper[j])
            span = 0.5 * (self.safe_upper[j] - self.safe_lower[j])
            norm_q = (current_q[j] - mid) / max(0.1, span)

            if abs(norm_q) > 0.65:
                severity = (abs(norm_q) - 0.65) / 0.35
                q_null_repulse[j] = (
                    -np.sign(norm_q)
                    * self.k_null_repulse
                    * severity**2
                )

        rates = (
            rates_task
            + N_null @ (q_null_posture + q_null_repulse)
        )

        max_rate = np.max(np.abs(rates))
        if max_rate > self.max_joint_speed:
            rates = rates / max_rate * self.max_joint_speed

        joint_limit_breached = False
        hard_margin = 0.02

        for j in range(6):
            if (
                current_q[j] <= self.lower_limits[j] + hard_margin
                and rates[j] < 0
            ):
                rates[j] = 0.0
                joint_limit_breached = True

            elif (
                current_q[j] >= self.upper_limits[j] - hard_margin
                and rates[j] > 0
            ):
                rates[j] = 0.0
                joint_limit_breached = True

        self.status = (
            SafetyStatus.JOINT_LIMIT
            if joint_limit_breached
            else SafetyStatus.TRACKING
        )
        self.last_rates = rates.copy()

        self.telemetry = {
            "distance": distance,
            "pinch_distance": dist_to_pinch,
            "standoff_error": dist_err,
            "orientation_error": ang_error,
            "joint_limit_breached": joint_limit_breached,
            "rates": rates.copy(),
            "target_cam": p_c.copy(),
        }
        return self.status, rates, self.telemetry