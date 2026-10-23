"""Stationary safety camera monitor with 3D Kalman filter tracking.

Provides a visual safety net over the robot workspace:
1. Renders overhead stationary RGB-D views of the table.
2. Detects unexpected obstacles / intruders in the workspace.
3. Filters intruder measurements through a 3D constant-velocity Kalman Filter.
4. Signals an interrupt (halt or retreat to safe point) upon safety boundary breach.
5. Provides interactive mocap obstacle spawning and keyboard control for live testing.
"""

from dataclasses import dataclass
from enum import Enum
import cv2
import mujoco
import numpy as np


class SafetyStatus(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    INTERRUPT = "interrupt"
    RETREAT = "retreat"


@dataclass
class SafetyConfig:
    """Configurable safety thresholds and workspace boundaries."""
    # Table workspace bounding box in robot base frame (metres)
    workspace_x: tuple[float, float] = (0.10, 0.52)
    workspace_y: tuple[float, float] = (-0.28, 0.28)
    workspace_z: tuple[float, float] = (-0.02, 0.45)

    # Minimum distance between intruder and active robot pinch point (metres)
    critical_distance: float = 0.08
    warning_distance: float = 0.18

    # Number of consecutive Kalman-confirmed frames before raising interrupt
    persistence_frames: int = 2

    # Retraction target (safe home joint angles in radians)
    safe_joints: tuple[float, ...] = (0.0, -1.5707963, 1.5707963, -1.5707963, -1.5707963, 0.0)


class ObjectTrackingKalmanFilter:
    """3D Constant-Velocity Discrete Kalman Filter for intruder position & velocity.

    State:
        x = [x, y, z, vx, vy, vz]^T  (6 states)
    Measurement:
        y = [x, y, z]^T              (3 states)
    """

    def __init__(self, position_std=0.005, acceleration_std=2.0):
        self.R = (position_std ** 2) * np.eye(3)
        self.accel_variance = acceleration_std ** 2

        self.x = np.zeros(6)
        self.P = np.diag([0.01, 0.01, 0.01, 0.5, 0.5, 0.5])
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
        self.P = np.diag([0.01, 0.01, 0.01, 0.5, 0.5, 0.5])
        self.last_time = None
        self.nis = float("nan")
        self.tracking = False
        self.confirmed_frames = 0
        self.missed_frames = 0

    def predict(self, dt: float):
        if dt <= 0:
            return

        I3 = np.eye(3)
        A = np.block([
            [I3, dt * I3],
            [np.zeros((3, 3)), I3],
        ])

        B = np.vstack([0.5 * (dt ** 2) * I3, dt * I3])
        Q = self.accel_variance * (B @ B.T)

        self.x = A @ self.x
        self.P = A @ self.P @ A.T + Q

    def update(self, measured_xyz: np.ndarray, now: float) -> bool:
        """Update filter with a candidate 3D position detection."""
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

        C = np.hstack([np.eye(3), np.zeros((3, 3))])

        innovation = measured - C @ self.x
        S = C @ self.P @ C.T + self.R
        self.nis = float(innovation.T @ np.linalg.solve(S, innovation))

        # Chi-square gating (99% confidence for 3 DOF is ~11.34)
        if self.nis > 25.0:
            self.missed_frames += 1
            return False

        # Kalman Gain
        K = self.P @ C.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation

        # Joseph form covariance update for numerical stability
        I6 = np.eye(6)
        IKC = I6 - K @ C
        self.P = IKC @ self.P @ IKC.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        self.confirmed_frames += 1
        self.missed_frames = 0
        return True

    def mark_missed(self, max_missed=3):
        """Signal that no detection was observed in the current frame."""
        self.missed_frames += 1
        if self.missed_frames > max_missed:
            self.reset()


class InteractiveObstacle:
    """Controls a mocap test obstacle in MuJoCo.

    Using a mocap body guarantees zero gravity drift, no physical bounce/sliding,
    and instantaneous, solid keyboard positioning.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, name: str = "obstacle"):
        self.model = model
        self.data = data
        self.name = name
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        self.active = False
        self.step_size = 0.02  # metres per keypress

        if self.body_id >= 0:
            self.mocap_id = int(model.body_mocapid[self.body_id])
        else:
            self.mocap_id = -1

    def spawn(self, x: float = 0.35, y: float = 0.0, z: float = 0.08):
        """Place the obstacle inside the table workspace."""
        if self.mocap_id >= 0:
            self.data.mocap_pos[self.mocap_id] = [x, y, z]
            self.data.mocap_quat[self.mocap_id] = [1.0, 0.0, 0.0, 0.0]
        self.active = True
        print(f"[Obstacle] Spawned at ({x:.2f}, {y:.2f}, {z:.2f}) m.")

    def despawn(self):
        """Move the obstacle out of view and deactivate."""
        if self.mocap_id >= 0:
            self.data.mocap_pos[self.mocap_id] = [10.0, 10.0, 10.0]
        self.active = False
        print("[Obstacle] Despawned from workspace.")

    def move(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0):
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
        """Handle keyboard input for obstacle spawning and 3D translation."""
        if key in (ord('s'), ord('S')):
            if self.active:
                self.despawn()
            else:
                self.spawn()
            return True

        if key in (ord('r'), ord('R')):
            if self.active:
                self.spawn(0.35, 0.0, 0.08)
                return True

        if not self.active:
            return False

        step = self.step_size

        # Up Arrow / 'i' -> Move +X (forward toward table back)
        if key in (ord('i'), ord('I'), 0x260000, 2490368, 82, 0):
            self.move(dx=step)
            return True
        # Down Arrow / 'k' -> Move -X (backward toward robot base)
        elif key in (ord('k'), ord('K'), 0x280000, 2621440, 84, 1):
            self.move(dx=-step)
            return True
        # Left Arrow / 'j' -> Move +Y (left)
        elif key in (ord('j'), ord('J'), 0x250000, 2424832, 81, 2):
            self.move(dy=step)
            return True
        # Right Arrow / 'l' -> Move -Y (right)
        elif key in (ord('l'), ord('L'), 0x270000, 2555904, 83, 3):
            self.move(dy=-step)
            return True
        # 'u' -> Move +Z (up)
        elif key in (ord('u'), ord('U')):
            self.move(dz=step)
            return True
        # 'o' -> Move -Z (down)
        elif key in (ord('o'), ord('O')):
            self.move(dz=-step)
            return True

        return False


class StationarySafetyCamera:
    """Manages the stationary overhead camera, vision detection, and Kalman tracking."""

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

        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.camera_id < 0:
            self.fovy = 60.0
        else:
            self.fovy = float(model.cam_fovy[self.camera_id])

        focal = height / (2.0 * np.tan(np.deg2rad(self.fovy) / 2.0))
        self.fx = focal
        self.fy = focal
        self.cx = (width - 1) / 2.0
        self.cy = (height - 1) / 2.0

        self.filter = ObjectTrackingKalmanFilter()
        self.status = SafetyStatus.NORMAL
        self.last_detection = None
        self.detection_pixel = None
        self.rng = np.random.default_rng(2026)

    def render(self, renderer: mujoco.Renderer, data: mujoco.MjData, options: mujoco.MjvOption | None = None):
        """Render RGB and depth from the stationary camera."""
        renderer.disable_depth_rendering()
        renderer.update_scene(data, camera=self.camera_name, scene_option=options)
        rgb = renderer.render().copy()

        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=self.camera_name, scene_option=options)
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()

        return rgb, depth

    def get_camera_extrinsics(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Return camera position and rotation matrix relative to world frame."""
        if self.camera_id < 0:
            cam_pos = np.array([0.3, 0.0, 0.85])
            cam_rot = np.diag([1.0, -1.0, -1.0])
            return cam_pos, cam_rot

        cam_pos = data.cam_xpos[self.camera_id].copy()
        cam_rot = data.cam_xmat[self.camera_id].reshape(3, 3).copy()
        return cam_pos, cam_rot

    def project_world_to_pixel(self, world_xyz: np.ndarray, data: mujoco.MjData) -> tuple[int, int] | None:
        """Pinhole projection of a 3D point into the stationary camera frame."""
        cam_pos, cam_rot = self.get_camera_extrinsics(data)
        optical_to_mujoco = np.diag([1.0, -1.0, -1.0])

        # Point relative to camera optical axis
        diff = world_xyz - cam_pos
        pt_optical = optical_to_mujoco @ (cam_rot.T @ diff)
        z_c = pt_optical[2]

        if z_c <= 0.05:
            return None

        u = int(round(self.cx + self.fx * (pt_optical[0] / z_c)))
        v = int(round(self.cy + self.fy * (pt_optical[1] / z_c)))

        if 0 <= u < self.width and 0 <= v < self.height:
            return (u, v)
        return None

    def detect_intruder(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        data: mujoco.MjData,
    ) -> np.ndarray | None:
        """Identify unexpected intruder directly from RGB-D camera pixels.

        Detects Cyan color cluster in camera frame and reprojects pixels into 3D.
        """
        self.detection_pixel = None
        if rgb is None or depth is None:
            return None

        # Detect Cyan / Teal intruder in camera HSV image
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([80, 100, 50]), np.array([100, 255, 255]))

        # Filter valid depth points
        valid = (mask > 0) & np.isfinite(depth) & (depth > 0.1) & (depth < 1.5)
        rows, cols = np.nonzero(valid)

        if len(rows) < 15:
            return None

        # Camera pixel centroid
        u = int(np.median(cols))
        v = int(np.median(rows))
        self.detection_pixel = (u, v)

        # Depth sampled from camera depth buffer
        z = float(np.median(depth[rows, cols]))

        # Optical coordinate back-projection (pinhole model)
        x_c = (u - self.cx) * z / self.fx
        y_c = (v - self.cy) * z / self.fy
        pt_cam = np.array([x_c, y_c, z])

        # Transform from camera frame to world frame using camera extrinsics
        cam_pos, cam_rot = self.get_camera_extrinsics(data)
        optical_to_mujoco = np.diag([1.0, -1.0, -1.0])
        pt_world = cam_pos + cam_rot @ optical_to_mujoco @ pt_cam

        # Compensate for sphere radius (3.5 cm) so the estimate is the sphere centroid
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
        """Process one camera frame, update Kalman filter, and return safety status."""
        raw_detection = self.detect_intruder(rgb, depth, data)
        self.last_detection = raw_detection

        if raw_detection is not None:
            self.filter.update(raw_detection, now)
        else:
            self.filter.mark_missed()

        # If no active tracked obstacle
        if not self.filter.tracking or self.filter.confirmed_frames < self.config.persistence_frames:
            self.status = SafetyStatus.NORMAL
            return self.status

        intruder_pos = self.filter.position
        cfg = self.config

        # Check proximity to robot end-effector (pinch point)
        if robot_pinch_pos is not None:
            distance_to_robot = float(np.linalg.norm(intruder_pos - robot_pinch_pos))
            if distance_to_robot < cfg.critical_distance:
                self.status = SafetyStatus.RETREAT
                return self.status
            elif distance_to_robot < cfg.warning_distance:
                self.status = SafetyStatus.INTERRUPT
                return self.status

        # Check workspace envelope
        inside_envelope = (
            (cfg.workspace_x[0] <= intruder_pos[0] <= cfg.workspace_x[1]) and
            (cfg.workspace_y[0] <= intruder_pos[1] <= cfg.workspace_y[1]) and
            (cfg.workspace_z[0] <= intruder_pos[2] <= cfg.workspace_z[1])
        )

        if inside_envelope:
            self.status = SafetyStatus.INTERRUPT
        else:
            self.status = SafetyStatus.WARNING

        return self.status

    def annotate_view(self, rgb: np.ndarray, depth: np.ndarray, obstacle_active: bool) -> np.ndarray:
        """Annotate overhead camera view with safety HUD and Kalman filter state."""
        annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        color_map = {
            SafetyStatus.NORMAL: (0, 200, 0),
            SafetyStatus.WARNING: (0, 220, 255),
            SafetyStatus.INTERRUPT: (0, 0, 255),
            SafetyStatus.RETREAT: (255, 0, 255),
        }
        status_color = color_map.get(self.status, (255, 255, 255))

        # Status badge background
        cv2.rectangle(annotated, (6, 6), (230, 28), (30, 30, 30), cv2.FILLED)
        cv2.putText(
            annotated,
            f"SAFETY: {self.status.value.upper()}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            status_color,
            2,
            cv2.LINE_AA,
        )

        # Kalman Filter telemetry overlay
        if self.filter.tracking:
            pos = np.round(self.filter.position, 3)
            vel = np.round(self.filter.velocity, 2)
            speed = f"{self.filter.speed:.2f}"
            nis = f"{self.filter.nis:.1f}" if np.isfinite(self.filter.nis) else "init"

            cv2.putText(annotated, f"KF Pos: {pos} m", (10, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"KF Vel: {vel} m/s (Speed {speed})", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"NIS: {nis} | Confirmed: {self.filter.confirmed_frames}", (10, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

            if self.detection_pixel:
                cv2.circle(annotated, self.detection_pixel, 8, (0, 0, 255), 2)
                cv2.drawMarker(annotated, self.detection_pixel, (0, 255, 255),
                               cv2.MARKER_CROSS, 14, 1)

        # Interactive controls prompt at bottom
        obs_text = "[S] Despawn Obstacle" if obstacle_active else "[S] Spawn Obstacle"
        cv2.putText(
            annotated,
            f"{obs_text} | [IJKL/Arrows] Move | [R] Reset",
            (8, annotated.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return annotated
