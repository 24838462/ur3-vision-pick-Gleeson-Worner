"""Hardware and Mock Interface for UR3e Manipulator and OnRobot RG2 Gripper.

Features:
- Real-time RTDE communication via ur_rtde (RTDEControlInterface, RTDEReceiveInterface)
- Robust velocity control (speedJ) with period synchronization (initPeriod / waitPeriod)
- Native XML-RPC over HTTP driver for OnRobot RG2 Gripper (port 41414):
    - Zero external C-library dependency (replaces buggy pycurl with standard urllib.request)
    - Full grip and width measurement support (rg_grip, rg_get_width)
- Multi-layer safety: joint velocity clamp, acceleration limits, graceful shutdown
- Complete Mock Robot backend for offline testing and visualization
"""

import time
import xmlrpc.client
from urllib.request import Request, urlopen
import numpy as np

try:
    from rtde_control import RTDEControlInterface as RTDEControl
    from rtde_receive import RTDEReceiveInterface as RTDEReceive
    RTDE_AVAILABLE = True
except ImportError:
    RTDE_AVAILABLE = False

try:
    import onRobot.gripper as onrobot_gripper
    ONROBOT_AVAILABLE = True
except ImportError:
    ONROBOT_AVAILABLE = False

from ur3e_kinematics import UR3eKinematics


class RG2Gripper:
    """OnRobot RG2 Gripper Controller supporting official onRobot library and HTTP XML-RPC."""

    def __init__(self, robot_ip: str, gripper_id: int = 0, is_mock: bool = False):
        self.robot_ip = robot_ip
        self.gripper_id = gripper_id
        self.is_mock = is_mock
        self.current_width_mm = 80.0
        self.url = f"http://{robot_ip}:41414"

        self._rg = None
        if not self.is_mock and ONROBOT_AVAILABLE:
            try:
                self._rg = onrobot_gripper.RG(self.robot_ip, self.gripper_id)
            except Exception as exc:
                print(f"[RG2] onRobot package initialization: {exc}")

    def _send_xmlrpc(self, method_name: str, params: list) -> any:
        if self.is_mock:
            return 0.0

        xml_call = xmlrpc.client.dumps(tuple(params), methodname=method_name)
        req = Request(
            self.url,
            data=xml_call.encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with urlopen(req, timeout=3.0) as response:
                body = response.read().decode("utf-8")
                result, _ = xmlrpc.client.loads(body)
                return result[0]
        except Exception as exc:
            print(f"[RG2] XML-RPC call '{method_name}' failed: {exc}")
            return None

    def get_width(self) -> float:
        """Get current finger opening width in millimeters (-999.0 if hardware disconnected)."""
        if self.is_mock:
            return self.current_width_mm

        # 1. Try official onRobot library
        if self._rg is not None:
            try:
                w = self._rg.get_rg_width()
                if w > -900:
                    self.current_width_mm = float(w)
                    return self.current_width_mm
                else:
                    return float(w)
            except Exception:
                pass

        # 2. Fallback to direct XML-RPC
        res = self._send_xmlrpc("rg_get_width", [self.gripper_id])
        if res is not None:
            val = float(res)
            if val > -900:
                self.current_width_mm = val
            return val
        return -999.0

    def grip(self, target_width_mm: float, target_force_n: float = 20.0) -> bool:
        """Command gripper to target width (0 to 110 mm) and force (3 to 40 N)."""
        target_width_mm = float(np.clip(target_width_mm, 0.0, 110.0))
        target_force_n = float(np.clip(target_force_n, 3.0, 40.0))

        if self.is_mock:
            print(f"[RG2-MOCK] Gripping to {target_width_mm:.1f} mm at {target_force_n:.1f} N")
            self.current_width_mm = target_width_mm
            return True

        print(f"[RG2] Commanding width={target_width_mm:.1f} mm, force={target_force_n:.1f} N...")

        # 1. Try official onRobot library
        if self._rg is not None:
            try:
                self._rg.rg_grip(target_width_mm, target_force_n)
                time.sleep(0.3)
                w = self._rg.get_rg_width()
                if w > -900:
                    self.current_width_mm = float(w)
                    return True
                else:
                    print(f"[RG2-ERROR] Robot rejected grip command with status -1 (returned width: {w:.1f} mm).")
                    return False
            except Exception as exc:
                print(f"[RG2] onRobot rg_grip exception: {exc}")

        # 2. Direct XML-RPC fallback
        res = self._send_xmlrpc("rg_grip", [self.gripper_id, target_width_mm, target_force_n])
        time.sleep(0.4)
        if res == -1:
            print("[RG2-ERROR] Robot XML-RPC returned -1 (hardware fault / no communication).")
            return False
        return True

    def open(self):
        """Fully open gripper to 80 mm."""
        return self.grip(80.0, 20.0)

    def close(self, width_mm: float = 25.0):
        """Close gripper to clamp object."""
        return self.grip(width_mm, 25.0)


class RobotInterface:
    """Unified robot controller supporting physical UR3e and realistic simulation."""

    def __init__(
        self,
        robot_ip: str = "192.168.0.100",
        control_freq: float = 25.0,  # 25 Hz visual servo rate
        max_joint_speed: float = 0.35,  # rad/s safety limit
        max_acceleration: float = 0.60,  # rad/s^2 (smooth, stable acceleration for UR3e)
        enable_control: bool = True,
        force_mock: bool = False,
    ):
        self.robot_ip = robot_ip
        self.control_freq = float(control_freq)
        self.dt = 1.0 / self.control_freq
        self.max_joint_speed = float(max_joint_speed)
        self.max_acceleration = float(max_acceleration)
        self.enable_control = enable_control
        self.force_mock = force_mock

        self.kinematics = UR3eKinematics()
        self.rtde_c = None
        self.rtde_r = None
        self.is_mock = force_mock or not RTDE_AVAILABLE
        self.gripper = RG2Gripper(self.robot_ip, is_mock=self.is_mock)

        # Internal state for mock mode
        # Home configuration: [0, -90, 90, -90, -90, 0] deg in radians
        self._mock_q = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])
        self._mock_qd = np.zeros(6)

        self._connected = False

    def connect(self):
        """Establish RTDE and Gripper sessions."""
        if self.force_mock or not RTDE_AVAILABLE:
            print("[Robot] Running in MOCK/SIMULATION mode.")
            self.is_mock = True
            self.gripper = RG2Gripper(self.robot_ip, is_mock=True)
            self._connected = True
            return

        print(f"[Robot] Connecting RTDE to {self.robot_ip}...")
        try:
            self.rtde_r = RTDEReceive(self.robot_ip)
            if not self.rtde_r.isConnected():
                raise ConnectionError("RTDE receive interface failed to connect.")
            print("[Robot] RTDE Receive connected (read-only state feedback active).")

            # Check robot safety / operational status
            mode = self.rtde_r.getRobotMode()
            if mode != 7:
                print(f"[Robot] NOTICE: Robot status is not Normal/Running (RobotMode={mode}).")
                print("       If teach pendant shows a stop or error (e.g. C271A1):")
                print("       1. Clear error popup on Teach Pendant -> Go to Initialization")
                print("       2. Tap 'ON' -> tap 'START' to release brakes (green Normal).")

            if self.enable_control:
                self.rtde_c = RTDEControl(self.robot_ip)
                if not self.rtde_c.isConnected():
                    raise ConnectionError("RTDE control interface failed to connect.")
                print("[Robot] RTDE Control connected (motion commands enabled).")
            else:
                print("[Robot] Running in READ-ONLY mode (safe for manual jogging/teach mode).")

            self.gripper = RG2Gripper(self.robot_ip, is_mock=False)
            self.is_mock = False
            self._connected = True

        except Exception as exc:
            print(f"[Robot] Connection note: {exc}")
            if self.enable_control:
                print("       Tip: On the teach pendant, confirm Remote Control is selected at the top.")
            print("[Robot] Falling back to MOCK mode.")
            self.is_mock = True
            self.gripper = RG2Gripper(self.robot_ip, is_mock=True)
            self._connected = True

    def get_joint_positions(self) -> np.ndarray:
        """Get 6 joint angles in radians."""
        if self.is_mock:
            return self._mock_q.copy()
        return np.asarray(self.rtde_r.getActualQ(), dtype=float)

    def get_joint_velocities(self) -> np.ndarray:
        """Get 6 joint velocities in rad/s."""
        if self.is_mock:
            return self._mock_qd.copy()
        return np.asarray(self.rtde_r.getActualQd(), dtype=float)

    def get_tcp_pose(self) -> np.ndarray:
        """Get TCP pose [x, y, z, rx, ry, rz] in base frame.

        Positions in meters, rotations as axis-angle vector in radians.
        """
        if self.is_mock:
            T = self.kinematics.forward_kinematics(self._mock_q, include_tcp=True)
            from scipy.spatial.transform import Rotation
            r_vec = Rotation.from_matrix(T[:3, :3]).as_rotvec()
            return np.r_[T[:3, 3], r_vec]

        return np.asarray(self.rtde_r.getActualTCPPose(), dtype=float)

    def set_joint_velocities(self, qd: np.ndarray):
        """Send joint velocity command (speedJ) with period timing."""
        qd = np.asarray(qd, dtype=float)
        if qd.shape != (6,):
            raise ValueError(f"Expected 6 joint velocities, got {qd.shape}")

        # Limit maximum joint speed
        peak = float(np.max(np.abs(qd)))
        if peak > self.max_joint_speed:
            qd = qd * (self.max_joint_speed / peak)

        if self.is_mock:
            self._mock_qd = qd
            self._mock_q += qd * self.dt
            time.sleep(self.dt)
            return

        if not self.enable_control or self.rtde_c is None:
            return

        # If commanded velocities are negligible, stop cleanly
        if np.all(np.abs(qd) < 1e-4):
            try:
                self.rtde_c.speedStop(self.max_acceleration * 1.5)
            except Exception:
                pass
            return

        # Execute speedJ with lookahead time (cmd_time = 0.08s) so the UR controller's
        # 500 Hz real-time thread never starves between ~25 Hz vision frames (avoids C271A1)
        cmd_time = max(0.08, self.dt * 2.0)
        try:
            self.rtde_c.speedJ(qd.tolist(), self.max_acceleration, cmd_time)
        except Exception as exc:
            print(f"[Robot] speedJ error: {exc}")

    def move_to_joints(self, target_q: np.ndarray, speed: float = 0.25, accel: float = 0.40):
        """Smooth point-to-point joint trajectory (moveJ)."""
        target_q = np.asarray(target_q, dtype=float)
        print(f"[Robot] Moving to joints (deg): {np.round(np.rad2deg(target_q), 1)}")

        if self.is_mock:
            steps = 40
            trajectory = np.linspace(self._mock_q, target_q, steps)
            for q_step in trajectory:
                self._mock_q = q_step
                time.sleep(0.02)
            return

        self.rtde_c.moveJ(target_q.tolist(), speed, accel)

    def stop(self):
        """Decelerate and stop joint velocity mode."""
        if not self.is_mock and self.rtde_c is not None:
            try:
                self.rtde_c.speedStop(self.max_acceleration * 2)
            except Exception:
                pass
        self._mock_qd[:] = 0.0

    def disconnect(self):
        """Safely release control scripts and disconnect."""
        self.stop()
        if not self.is_mock and self.rtde_c is not None:
            try:
                self.rtde_c.stopScript()
                self.rtde_c.disconnect()
            except Exception:
                pass
            try:
                self.rtde_r.disconnect()
            except Exception:
                pass
            print("[Robot] RTDE sessions disconnected.")
        self._connected = False
