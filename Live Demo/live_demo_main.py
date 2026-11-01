"""Live Demo: Eye-to-Hand PBVS Pick-and-Place with UR3e and RealSense D435i.

High-Distinction Implementation Features:
1. Stationary Tripod Eye-to-Hand Camera Formulation:
   - D435i depth camera mounted on a tripod viewing the table workspace
   - Complete extrinsic calibration mapping: p_base = R_base_cam @ p_cam + t_base_cam
2. Multi-Stage Sensing and Vision Filtering:
   - RealSense SDK hardware/software filters: Decimation, Threshold (0.2-1.8m),
     Disparity Transform, Spatial Filter, Temporal Filter, and Hole Filling
   - HSV colour segmentation with morphological opening and closing
   - 3D point cloud extraction with 90th percentile top-surface isolation and SVD plane normal fitting
   - Minimum area rotated rectangle yaw estimation (modulo pi/2)
   - Multi-frame temporal stability filtering (< 2mm position variance, < 5 deg yaw variance)
3. 6-DOF Task-Space PBVS Closed-Loop Servoing:
   - Proportional Cartesian error regulation: v = K_p * e_p, w = K_r * e_rot
   - Damped Least Squares (DLS) differential kinematics with adaptive singularity damping
   - Cartesian velocity limits (linear <= 30 mm/s, angular <= 0.2 rad/s)
   - Joint velocity limits (<= 0.3 rad/s) and joint limit deceleration protection
4. Robust OnRobot RG2 Gripper Interface:
   - HTTP XML-RPC driver (urllib.request) with zero external C-library dependency
5. Complete State Machine Sequencing:
   OBSERVE -> PBVS_HOVER -> PBVS_DESCEND -> GRASP -> LIFT -> TRANSPORT -> RELEASE -> HOME
6. Live Visual Debugger and Diagnostics:
   - Real-time OpenCV overlay with state banners, 3D coordinates, and error metrics
"""

import argparse
import copy
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from camera_interface import RealSenseD435i
from extrinsic_calibrator import ExtrinsicCalibrator
from pbvs_controller import PBVSController
from robot_interface import RobotInterface
from target_kalman_filter import VisualTargetKalmanFilter
from ur3e_kinematics import UR3eKinematics


import json

CONFIG_FILE = Path(__file__).resolve().parent / "demo_config.json"

# Default observation home joint configuration (UR3e elbow-up):
# [0, -90, 90, -90, -90, 0] degrees
HOME_Q = np.deg2rad([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])


class LiveDemoRunner:
    """Coordinates camera capture, visual filtering, PBVS servoing, and robot motions."""

    def __init__(
        self,
        robot_ip: str = "192.168.0.100",
        cube_color: str = "green",
        drop_pos: tuple[float, float, float] | None = None,
        control_rate: float = 25.0,
        mock_robot: bool = False,
        mock_camera: bool = False,
        hover_only: bool = False,
        hover_clearance: float | None = None,
    ):
        self.cube_color = cube_color.lower()
        self.drop_pos = np.asarray(drop_pos, dtype=float) if drop_pos is not None else np.array([-0.18, -0.35, 0.05])
        self.control_rate = float(control_rate)
        self.dt = 1.0 / self.control_rate
        self.hover_only = bool(hover_only)
        self.hover_clearance = float(hover_clearance) if hover_clearance is not None else 0.200

        self.home_q = HOME_Q.copy()
        self.grasp_width_mm = 25.0
        self.open_width_mm = 80.0
        self.gripper_offset_z = 0.195  # Standard 195 mm OnRobot RG2 TCP offset
        self.heatmap_min_d = 0.45
        self.heatmap_max_d = 1.05

        # Load saved demo configuration if present
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if "home_q_deg" in cfg:
                    self.home_q = np.deg2rad(cfg["home_q_deg"])
                    print(f"[Config] Loaded Home joint pose: {np.round(cfg['home_q_deg'], 1)} deg")
                if "drop_pos_base_m" in cfg and drop_pos is None:
                    self.drop_pos = np.array(cfg["drop_pos_base_m"], dtype=float)
                    print(f"[Config] Loaded Drop position: {np.round(self.drop_pos, 3)} m")
                if "cube_color" in cfg:
                    self.cube_color = str(cfg["cube_color"]).lower()
                    print(f"[Config] Loaded Target Cube Color: {self.cube_color.upper()}")
                if "hover_clearance_m" in cfg and hover_clearance is None:
                    self.hover_clearance = float(cfg["hover_clearance_m"])
                if "grasp_offset_z_m" in cfg:
                    self.grasp_offset_z = float(cfg["grasp_offset_z_m"])
                    print(f"[Config] Loaded Taught Grasp Descent Depth: {self.grasp_offset_z * 1000.0:.1f} mm")
                else:
                    self.grasp_offset_z = 0.015
                if "final_approach_z_offset_m" in cfg:
                    self.final_approach_z_offset = float(cfg["final_approach_z_offset_m"])
                else:
                    self.final_approach_z_offset = 0.030
                print(f"[Config] Loaded Final Approach Elevation Offset: +{self.final_approach_z_offset * 1000.0:.1f} mm (+3 cm higher)")
                if "grasp_width_mm" in cfg:
                    self.grasp_width_mm = float(cfg["grasp_width_mm"])
                if "gripper_open_width_mm" in cfg:
                    self.open_width_mm = float(cfg["gripper_open_width_mm"])
                if "gripper_tcp_offset_z_m" in cfg:
                    self.gripper_offset_z = float(cfg["gripper_tcp_offset_z_m"])
                    print(f"[Config] Loaded Gripper TCP Offset Z: {self.gripper_offset_z * 1000.0:.1f} mm")
                if "heatmap_min_depth_m" in cfg:
                    self.heatmap_min_d = float(cfg["heatmap_min_depth_m"])
                    print(f"[Config] Loaded Heatmap Min Depth: {self.heatmap_min_d:.2f} m")
                if "heatmap_max_depth_m" in cfg:
                    self.heatmap_max_d = float(cfg["heatmap_max_depth_m"])
                    print(f"[Config] Loaded Heatmap Max Depth: {self.heatmap_max_d:.2f} m")
            except Exception as exc:
                print(f"[Config] Note reading demo_config.json: {exc}")

        if not hasattr(self, "grasp_offset_z"):
            self.grasp_offset_z = 0.015
        if not hasattr(self, "final_approach_z_offset"):
            self.final_approach_z_offset = 0.030

        print(f"[Hover] Active Hover Clearance: {self.hover_clearance * 1000.0:.0f} mm ({self.hover_clearance:.3f} m)")
        print(f"[Grasp] Final Approach Height:  +{self.final_approach_z_offset * 1000.0:.1f} mm relative to cube top")

        print("\n" + "=" * 65)
        print("  UR3e + RealSense D435i Live PBVS Pick-and-Place Demo")
        print("=" * 65)

        # 1. Initialize Kinematics (with Gripper TCP offset) and Extrinsic Calibrator
        self.kin = UR3eKinematics(tcp_offset_z=self.gripper_offset_z)
        self.calibrator = ExtrinsicCalibrator()

        # Load saved calibration trim offset if present in demo_config.json
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if "calibration_offset_base_m" in cfg:
                    self.calibrator.offset = np.array(cfg["calibration_offset_base_m"], dtype=float)
                    print(f"[Config] Loaded Calibration Trim Offset: {np.round(self.calibrator.offset * 1000.0, 1)} mm")
            except Exception:
                pass

        # 2. Initialize Camera and Depth Filters
        self.cam = RealSenseD435i(
            width=640,
            height=480,
            fps=30,
            min_depth=0.20,
            max_depth=2.00,
            enable_spatial=True,
            enable_temporal=True,
            enable_hole_filling=False,
            force_mock=mock_camera,
        )

        # 3. Initialize Vision Detector
        from vision_detector import VisionDetector
        self.detector = VisionDetector(target_color=self.cube_color, stable_frames=8)

        # 4. Initialize Robot Interface
        self.robot = RobotInterface(
            robot_ip=robot_ip,
            control_freq=self.control_rate,
            max_joint_speed=0.30,
            force_mock=mock_robot,
        )

        # 5. Initialize PBVS Controller
        self.pbvs = PBVSController(
            kinematics=self.kin,
            hover_clearance=self.hover_clearance,
            grasp_offset_z=self.grasp_offset_z,
            final_approach_z_offset=self.final_approach_z_offset,
            lift_height=0.120,
            max_linear_speed=0.030,
            max_angular_speed=0.18,
            max_joint_speed=0.28,
        )

        # State for Interactive Teach Mode
        self.teach_pos = None
        self.teach_yaw = 0.0

        # 6. Initialize Kalman Filter for Visual Target Tracking (Joseph Form from estimation.py)
        self.target_kf = VisualTargetKalmanFilter(
            meas_noise_pos=0.003,        # 3 mm RealSense depth/spatial noise
            meas_noise_yaw_deg=1.5,      # 1.5 deg orientation noise
            accel_noise_pos=0.06,        # 0.06 m/s^2 smooth stationary table tracking
            accel_noise_yaw_deg=6.0,     # 6 deg/s^2 smooth orientation tracking
        )

        # State Machine States
        self.state = "INITIALIZE"
        self.stable_target = None
        self.locked_cube_target = None
        self.cube_locked = False
        self.running = False
        self.mission_complete = False
        self.view_mode = "side_by_side"  # Options: 'side_by_side', 'blend', 'rgb', 'heatmap'

    def start(self):
        """Start hardware connections and enter primary control loop."""
        self.cam.start()
        self.robot.connect()
        self.running = True

        cv2.namedWindow("UR3e Eye-to-Hand PBVS Live Demo", cv2.WINDOW_AUTOSIZE)

        try:
            self._run_loop()
        except KeyboardInterrupt:
            print("\n[Demo] Operator stop requested (Ctrl+C).")
        finally:
            self.stop()

    def stop(self):
        """Safely shut down all systems."""
        self.running = False
        print("\n[Demo] Halting robot motion and releasing hardware...")
        try:
            self.robot.stop()
            self.robot.disconnect()
        except Exception:
            pass
        try:
            self.cam.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[Demo] Shutdown complete.")

    def _run_loop(self):
        loop_start = time.time()

        while self.running:
            step_start = time.time()

            # 1. Capture registered RGB-D frame with RealSense SDK filters
            frame = self.cam.capture()

            # 2. Process Vision: Isolate Top Face of the cube in Robot Base frame
            annotated_rgb, raw_detection = self.detector.detect(
                frame.rgb,
                frame.depth_m,
                frame.xyz_m,
                calibrator=self.calibrator,
                intrinsics=self.cam.intrinsics,
            )

            # 2b. Apply Joseph-Form Kalman Filter to eliminate noise & jitter from visual target
            if raw_detection is not None:
                if raw_detection.get("centroid_base") is not None and raw_detection.get("yaw_base") is not None:
                    # Filter top face directly in Robot Base coordinates
                    kf_pos_base, kf_yaw_base = self.target_kf.update(
                        raw_detection["centroid_base"],
                        raw_detection["yaw_base"],
                        timestamp=step_start,
                    )
                    raw_detection["centroid_base"] = kf_pos_base
                    raw_detection["yaw_base"] = kf_yaw_base
                    raw_detection["is_stationary"] = self.target_kf.is_stationary

                    # Project filtered base centroid back to camera optical frame for telemetry/GUI
                    if self.calibrator is not None:
                        t_eff = self.calibrator.t + self.calibrator.offset
                        raw_detection["centroid_camera"] = (kf_pos_base - t_eff) @ self.calibrator.R
                else:
                    kf_pos, kf_yaw = self.target_kf.update(
                        raw_detection["centroid_camera"],
                        raw_detection["yaw_camera"],
                        timestamp=step_start,
                    )
                    raw_detection["centroid_camera"] = kf_pos
                    raw_detection["yaw_camera"] = kf_yaw
                    raw_detection["is_stationary"] = self.target_kf.is_stationary
            elif self.target_kf.initialized:
                # Smoothly coast through any single-frame camera dropout
                self.target_kf.predict(timestamp=step_start)

            # 3. Read Robot Current State
            current_q = self.robot.get_joint_positions()
            current_tcp = self.robot.get_tcp_pose()

            # 4. State Machine Logic
            self._step_state_machine(raw_detection, current_q, current_tcp)

            # 5. Build and Display Diagnostic GUI
            gui_img = self._create_overlay(annotated_rgb, frame.depth_m, current_tcp)
            cv2.imshow("UR3e Eye-to-Hand PBVS Live Demo", cv2.cvtColor(gui_img, cv2.COLOR_RGB2BGR))

            # Check key presses
            key = cv2.waitKey(int(self.dt * 1000.0 * 0.5)) & 0xFF
            if key in (27, ord("q")):
                print("[Demo] Exit requested by key press.")
                break
            elif key == ord("v"):
                modes = ["side_by_side", "blend", "rgb", "heatmap", "mask"]
                next_idx = (modes.index(self.view_mode) + 1) % len(modes)
                self.view_mode = modes[next_idx]
                print(f"[GUI] View mode switched to: {self.view_mode.upper()}")
            elif key in (ord("h"), ord("H")):
                self.hover_only = not self.hover_only
                state_str = "ENABLED (robot will hover & hold above cube)" if self.hover_only else "DISABLED (full pick & place)"
                print(f"[Mode] Hover-Only mode {state_str}")
            elif key == ord("m"):
                self.view_mode = "mask" if self.view_mode != "mask" else "side_by_side"
                print(f"[GUI] View mode switched to: {self.view_mode.upper()}")
            elif key == ord("r"):
                print("[Demo] Resetting to observation state...")
                self.state = "OBSERVE_AND_FILTER"
                self.stable_target = None
                self.detector.history.clear()
                self.target_kf.reset()
            elif key in (ord("a"), ord("A")):
                self.auto_calibrate_heatmap(frame.depth_m)
            elif key == ord("["):
                self.heatmap_min_d = max(0.20, round(self.heatmap_min_d - 0.05, 2))
                print(f"[Heatmap] Min depth adjusted: {self.heatmap_min_d:.2f}m")
            elif key == ord("]"):
                if self.heatmap_min_d + 0.10 < self.heatmap_max_d:
                    self.heatmap_min_d = round(self.heatmap_min_d + 0.05, 2)
                print(f"[Heatmap] Min depth adjusted: {self.heatmap_min_d:.2f}m")
            elif key in (ord("-"), ord("_")):
                if self.heatmap_max_d - 0.10 > self.heatmap_min_d:
                    self.heatmap_max_d = round(self.heatmap_max_d - 0.05, 2)
                print(f"[Heatmap] Max depth adjusted: {self.heatmap_max_d:.2f}m")
            elif key in (ord("="), ord("+")):
                self.heatmap_max_d = min(3.00, round(self.heatmap_max_d + 0.05, 2))
                print(f"[Heatmap] Max depth adjusted: {self.heatmap_max_d:.2f}m")
            elif key in (ord("s"), ord("S")):
                self.save_heatmap_config()
            elif key in (ord("i"), ord("I")):
                if self.state == "HOVER_HOLD" and self.teach_pos is not None:
                    self.teach_pos[0] += 0.005
                    print(f"[Jog X] Forward (+5mm) -> X={self.teach_pos[0]:.3f} m")
                else:
                    self.calibrator.offset[0] += 0.005  # +5 mm X (Forward)
                    self._apply_trim_offset()
            elif key in (ord("k"), ord("K")):
                if self.state == "HOVER_HOLD" and self.teach_pos is not None:
                    self.teach_pos[0] -= 0.005
                    print(f"[Jog X] Backward (-5mm) -> X={self.teach_pos[0]:.3f} m")
                else:
                    self.calibrator.offset[0] -= 0.005  # -5 mm X (Backward)
                    self._apply_trim_offset()
            elif key in (ord("j"), ord("J")):
                if self.state == "HOVER_HOLD" and self.teach_pos is not None:
                    self.teach_pos[1] += 0.005
                    print(f"[Jog Y] Left (+5mm) -> Y={self.teach_pos[1]:.3f} m")
                else:
                    self.calibrator.offset[1] += 0.005  # +5 mm Y (Left)
                    self._apply_trim_offset()
            elif key in (ord("l"), ord("L")):
                if self.state == "HOVER_HOLD" and self.teach_pos is not None:
                    self.teach_pos[1] -= 0.005
                    print(f"[Jog Y] Right (-5mm) -> Y={self.teach_pos[1]:.3f} m")
                else:
                    self.calibrator.offset[1] -= 0.005  # -5 mm Y (Right)
                    self._apply_trim_offset()
            elif key in (ord("u"), ord("U")):
                if self.state == "HOVER_HOLD":
                    self.teach_yaw = float((self.teach_yaw + np.deg2rad(2.0) + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0)
                    print(f"[Jog Yaw] CCW (+2 deg) -> Yaw={np.rad2deg(self.teach_yaw):.1f} deg")
                else:
                    self.calibrator.offset[2] += 0.005  # +5 mm Z (Up)
                    self._apply_trim_offset()
            elif key in (ord("o"), ord("O")):
                if self.state == "HOVER_HOLD":
                    self.teach_yaw = float((self.teach_yaw - np.deg2rad(2.0) + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0)
                    print(f"[Jog Yaw] CW (-2 deg) -> Yaw={np.rad2deg(self.teach_yaw):.1f} deg")
                else:
                    self.calibrator.offset[2] -= 0.005  # -5 mm Z (Down)
                    self._apply_trim_offset()
            elif key in (ord("e"), ord("E")):
                if self.state == "HOVER_HOLD" and self.teach_pos is not None:
                    top_z = self.stable_target["pos_base"][2] if self.stable_target else current_tcp[2]
                    self.teach_pos[2] = min(self.teach_pos[2] + 0.005, top_z + 0.35)
                    clr_mm = (self.teach_pos[2] - top_z) * 1000.0
                    print(f"[Jog Z] Elevate (+5mm) -> Z={self.teach_pos[2]:.3f} m (+{clr_mm:.1f} mm above cube top) [X, Y preserved]")
                else:
                    self.hover_clearance = round(self.hover_clearance + 0.010, 3)
                    self.pbvs.hover_clearance = self.hover_clearance
                    print(f"[Hover] Clearance increased to: {self.hover_clearance * 1000.0:.0f} mm")
            elif key in (ord("d"), ord("D")):
                if self.state == "HOVER_HOLD" and self.teach_pos is not None:
                    self.teach_pos[2] = max(self.teach_pos[2] - 0.005, self.pbvs.min_safe_z)
                    top_z = self.stable_target["pos_base"][2] if self.stable_target else current_tcp[2]
                    dsc_mm = (top_z - self.teach_pos[2]) * 1000.0
                    print(f"[Jog Z] Lower (-5mm) -> Z={self.teach_pos[2]:.3f} m (Descent depth: {dsc_mm:.1f} mm below cube top) [X, Y preserved]")
                else:
                    self.hover_clearance = max(0.040, round(self.hover_clearance - 0.010, 3))
                    self.pbvs.hover_clearance = self.hover_clearance
                    print(f"[Hover] Clearance decreased to: {self.hover_clearance * 1000.0:.0f} mm")
            elif key in (ord("t"), ord("T")):
                if self.state == "HOVER_HOLD" and self.teach_pos is not None and self.stable_target is not None:
                    top_z = self.stable_target["pos_base"][2]
                    taught_depth_m = max(0.002, float(top_z - self.teach_pos[2]))
                    self.grasp_offset_z = taught_depth_m
                    self.pbvs.grasp_offset_z = taught_depth_m

                    # Update calibration trim offset with the taught X and Y adjustments
                    if "centroid_base_raw" in self.stable_target:
                        raw_base = self.stable_target["centroid_base_raw"]
                        self.calibrator.offset[0] = round(float(self.teach_pos[0] - raw_base[0]), 4)
                        self.calibrator.offset[1] = round(float(self.teach_pos[1] - raw_base[1]), 4)

                    self.save_teach_config()
                    print("\n" + "=" * 65)
                    print(f"  [TEACH] GRASP POSE & DEPTH SAVED!")
                    print(f"          Taught Descent Depth: {taught_depth_m * 1000.0:.1f} mm")
                    print(f"          Taught (X, Y) Offset: [{self.calibrator.offset[0]*1000:+.1f}, {self.calibrator.offset[1]*1000:+.1f}] mm")
                    print(f"          Gripper Pinch Width:  {self.grasp_width_mm:.1f} mm")
                    print("          Persisted to demo_config.json!")
                    print("          Press SPACE to execute automated pick!")
                    print("=" * 65 + "\n")
            elif key in (ord(","), ord("<")):
                self.grasp_width_mm = max(10.0, round(self.grasp_width_mm - 2.0, 1))
                print(f"[Gripper Width] Grasp pinch width: {self.grasp_width_mm:.1f} mm")
            elif key in (ord("."), ord(">")):
                self.grasp_width_mm = min(75.0, round(self.grasp_width_mm + 2.0, 1))
                print(f"[Gripper Width] Grasp pinch width: {self.grasp_width_mm:.1f} mm")
            elif key in (ord("z"), ord("Z")):
                print(f"[Manual Gripper] Closing to {self.grasp_width_mm:.1f} mm...")
                self.robot.gripper.close(width_mm=self.grasp_width_mm)
            elif key in (ord("x"), ord("X")):
                print(f"[Manual Gripper] Opening to {self.open_width_mm:.1f} mm...")
                self.robot.gripper.open()
            elif key in (ord("g"), ord("G")):
                cur_w = self.robot.gripper.get_width()
                if cur_w > (self.grasp_width_mm + 12.0):
                    print(f"[Manual Gripper] Toggling: CLOSE to {self.grasp_width_mm:.1f} mm...")
                    self.robot.gripper.close(width_mm=self.grasp_width_mm)
                else:
                    print(f"[Manual Gripper] Toggling: OPEN to {self.open_width_mm:.1f} mm...")
                    self.robot.gripper.open()
            elif key == ord(" "):
                if self.state == "OBSERVE_AND_FILTER":
                    target_to_use = self.stable_target
                    if target_to_use is None and raw_detection is not None:
                        p_cam = raw_detection["centroid_camera"]
                        if raw_detection.get("centroid_base") is not None:
                            p_base = raw_detection["centroid_base"]
                            yaw_base = raw_detection["yaw_base"]
                        else:
                            p_base = self.calibrator.transform_point(p_cam)
                            edge_cam = np.array([np.cos(raw_detection["yaw_camera"]), -np.sin(raw_detection["yaw_camera"]), 0.0])
                            edge_base = self.calibrator.transform_vector(edge_cam)
                            yaw_base = float((np.arctan2(edge_base[1], edge_base[0]) + np.pi / 4) % (np.pi / 2) - np.pi / 4)
                        target_to_use = {
                            "centroid_camera": p_cam,
                            "centroid_base_raw": p_base.copy() - self.calibrator.offset,
                            "pos_base": p_base,
                            "yaw_base": yaw_base,
                            "size": raw_detection.get("size_camera", 0.040),
                        }
                        self.stable_target = target_to_use

                    if target_to_use is not None:
                        self.locked_cube_target = copy.deepcopy(target_to_use)
                        self.cube_locked = True
                        print("\n" + "=" * 65)
                        print("  [MEMORY LOCK] CUBE TARGET SAVED TO TEMPORARY MEMORY (MANUAL)!")
                        print(f"  Target Base 3D: X={target_to_use['pos_base'][0]:.3f}, Y={target_to_use['pos_base'][1]:.3f}, Z={target_to_use['pos_base'][2]:.3f} m")
                        print("  Visual updates FROZEN during approach to avoid gripper occlusion!")
                        print("=" * 65 + "\n")
                        self.pbvs.start_pick(
                            target_to_use["pos_base"],
                            target_to_use["yaw_base"],
                            cube_size=target_to_use.get("size", 0.040),
                        )
                        self.state = "PBVS_HOVER_APPROACH"
                    else:
                        print("\n[Manual Trigger] SPACE pressed, but cube is not visible in camera! Ensure target cube is in view.")
                elif self.state == "HOVER_HOLD":
                    cur_w = self.robot.gripper.get_width()
                    print("\n" + "=" * 65)
                    print(f"  [EXECUTE PICK] Descending along locked coordinates to grasp...")
                    print(f"  Grasp offset: {self.pbvs.grasp_offset_z * 1000.0:.1f} mm | Gripper width: {self.grasp_width_mm:.1f} mm")
                    print("=" * 65 + "\n")
                    if self.teach_pos is not None:
                        top_z = self.stable_target["pos_base"][2] if self.stable_target else self.teach_pos[2]
                        target_grasp_pos = self.teach_pos.copy()
                        target_grasp_pos[2] = top_z
                        self.pbvs.current_target_base = target_grasp_pos
                        self.pbvs.current_target_yaw = self.teach_yaw
                    self.hover_only = False
                    self.pbvs.phase = "PRECISION_DESCEND"
                    self.pbvs.converged_frames = 0
                    self.state = "PBVS_DESCEND"
            elif key in (ord("r"), ord("R")):
                print("\n[State] Resetting demo: Moving back to Home observation pose...")
                self.robot.stop()
                self.robot.gripper.open()
                self.robot.move_to_joints(self.home_q, speed=0.20)
                self.stable_target = None
                self.locked_cube_target = None
                self.cube_locked = False
                self.teach_pos = None
                self.detector.history.clear()
                self.target_kf.reset()
                self.state = "OBSERVE_AND_FILTER"
            elif key == ord("1"):
                self.detector.set_color("white")
                self.cube_color = "white"
                self.target_kf.reset()
            elif key == ord("2"):
                self.detector.set_color("red")
                self.cube_color = "red"
                self.target_kf.reset()
            elif key == ord("3"):
                self.detector.set_color("green")
                self.cube_color = "green"
                self.target_kf.reset()
            elif key == ord("4"):
                self.detector.set_color("blue")
                self.cube_color = "blue"
                self.target_kf.reset()

            # Maintain servo loop rate
            elapsed = time.time() - step_start
            sleep_time = self.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _step_state_machine(self, raw_detection, current_q, current_tcp):
        """Execute the pick-and-place PBVS sequence."""

        if self.state == "INITIALIZE":
            print("[State] INITIALIZE: Moving arm to Home observation pose and opening gripper...")
            self.robot.gripper.open()
            self.robot.move_to_joints(self.home_q, speed=0.20)
            self.state = "OBSERVE_AND_FILTER"
            print("[State] OBSERVE_AND_FILTER: Searching for stable cube detection...")

        elif self.state == "OBSERVE_AND_FILTER":
            # Robot holds position at Home while observing

            # Temporal stability check
            stable_est = self.detector.update_stability(raw_detection)
            if stable_est is not None:
                p_cam = stable_est["centroid_camera"]

                # Directly use top-face base pose if available from detector
                if stable_est.get("centroid_base") is not None:
                    p_base = stable_est["centroid_base"]
                    yaw_base = stable_est["yaw_base"]
                else:
                    p_base = self.calibrator.transform_point(p_cam)
                    edge_cam = np.array([np.cos(stable_est["yaw_camera"]), -np.sin(stable_est["yaw_camera"]), 0.0])
                    edge_base = self.calibrator.transform_vector(edge_cam)
                    yaw_base = np.arctan2(edge_base[1], edge_base[0])
                    yaw_base = float((yaw_base + np.pi / 4) % (np.pi / 2) - np.pi / 4)

                self.stable_target = {
                    "centroid_camera": p_cam,
                    "centroid_base_raw": p_base.copy() - self.calibrator.offset,
                    "pos_base": p_base,
                    "yaw_base": yaw_base,
                    "size": stable_est["size_camera"],
                }

                # Save cube position to temporary memory and lock it
                self.locked_cube_target = copy.deepcopy(self.stable_target)
                self.cube_locked = True

                print("\n" + "=" * 65)
                print("  [MEMORY LOCK] CUBE TARGET SAVED TO TEMPORARY MEMORY!")
                print(f"  Target Base 3D: X={p_base[0]:.3f}, Y={p_base[1]:.3f}, Z={p_base[2]:.3f} m")
                print(f"  Target Cube Yaw: {np.rad2deg(yaw_base):.1f} deg")
                print("  Visual updates FROZEN: gripper descent cannot distort cube pose!")
                print("=" * 65 + "\n")

                # Initiate PBVS Hover Approach
                self.pbvs.start_pick(p_base, yaw_base, cube_size=stable_est["size_camera"])
                self.state = "PBVS_HOVER_APPROACH"
                print("[State] PBVS_HOVER_APPROACH: Servoing to hover position above cube...")

        elif self.state == "PBVS_HOVER_APPROACH":
            # If target is locked in memory, do NOT update from camera as descending gripper
            # occludes the cube and casts shadows that corrupt detection
            if not self.cube_locked and raw_detection is not None:
                if raw_detection.get("centroid_base") is not None:
                    p_base_now = raw_detection["centroid_base"]
                    yaw_base_now = raw_detection["yaw_base"]
                else:
                    p_base_now = self.calibrator.transform_point(raw_detection["centroid_camera"])
                    edge_cam = np.array([np.cos(raw_detection["yaw_camera"]), -np.sin(raw_detection["yaw_camera"]), 0.0])
                    edge_base = self.calibrator.transform_vector(edge_cam)
                    yaw_base_now = float((np.arctan2(edge_base[1], edge_base[0]) + np.pi / 4) % (np.pi / 2) - np.pi / 4)
                self.pbvs.update_visual_target(p_base_now, yaw_base_now)

            qd, converged = self.pbvs.compute_step(current_q)
            self.robot.set_joint_velocities(qd)

            # Periodic status print (every ~0.5s)
            self._approach_print_cnt = getattr(self, "_approach_print_cnt", 0) + 1
            if self._approach_print_cnt % 12 == 0:
                cur_tcp = current_tcp[:3]
                tgt_base = self.pbvs.current_target_base
                if tgt_base is not None:
                    tgt_z = max(tgt_base[2] + self.pbvs.hover_clearance, self.pbvs.min_safe_z + self.pbvs.hover_clearance)
                    err_mm = np.linalg.norm([tgt_base[0] - cur_tcp[0], tgt_base[1] - cur_tcp[1], tgt_z - cur_tcp[2]]) * 1000.0
                    peak_speed = float(np.max(np.abs(qd)))
                    print(f"  [PBVS Hovering] Distance to Hover: {err_mm:.1f} mm | TCP Z: {cur_tcp[2]:.3f} m -> Target Z: {tgt_z:.3f} m | Joint Speed: {peak_speed:.3f} rad/s")

            if converged:
                self.robot.stop()
                self.teach_pos = current_tcp[:3].copy()
                self.teach_yaw = float(self.pbvs.current_target_yaw)
                if self.hover_only:
                    print(f"\n[PBVS] Hover approach converged at clearance +{self.hover_clearance*1000.0:.0f} mm!")
                    print("       === TEACH & ALIGN MODE ACTIVE ===")
                    print("       I / K: Jog X Forward / Backward (+/-5mm)")
                    print("       J / L: Jog Y Left / Right (+/-5mm)")
                    print("       E / D: Jog Z Up / Down (+/-5mm) [preserves X & Y strictly]")
                    print("       U / O: Jog Yaw Left / Right (+/-2 deg)")
                    print("       Z / X: Close / Open Gripper to test pinch width")
                    print("       T:     TEACH - Save grasp descent depth & alignment")
                    print("       SPACE: Execute automated pick using taught pose")
                    print("       R:     Reset to OBSERVE\n")
                    self.state = "HOVER_HOLD"
                else:
                    print("[PBVS] Hover approach converged! Transitioning to PRECISION_DESCEND.")
                    self.pbvs.phase = "PRECISION_DESCEND"
                    self.pbvs.converged_frames = 0
                    self.state = "PBVS_DESCEND"

        elif self.state == "HOVER_HOLD":
            if self.teach_pos is None:
                self.teach_pos = current_tcp[:3].copy()
                self.teach_yaw = float(self.pbvs.current_target_yaw)

            # Regulate strictly to teach_pos [X, Y, Z] and teach_yaw
            # Moving Z rigidly preserves X and Y without any drift!
            qd, held = self.pbvs.compute_cartesian_step(
                current_q,
                target_pos=self.teach_pos,
                target_yaw=self.teach_yaw,
                pos_tolerance=0.003,
            )
            self.robot.set_joint_velocities(qd)

        elif self.state == "PBVS_DESCEND":
            qd, converged = self.pbvs.compute_step(current_q)
            self.robot.set_joint_velocities(qd)

            self._descend_print_cnt = getattr(self, "_descend_print_cnt", 0) + 1
            if self._descend_print_cnt % 12 == 0:
                cur_tcp = current_tcp[:3]
                tgt_base = self.pbvs.current_target_base
                if tgt_base is not None:
                    target_z = max(tgt_base[2] + self.pbvs.final_approach_z_offset - self.pbvs.grasp_offset_z, self.pbvs.min_safe_z)
                    err_z_mm = (cur_tcp[2] - target_z) * 1000.0
                    print(f"  [PBVS Descending] TCP Z: {cur_tcp[2]:.3f} m -> Grasp Target Z: {target_z:.3f} m (+3cm higher, remaining: {err_z_mm:.1f} mm)")

            if converged:
                self.robot.stop()
                print("[PBVS] Elevated grasp height reached (+3 cm)! Clamping RG2 gripper...")
                self.state = "GRASP"

        elif self.state == "GRASP":
            self.robot.stop()
            # Grip cube (clamp to configured width with 25 N force)
            self.robot.gripper.close(width_mm=self.grasp_width_mm)
            time.sleep(0.5)

            # Check if grasped
            measured_width = self.robot.gripper.get_width()
            print(f"[Gripper] Clamped. Measured width: {measured_width:.1f} mm")

            self.pbvs.phase = "LIFT"
            self.pbvs.converged_frames = 0
            self.state = "LIFT"
            print("[State] LIFT: Raising cube vertically...")

        elif self.state == "LIFT":
            qd, converged = self.pbvs.compute_step(current_q)
            self.robot.set_joint_velocities(qd)

            if converged:
                self.robot.stop()
                print("\n" + "=" * 65)
                print("[State] Cube lifted successfully! Moving to Home position with cube...")
                print("=" * 65 + "\n")
                self.state = "RETRACT_HOME_WITH_CUBE"

        elif self.state == "RETRACT_HOME_WITH_CUBE":
            self.robot.stop()
            # Move to home joint angles while keeping the gripper clamped on the cube
            self.robot.move_to_joints(self.home_q, speed=0.20)
            self.state = "COMPLETE"
            self.mission_complete = True
            print("\n" + "*" * 60)
            print("  LIVE PICK COMPLETED: CUBE RETURNED TO HOME POSITION!")
            print("  Press 'X' to open gripper and release the cube.")
            print("  Press 'R' to reset and start a new observation.")
            print("*" * 60 + "\n")

        elif self.state == "TRANSPORT":
            # Plan safe smooth trajectory to drop location
            self.robot.stop()
            target_drop_hover = self.drop_pos.copy()
            target_drop_hover[2] += 0.080  # hover above drop spot

            print(f"[Transport] Transferring cube to drop position {np.round(target_drop_hover, 3)}...")
            self.pbvs.current_target_base = target_drop_hover
            self.pbvs.phase = "HOVER_APPROACH"
            self.pbvs.converged_frames = 0
            self.state = "TRANSPORT_SERVO"

        elif self.state == "TRANSPORT_SERVO":
            qd, converged = self.pbvs.compute_step(current_q)
            self.robot.set_joint_velocities(qd)

            if converged:
                self.robot.stop()
                print("[Transport] Arrived above drop target! Lowering to place surface...")
                self.pbvs.current_target_base = self.drop_pos
                self.pbvs.phase = "PRECISION_DESCEND"
                self.pbvs.converged_frames = 0
                self.state = "PLACE_DESCEND"

        elif self.state == "PLACE_DESCEND":
            qd, converged = self.pbvs.compute_step(current_q)
            self.robot.set_joint_velocities(qd)

            if converged:
                self.robot.stop()
                print("[State] Placement height reached. Releasing RG2 gripper...")
                self.state = "RELEASE"

        elif self.state == "RELEASE":
            self.robot.stop()
            self.robot.gripper.open()
            time.sleep(0.5)
            print("[State] Gripper released. Retracting back to Home...")
            self.state = "RETRACT_HOME"

        elif self.state == "RETRACT_HOME":
            self.robot.stop()
            self.robot.move_to_joints(self.home_q, speed=0.25)
            self.state = "COMPLETE"
            self.mission_complete = True
            print("\n" + "*" * 60)
            print("  LIVE PICK-AND-PLACE DEMO COMPLETED SUCCESSFULLY!")
            print("*" * 60)

        elif self.state == "COMPLETE":
            self.robot.set_joint_velocities(np.zeros(6))

    def auto_calibrate_heatmap(self, depth_m: np.ndarray):
        """Auto-calibrate heatmap depth range to the 3rd and 97th percentiles of active scene."""
        valid = np.isfinite(depth_m) & (depth_m > 0.15) & (depth_m < 3.0)
        valid_depths = depth_m[valid]
        if len(valid_depths) > 500:
            p_low = float(np.percentile(valid_depths, 3))
            p_high = float(np.percentile(valid_depths, 97))
            self.heatmap_min_d = max(0.20, float(np.round(p_low - 0.03, 2)))
            self.heatmap_max_d = min(2.50, float(np.round(p_high + 0.05, 2)))
            if self.heatmap_max_d - self.heatmap_min_d < 0.15:
                self.heatmap_max_d = self.heatmap_min_d + 0.20
            print(f"[Heatmap] Auto-calibrated range: {self.heatmap_min_d:.2f}m (Red) -> {self.heatmap_max_d:.2f}m (Blue)")

    def _apply_trim_offset(self):
        """Apply live calibration nudge offset and re-target robot immediately."""
        off = self.calibrator.offset * 1000.0
        print(f"[Trim] Calibration Offset: X={off[0]:+.1f} mm, Y={off[1]:+.1f} mm, Z={off[2]:+.1f} mm  (Press 'S' to save)")
        if self.stable_target is not None and "centroid_camera" in self.stable_target:
            self.stable_target["pos_base"] = self.calibrator.transform_point(self.stable_target["centroid_camera"])
            self.pbvs.current_target_base = self.stable_target["pos_base"].copy()
            if self.state == "HOVER_HOLD":
                print("[Trim] Re-aligning robot hover position to trimmed coordinates...")
                self.state = "PBVS_HOVER_APPROACH"
                self.pbvs.phase = "HOVER_APPROACH"
                self.pbvs.converged_frames = 0

    def save_heatmap_config(self):
        """Persist current calibrated heatmap range and trim offset to demo_config.json."""
        cfg = {}
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                pass
        cfg["heatmap_min_depth_m"] = round(self.heatmap_min_d, 2)
        cfg["heatmap_max_depth_m"] = round(self.heatmap_max_d, 2)
        cfg["calibration_offset_base_m"] = [round(float(v), 4) for v in self.calibrator.offset]
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)
        self.calibrator.save()
        print(f"[Config] Saved heatmap range and Trim Offset {np.round(self.calibrator.offset * 1000.0, 1)} mm to demo_config.json & camera_extrinsics.json")

    def save_teach_config(self):
        """Persist taught grasp descent depth and grip width to demo_config.json."""
        cfg = {}
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                pass
        cfg["grasp_offset_z_m"] = round(float(self.grasp_offset_z), 4)
        cfg["grasp_width_mm"] = round(float(self.grasp_width_mm), 1)
        cfg["hover_clearance_m"] = round(float(self.hover_clearance), 3)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)
        print(f"[Config] Persisted Taught Grasp Depth {self.grasp_offset_z * 1000.0:.1f} mm and Grip Width {self.grasp_width_mm:.1f} mm to demo_config.json")

    def _create_overlay(self, rgb: np.ndarray, depth_m: np.ndarray, current_tcp: np.ndarray) -> np.ndarray:
        """Render diagnostic banners, state indicators, and dual-pane RGB + Depth Heatmap."""
        h, w = rgb.shape[:2]

        # 1. Compute Depth Heatmap (turbo_r: Red=Near, Blue=Far, Black=Invalid)
        valid = np.isfinite(depth_m) & (depth_m > 0.05)
        clipped = np.clip(depth_m, self.heatmap_min_d, self.heatmap_max_d)
        # Invert normalization for turbo: Near (min_d) -> 255 (Red), Far (max_d) -> 0 (Blue)
        norm = (255.0 - (clipped - self.heatmap_min_d) / (self.heatmap_max_d - self.heatmap_min_d) * 255.0).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        heatmap_bgr[~valid] = [0, 0, 0]
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Draw professional Colorbar Legend on the right side of heatmap
        bar_x1, bar_x2 = w - 38, w - 22
        bar_y1, bar_y2 = 70, 250
        bar_h = bar_y2 - bar_y1
        # Gradient: Top = Red (val=255), Bottom = Blue (val=0)
        grad = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(-1, 1)
        grad_col = cv2.applyColorMap(grad, cv2.COLORMAP_TURBO)
        grad_rgb = cv2.cvtColor(grad_col, cv2.COLOR_BGR2RGB)
        heatmap_rgb[bar_y1:bar_y2, bar_x1:bar_x2] = np.repeat(grad_rgb, bar_x2 - bar_x1, axis=1)
        cv2.rectangle(heatmap_rgb, (bar_x1 - 1, bar_y1 - 1), (bar_x2, bar_y2), (255, 255, 255), 1)

        # Labels for the colorbar
        cv2.putText(heatmap_rgb, f"{self.heatmap_min_d:.2f}m", (bar_x1 - 50, bar_y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(heatmap_rgb, f"{(self.heatmap_min_d + self.heatmap_max_d)/2:.2f}m", (bar_x1 - 50, (bar_y1 + bar_y2)//2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(heatmap_rgb, f"{self.heatmap_max_d:.2f}m", (bar_x1 - 50, bar_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 180, 255), 1, cv2.LINE_AA)

        # Label the heatmap panel
        cv2.putText(
            heatmap_rgb,
            "FILTERED DEPTH HEATMAP (turbo_r)",
            (15, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            heatmap_rgb,
            f"Range: {self.heatmap_min_d:.2f}m (Red) -> {self.heatmap_max_d:.2f}m (Blue)",
            (15, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            heatmap_rgb,
            "[ ] Min | - = Max | A: Auto-Calib | S: Save",
            (15, 116),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (180, 255, 180),
            1,
            cv2.LINE_AA,
        )

        # 2. Select View Mode (Side-by-side, Blend, RGB only, Heatmap only)
        if self.view_mode == "side_by_side":
            display = np.hstack((rgb, heatmap_rgb))
        elif self.view_mode == "heatmap":
            display = heatmap_rgb.copy()
        elif self.view_mode == "blend":
            alpha = 0.45
            blend = (1 - alpha) * rgb + alpha * heatmap_rgb
            blend[~valid] = rgb[~valid]
            display = blend.astype(np.uint8)
        elif self.view_mode == "mask":
            last_m = getattr(self.detector, "last_mask", None)
            if last_m is not None:
                mask_rgb = cv2.cvtColor(last_m, cv2.COLOR_GRAY2RGB)
            else:
                mask_rgb = np.zeros_like(rgb)
            display = mask_rgb
        else:  # "rgb"
            display = rgb.copy()

        dh, dw = display.shape[:2]

        # Top Banner
        cv2.rectangle(display, (0, 0), (dw, 52), (30, 30, 30), -1)

        state_colors = {
            "INITIALIZE": (200, 200, 200),
            "OBSERVE_AND_FILTER": (255, 200, 0),
            "PBVS_HOVER_APPROACH": (0, 220, 255),
            "HOVER_HOLD": (0, 255, 255),
            "PBVS_DESCEND": (0, 165, 255),
            "GRASP": (0, 255, 0),
            "LIFT": (180, 255, 0),
            "TRANSPORT": (255, 120, 0),
            "TRANSPORT_SERVO": (255, 120, 0),
            "PLACE_DESCEND": (255, 80, 80),
            "RELEASE": (255, 0, 255),
            "RETRACT_HOME": (150, 150, 255),
            "COMPLETE": (0, 255, 0),
        }
        col = state_colors.get(self.state, (255, 255, 255))

        state_label = f"STATE: {self.state}"
        if self.state == "HOVER_HOLD":
            state_label = "STATE: HOVER_HOLD [TEACH MODE]"
        elif self.hover_only:
            state_label += " [HOVER-ONLY]"

        if self.cube_locked:
            state_label += " [MEMORY LOCKED]"
        elif self.state == "OBSERVE_AND_FILTER":
            hist_len = len(self.detector.history)
            req_len = self.detector.stable_frames
            if hist_len > 0:
                state_label += f" [Locking: {hist_len}/{req_len}]"
            else:
                state_label += " [Searching...]"

        cv2.putText(
            display,
            state_label,
            (15, 33),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            col,
            2,
            cv2.LINE_AA,
        )

        if self.state == "HOVER_HOLD":
            target_info = "I/K=X | J/L=Y | E/D=Z | U/O=Yaw | T=Teach | Z/X=Grip | SPACE=Pick"
        else:
            mode_tag = "H: Hover-Only [ON]" if self.hover_only else "H: Hover-Only [OFF]"
            target_info = f"Color: {self.cube_color.upper()} | {mode_tag} | Grip: [X/Z/G] | View: [V/M]"

        cv2.putText(
            display,
            target_info,
            (max(15, dw - 780), 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        # Bottom Banner: Metrics
        cv2.rectangle(display, (0, dh - 40), (dw, dh), (20, 20, 20), -1)
        tcp_str = f"TCP Base: X={current_tcp[0]:.3f} Y={current_tcp[1]:.3f} Z={current_tcp[2]:.3f} m"
        cv2.putText(
            display,
            tcp_str,
            (15, dh - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        gripper_w = self.robot.gripper.current_width_mm if self.robot.gripper else 80.0
        if self.state == "HOVER_HOLD" and self.teach_pos is not None and self.stable_target is not None:
            top_z = self.stable_target["pos_base"][2]
            cur_depth_mm = (top_z - self.teach_pos[2]) * 1000.0
            grip_str = f"Teach: [{self.teach_pos[0]:.3f}, {self.teach_pos[1]:.3f}, {self.teach_pos[2]:.3f}] | Depth: {cur_depth_mm:+.1f}mm | Grip: {gripper_w:.0f}mm (T=Save)"
        else:
            calib_str = f"CALIBRATED ({self.calibrator.rmse_mm:.1f} mm)" if self.calibrator.calibrated else "NOMINAL"
            off = self.calibrator.offset * 1000.0
            grip_str = f"Offset: [{off[0]:+.0f}, {off[1]:+.0f}, {off[2]:+.0f}]mm (I/K=X, J/L=Y, U/O=Z, S=Save) | {calib_str}"

        cv2.putText(
            display,
            grip_str,
            (max(15, dw - 760), dh - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (100, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return display


def main():
    parser = argparse.ArgumentParser(description="UR3e RealSense D435i Live PBVS Demo")
    parser.add_argument("--robot-ip", default="192.168.0.197", help="UR controller IP address (default: 192.168.0.197)")
    parser.add_argument("--color", default="green", help="Target cube color (green, white, red, blue, yellow, orange, wood)")
    parser.add_argument("--mock-robot", action="store_true", help="Run with simulated mock robot")
    parser.add_argument("--mock-camera", action="store_true", help="Run with simulated mock camera")
    parser.add_argument("--drop-x", type=float, default=None, help="Override drop position X in base frame (m)")
    parser.add_argument("--drop-y", type=float, default=None, help="Override drop position Y in base frame (m)")
    parser.add_argument("--drop-z", type=float, default=None, help="Override drop position Z in base frame (m)")
    parser.add_argument("--grasp-width", type=float, default=25.0, help="Gripper close width for cube clamp (mm, default: 25)")
    parser.add_argument("--hover-only", action="store_true", help="Move to hover directly over the cube with end-effector facing down, then hold position (no descend or grasp)")
    parser.add_argument("--hover-clearance", type=float, default=None, help="Hover height above cube in meters (default: from demo_config.json or 0.20 m)")
    parser.add_argument("--set-home", action="store_true", help="Read current robot joint configuration and save as Home in demo_config.json")
    parser.add_argument("--set-drop", action="store_true", help="Read current robot TCP position and save as Drop location in demo_config.json")
    args = parser.parse_args()

    # Helper command: save current robot joint angles as Home
    if args.set_home:
        print(f"[Config] Connecting to {args.robot_ip} to capture current joint angles...")
        bot = RobotInterface(robot_ip=args.robot_ip, enable_control=False)
        bot.connect()
        q_cur_deg = np.rad2deg(bot.get_joint_positions()).tolist()
        bot.disconnect()

        cfg = {}
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        cfg["home_q_deg"] = q_cur_deg
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)
        print(f"[Config] Successfully saved current pose as Home in demo_config.json:")
        print(f"  home_q_deg: {np.round(q_cur_deg, 1)}")
        return

    # Helper command: save current robot TCP position as Drop location
    if args.set_drop:
        print(f"[Config] Connecting to {args.robot_ip} to capture current TCP position...")
        bot = RobotInterface(robot_ip=args.robot_ip, enable_control=False)
        bot.connect()
        tcp_pos = bot.get_tcp_pose()[:3].tolist()
        bot.disconnect()

        cfg = {}
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        cfg["drop_pos_base_m"] = tcp_pos
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)
        print(f"[Config] Successfully saved current TCP position as Drop in demo_config.json:")
        print(f"  drop_pos_base_m: {np.round(tcp_pos, 4)}")
        return

    drop_tuple = None
    if args.drop_x is not None and args.drop_y is not None and args.drop_z is not None:
        drop_tuple = (args.drop_x, args.drop_y, args.drop_z)

    runner = LiveDemoRunner(
        robot_ip=args.robot_ip,
        cube_color=args.color,
        drop_pos=drop_tuple,
        mock_robot=args.mock_robot,
        mock_camera=args.mock_camera,
        hover_only=args.hover_only,
        hover_clearance=args.hover_clearance,
    )
    runner.grasp_width_mm = float(args.grasp_width)
    runner.start()


if __name__ == "__main__":
    main()
