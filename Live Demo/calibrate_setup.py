"""Kinematic Joint-Pose Extrinsic Calibration Tool for UR3e & RealSense D435i.

Computes the 6-DOF transformation between the Robot Base (Absolute World Frame)
and the Tripod Camera Optical Frame using multiple distinct robot joint configurations:
    p_base_world = R_base_cam @ p_cam + t_base_cam

At each calibration position:
1. Physical joint encoders [q1, q2, q3, q4, q5, q6] are sampled directly from RTDE.
2. Full forward kinematics T_base_tool(q) is computed in the Robot Base World Frame.
3. The Tool Head / Fingertip position is matched with the Camera Optical 3D point.
4. Rigid 3D-3D point set registration (Umeyama SVD algorithm) solves the optimal
   R_base_cam and t_base_cam minimizing world-frame residual errors.
5. All recorded joint vectors, world poses, camera points, and residual errors are
   saved to camera_extrinsics.json for complete traceability.
"""

import argparse
import json
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
from extrinsic_calibrator import ExtrinsicCalibrator, CALIBRATION_FILE
from robot_interface import RobotInterface
from ur3e_kinematics import UR3eKinematics


# Preset safe joint configurations (degrees) spanning the UR3e camera-facing workspace
SUGGESTED_SAFE_POSES_DEG = [
    [  0.0, -85.0,  75.0, -80.0, -90.0,   0.0],  # Pose 1: Center elevated
    [-15.0, -75.0,  65.0, -80.0, -90.0, -15.0],  # Pose 2: Left tilt
    [ 15.0, -75.0,  65.0, -80.0, -90.0,  15.0],  # Pose 3: Right tilt
    [ -5.0, -95.0,  95.0, -90.0, -90.0,   0.0],  # Pose 4: Center lower
    [ 10.0, -90.0,  85.0, -85.0, -90.0,  10.0],  # Pose 5: Forward center
]


clicked_pixel = None


def on_mouse(event, x, y, flags, param):
    global clicked_pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pixel = (x, y)
        print(f"[Calib] Clicked pixel: (u={x}, v={y})")


def run_joint_pose_calibration(robot_ip: str, mock_robot: bool = False):
    global clicked_pixel
    print("\n" + "=" * 70)
    print("  UR3e KINEMATIC JOINT-POSE EXTRINSIC CALIBRATION")
    print("  Robot Base = Absolute World Frame (Origin)")
    print("=" * 70)
    print("Instructions:")
    print("1. In Local mode, hold the Freedrive button on the teach pendant.")
    print("2. Move the robot into 4 or 5 diverse joint configurations in camera view.")
    print("   (Vary shoulder, elbow, and wrist angles to span the 3D workspace).")
    print("3. For each pose: press SPACE in the camera window, then click the tool head.")
    print("4. Press 'C' to compute the World-to-Camera transformation and save.")
    print("5. Press 'Q' or ESC to exit.")
    print("=" * 70)

    gripper_offset_z = 0.195
    cfg_file = Path(__file__).resolve().parent / "demo_config.json"
    if cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if "gripper_tcp_offset_z_m" in cfg:
                gripper_offset_z = float(cfg["gripper_tcp_offset_z_m"])
                print(f"[Calib] Loaded Gripper TCP Offset Z: {gripper_offset_z * 1000.0:.1f} mm from demo_config.json")
        except Exception:
            pass

    kinematics = UR3eKinematics(tcp_offset_z=gripper_offset_z)
    robot = RobotInterface(robot_ip=robot_ip, enable_control=False, force_mock=mock_robot)
    robot.connect()

    calibrator = ExtrinsicCalibrator()
    cam = RealSenseD435i(enable_spatial=True, enable_temporal=True)
    cam.start()

    cv2.namedWindow("UR3e Kinematic Calibration", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("UR3e Kinematic Calibration", on_mouse)

    records = []
    pts_camera = []
    pts_base_world = []

    try:
        while True:
            frame = cam.capture()
            rgb_display = frame.rgb.copy()

            # Read live physical robot joint configuration & TCP pose
            q_rad = robot.get_joint_positions()
            q_deg = np.rad2deg(q_rad)
            tcp_world = robot.get_tcp_pose()

            # Compute forward kinematics independently to verify against robot controller
            fk_transform = kinematics.forward_kinematics(q_rad, include_tcp=True)
            p_fk_world = fk_transform[:3, 3]

            # Draw previously recorded points on live RGB feed
            for r in records:
                px, py = r["pixel"]
                cv2.circle(rgb_display, (px, py), 6, (0, 255, 0), -1)
                cv2.putText(rgb_display, f"#{r['pose_id']}", (px + 8, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

            # Compute depth heatmap (Turbo_r colormap: Red=Near, Blue=Far)
            valid_d = np.isfinite(frame.depth_m) & (frame.depth_m > 0.05)
            calib_min_d = 0.40
            calib_max_d = 1.10
            clipped_d = np.clip(frame.depth_m, calib_min_d, calib_max_d)
            norm_d = (255.0 - (clipped_d - calib_min_d) / (calib_max_d - calib_min_d) * 255.0).astype(np.uint8)
            heatmap_bgr = cv2.applyColorMap(norm_d, cv2.COLORMAP_TURBO)
            heatmap_bgr[~valid_d] = [0, 0, 0]
            heatmap_display = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

            # Draw status on RGB
            status_text = f"Poses: {len(records)} recorded (min 4). SPACE=Sample, Click Tool Head, C=Solve, Q=Exit"
            cv2.putText(rgb_display, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(rgb_display, "Tip: Sample both high/low Z and top/bottom of view for tilt accuracy", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 255, 180), 1, cv2.LINE_AA)
            cv2.putText(heatmap_display, f"DEPTH HEATMAP ({calib_min_d:.2f}m Red -> {calib_max_d:.2f}m Blue)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

            # Bottom HUD for live joint angles and world TCP
            dual = np.hstack((rgb_display, heatmap_display))
            dh, dw = dual.shape[:2]

            # Banner overlay for live joint telemetry
            cv2.rectangle(dual, (0, dh - 50), (dw, dh), (20, 20, 20), -1)
            q_str = f"Joints [deg]: [{q_deg[0]:.1f}, {q_deg[1]:.1f}, {q_deg[2]:.1f}, {q_deg[3]:.1f}, {q_deg[4]:.1f}, {q_deg[5]:.1f}]"
            cv2.putText(dual, q_str, (10, dh - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1, cv2.LINE_AA)

            tcp_str = f"World Base TCP: X={tcp_world[0]:.4f}m, Y={tcp_world[1]:.4f}m, Z={tcp_world[2]:.4f}m (FK check diff: {np.linalg.norm(p_fk_world - tcp_world[:3])*1000:.1f} mm)"
            cv2.putText(dual, tcp_str, (10, dh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 100), 1, cv2.LINE_AA)

            cv2.imshow("UR3e Kinematic Calibration", cv2.cvtColor(dual, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(30) & 0xFF

            if key in (27, ord("q")):
                print("[Calib] Calibration cancelled.")
                break

            elif key == ord(" "):
                # Sample current joint configuration and world pose
                pose_idx = len(records) + 1
                print(f"\n" + "-" * 60)
                print(f"SAMPLING JOINT POSE #{pose_idx}")
                print(f"  Physical Joint Angles [deg]: {np.round(q_deg, 2)}")
                print(f"  World Base TCP [m]:          {np.round(tcp_world[:3], 4)}")
                print(f"  Tool Orientation [rad]:      {np.round(tcp_world[3:], 3)}")
                print(">> Please LEFT-CLICK on the center of the gripper fingertip in the camera image...")

                clicked_pixel = None
                while clicked_pixel is None:
                    k2 = cv2.waitKey(20) & 0xFF
                    if k2 in (27, ord("q")):
                        break

                if clicked_pixel is not None:
                    cx_raw, cy_raw = clicked_pixel
                    # Map click to RGB pane coordinates (640x480)
                    u = cx_raw if cx_raw < frame.rgb.shape[1] else cx_raw - frame.rgb.shape[1]
                    v = cy_raw

                    # Robust 7x7 patch around (u, v) with median filtering to reject edge noise & depth dropouts
                    u_min, u_max = max(0, u - 3), min(frame.xyz_m.shape[1], u + 4)
                    v_min, v_max = max(0, v - 3), min(frame.xyz_m.shape[0], v + 4)
                    patch = frame.xyz_m[v_min:v_max, u_min:u_max].reshape(-1, 3)
                    valid_patch = np.all(np.isfinite(patch), axis=1) & (patch[:, 2] > 0.15) & (patch[:, 2] < 2.5)

                    if np.sum(valid_patch) >= 5:
                        p_cam = np.median(patch[valid_patch], axis=0)
                    else:
                        p_cam = frame.xyz_m[v, u]

                    if not np.all(np.isfinite(p_cam)) or p_cam[2] <= 0.15:
                        print(f"Warning: Invalid depth measurement at (u={u}, v={v}). Try clicking again.")
                    else:
                        p_base = tcp_world[:3].copy()
                        pts_camera.append(p_cam)
                        pts_base_world.append(p_base)

                        record_entry = {
                            "pose_id": pose_idx,
                            "joint_positions_deg": q_deg.tolist(),
                            "joint_positions_rad": q_rad.tolist(),
                            "world_tcp_base_m": p_base.tolist(),
                            "camera_point_m": p_cam.tolist(),
                            "pixel": [int(u), int(v)],
                        }
                        records.append(record_entry)
                        print(f"Successfully recorded Pose #{pose_idx}:")
                        print(f"  Camera 3D: {np.round(p_cam, 3)} m  <==>  Base World 3D: {np.round(p_base, 3)} m")

            elif key == ord("c"):
                if len(records) < 4:
                    print(f"\n[Warning] Need at least 4 distinct joint poses to ensure full 3D spatial coverage (currently have {len(records)}).")
                else:
                    print("\n" + "=" * 60)
                    print("COMPUTING WORLD-TO-CAMERA EXTRINSIC TRANSFORMATION")
                    print("=" * 60)

                    p_cam_arr = np.array(pts_camera)
                    p_base_arr = np.array(pts_base_world)

                    # Solve optimal rigid body transform via Umeyama SVD
                    R, t, rmse = ExtrinsicCalibrator.solve_umeyama(p_cam_arr, p_base_arr)

                    # Compute individual residuals for each recorded joint configuration
                    transformed_cam = (R @ p_cam_arr.T).T + t
                    residuals = np.linalg.norm(transformed_cam - p_base_arr, axis=1) * 1000.0  # mm

                    print("\nJoint-Pose Calibration Verification Report:")
                    print("-" * 65)
                    print(f"{'Pose':<6} {'World TCP (X, Y, Z) [m]':<26} {'Transformed Cam [m]':<26} {'Error (mm)':<10}")
                    print("-" * 65)
                    for i, r in enumerate(records):
                        r["residual_error_mm"] = float(residuals[i])
                        w_str = f"[{p_base_arr[i,0]:.3f}, {p_base_arr[i,1]:.3f}, {p_base_arr[i,2]:.3f}]"
                        c_str = f"[{transformed_cam[i,0]:.3f}, {transformed_cam[i,1]:.3f}, {transformed_cam[i,2]:.3f}]"
                        print(f"#{r['pose_id']:<5} {w_str:<26} {c_str:<26} {residuals[i]:.2f}")

                    print("-" * 65)
                    print(f"TOTAL ROOT MEAN SQUARE ERROR (RMSE): {rmse * 1000.0:.2f} mm")
                    print("=" * 60)

                    # Save full report and matrix to camera_extrinsics.json
                    r_euler = Rotation.from_matrix(R).as_euler("xyz", degrees=True)
                    calib_payload = {
                        "R": R.tolist(),
                        "t": t.tolist(),
                        "euler_xyz_deg": r_euler.tolist(),
                        "rmse_mm": float(rmse * 1000.0),
                        "num_poses": len(records),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "joint_pose_records": records,
                        "calibrated": True,
                    }

                    with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
                        json.dump(calib_payload, f, indent=4)

                    print(f"[Calibration] Saved complete joint-pose calibration to:\n  '{CALIBRATION_FILE}'")
                    print("=" * 60)
                    break

    finally:
        cam.stop()
        robot.disconnect()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="UR3e Kinematic Joint-Pose Calibration")
    parser.add_argument("--robot-ip", default="192.168.0.197", help="UR controller IP address")
    parser.add_argument("--mock-robot", action="store_true", help="Use simulated mock robot")
    parser.add_argument("--reset-default", action="store_true", help="Reset to nominal default calibration")
    args = parser.parse_args()

    if args.reset_default:
        calib = ExtrinsicCalibrator()
        calib._set_default_nominal()
        calib.save()
        print("[Calib] Reset camera_extrinsics.json to default nominal pose.")
        return

    run_joint_pose_calibration(args.robot_ip, mock_robot=args.mock_robot)


if __name__ == "__main__":
    main()
