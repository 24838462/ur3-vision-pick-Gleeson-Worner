"""Unit test to verify top-face isolation and Kalman filtering math."""

import numpy as np
import cv2
from extrinsic_calibrator import ExtrinsicCalibrator
from target_kalman_filter import VisualTargetKalmanFilter

def test_calibrator_batch():
    calib = ExtrinsicCalibrator()
    # 1D point
    p1 = np.array([0.0, 0.0, 1.0])
    res1 = calib.transform_point(p1)
    assert res1.shape == (3,), f"Expected (3,), got {res1.shape}"

    # 2D point cloud
    p_cloud = np.random.uniform(0.5, 1.5, (100, 3))
    res_cloud = calib.transform_point(p_cloud)
    assert res_cloud.shape == (100, 3), f"Expected (100, 3), got {res_cloud.shape}"
    print("[PASS] ExtrinsicCalibrator handles both 1D and (N, 3) point clouds.")

def test_kalman_filter():
    kf = VisualTargetKalmanFilter(
        meas_noise_pos=0.003,
        meas_noise_yaw_deg=1.5,
        accel_noise_pos=0.06,
        accel_noise_yaw_deg=6.0,
    )

    # True stationary target
    true_pos = np.array([0.25, -0.15, 0.05])
    true_yaw = np.deg2rad(15.0)

    # Simulate 30 noisy frames
    for i in range(30):
        noise_p = np.random.normal(0, 0.004, 3)
        noise_y = np.random.normal(0, np.deg2rad(2.0))
        meas_p = true_pos + noise_p
        meas_y = (true_yaw + noise_y + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0
        filt_p, filt_y = kf.update(meas_p, meas_y, timestamp=i * 0.05)

    pos_err = np.linalg.norm(filt_p - true_pos)
    yaw_err_deg = np.rad2deg(abs((filt_y - true_yaw + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0))

    assert pos_err < 0.003, f"Filtered position error too large: {pos_err*1000:.2f} mm"
    assert yaw_err_deg < 1.0, f"Filtered yaw error too large: {yaw_err_deg:.2f} deg"
    assert kf.is_stationary, "Target should be recognized as stationary"
    print(f"[PASS] Kalman filter smoothed position error to {pos_err*1000:.2f} mm, yaw error to {yaw_err_deg:.2f} deg.")

def test_top_face_isolation():
    # Synthetic cube: 40 mm cube centered at (0.20, -0.10, 0.0) with top face at Z=0.040
    # Top face points: horizontal square
    top_x, top_y = np.meshgrid(np.linspace(0.18, 0.22, 10), np.linspace(-0.12, -0.08, 10))
    top_z = np.full_like(top_x, 0.040) + np.random.normal(0, 0.001, top_x.shape)
    top_pts = np.column_stack([top_x.ravel(), top_y.ravel(), top_z.ravel()])

    # Side face points: vertical descending to Z=0.0
    side_x = np.full(50, 0.18)
    side_y = np.linspace(-0.12, -0.08, 50)
    side_z = np.linspace(0.00, 0.038, 50)
    side_pts = np.column_stack([side_x, side_y, side_z])

    all_pts = np.vstack([top_pts, side_pts])

    # Apply top face isolation logic
    z_peak = float(np.percentile(all_pts[:, 2], 96))
    top_mask = (all_pts[:, 2] >= (z_peak - 0.005)) & (all_pts[:, 2] <= (z_peak + 0.004))
    cand_top = all_pts[top_mask]

    # Plane fit via SVD
    centered = cand_top - np.mean(cand_top, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal

    assert normal[2] > 0.95, f"Expected vertical normal, got {normal}"
    
    # Bounding rect
    rect = cv2.minAreaRect(np.ascontiguousarray(cand_top[:, :2], dtype=np.float32))
    (cx, cy), (w, l), _ = rect
    assert abs(cx - 0.20) < 0.003, f"Center X error: {cx}"
    assert abs(cy - (-0.10)) < 0.003, f"Center Y error: {cy}"
    print(f"[PASS] Top face isolation isolated pure top face: center=({cx:.3f}, {cy:.3f}), normal_z={normal[2]:.3f}.")

if __name__ == "__main__":
    test_calibrator_batch()
    test_kalman_filter()
    test_top_face_isolation()
    print("ALL TESTS PASSED!")
