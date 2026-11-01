"""Extrinsic Camera-to-Robot Calibration Module.

Provides:
- Rigid 3D-3D point set registration (Umeyama / Horn SVD algorithm)
- Transformation between RealSense Camera optical frame and UR3 Robot Base frame:
    p_base = R_base_cam @ p_cam + t_base_cam
- Interactive touch-off calibration wizard
- Save and load calibration matrices to/from JSON (camera_extrinsics.json)
"""

import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


CALIBRATION_FILE = Path(__file__).resolve().parent / "camera_extrinsics.json"


class ExtrinsicCalibrator:
    """Manages the spatial relationship between Tripod Camera and Robot Base."""

    def __init__(self, filepath: Path | str | None = None):
        self.filepath = Path(filepath) if filepath else CALIBRATION_FILE
        self.R = np.eye(3, dtype=float)
        self.t = np.zeros(3, dtype=float)
        self.offset = np.zeros(3, dtype=float)  # Fine-tuning trim offset in base frame [dx, dy, dz]
        self.calibrated = False
        self.rmse_mm = 0.0

        self.load()

    @property
    def transformation_matrix(self) -> np.ndarray:
        """4x4 homogeneous transformation matrix T_base_camera."""
        T = np.eye(4, dtype=float)
        T[:3, :3] = self.R
        T[:3, 3] = self.t + self.offset
        return T

    def transform_point(self, point_camera: np.ndarray) -> np.ndarray:
        """Transform 3D point(s) from camera optical coordinates to robot base coordinates.
        Supports both single point (3,) and point cloud array (N, 3).
        """
        pts = np.asarray(point_camera, dtype=float)
        t_eff = self.t + self.offset
        if pts.ndim == 1:
            return self.R @ pts + t_eff
        return pts @ self.R.T + t_eff

    def transform_vector(self, vector_camera: np.ndarray) -> np.ndarray:
        """Rotate vector(s) from camera optical frame to robot base frame."""
        vec = np.asarray(vector_camera, dtype=float)
        if vec.ndim == 1:
            return self.R @ vec
        return vec @ self.R.T

    def save(self, filepath: Path | str | None = None):
        """Save current calibration to JSON file."""
        target = Path(filepath) if filepath else self.filepath
        r_euler = Rotation.from_matrix(self.R).as_euler("xyz", degrees=True)
        data = {
            "R": self.R.tolist(),
            "t": self.t.tolist(),
            "offset_base_m": self.offset.tolist(),
            "euler_xyz_deg": r_euler.tolist(),
            "rmse_mm": float(self.rmse_mm),
            "calibrated": self.calibrated,
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"[Calibration] Saved extrinsic calibration to '{target}'")

    def load(self, filepath: Path | str | None = None) -> bool:
        """Load calibration from JSON file if available."""
        target = Path(filepath) if filepath else self.filepath
        if not target.exists():
            print(f"[Calibration] No calibration file at '{target}'. Using default nominal transform.")
            self._set_default_nominal()
            return False

        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.R = np.array(data["R"], dtype=float)
            self.t = np.array(data["t"], dtype=float)
            self.offset = np.array(data.get("offset_base_m", [0.0, 0.0, 0.0]), dtype=float)
            self.calibrated = bool(data.get("calibrated", True))
            self.rmse_mm = float(data.get("rmse_mm", 0.0))
            print(f"[Calibration] Successfully loaded extrinsic calibration from '{target}'")
            print(f"  t_base = {np.round(self.t, 4)} | Offset: {np.round(self.offset*1000.0, 1)} mm | RMSE: {self.rmse_mm:.2f} mm")
            return True
        except Exception as exc:
            print(f"[Calibration] Error loading '{target}': {exc}. Using default nominal.")
            self._set_default_nominal()
            return False

    def _set_default_nominal(self):
        """Default setup: Tripod camera mounted in front of robot looking down at ~45 deg."""
        # Camera optical frame: +X right, +Y down, +Z forward.
        # Robot base frame: +X forward, +Y left, +Z up.
        # Assume camera is at x=0.45m, y=-0.35m, z=0.55m relative to robot base.
        rot = Rotation.from_euler("xyz", [135.0, 0.0, 45.0], degrees=True)
        self.R = rot.as_matrix()
        self.t = np.array([0.45, -0.35, 0.55], dtype=float)
        self.calibrated = False

    @staticmethod
    def solve_umeyama(points_cam: np.ndarray, points_base: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute optimal rigid transformation (R, t) minimizing RMSE between point sets.

        p_base ~ R @ p_cam + t
        """
        p_c = np.asarray(points_cam, dtype=float)
        p_b = np.asarray(points_base, dtype=float)

        if p_c.shape != p_b.shape or p_c.shape[0] < 3:
            raise ValueError("Need at least 3 corresponding 3D points of equal shape (N, 3).")

        n = p_c.shape[0]
        centroid_c = np.mean(p_c, axis=0)
        centroid_b = np.mean(p_b, axis=0)

        # Center the points
        p_c_centered = p_c - centroid_c
        p_b_centered = p_b - centroid_b

        # Covariance matrix
        H = p_c_centered.T @ p_b_centered / n

        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure a right-handed coordinate system (det(R) == +1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_b - R @ centroid_c

        # Compute Root Mean Square Error (RMSE)
        transformed = (R @ p_c.T).T + t
        residuals = np.linalg.norm(transformed - p_b, axis=1)
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        return R, t, rmse

    def update_from_correspondences(self, points_cam: np.ndarray, points_base: np.ndarray) -> float:
        """Update R and t from corresponding point measurements and save to file."""
        R, t, rmse = self.solve_umeyama(points_cam, points_base)
        self.R = R
        self.t = t
        self.calibrated = True
        self.save()
        print(f"[Calibration] Calibration updated. Residual RMSE: {rmse * 1000.0:.2f} mm")
        return rmse
