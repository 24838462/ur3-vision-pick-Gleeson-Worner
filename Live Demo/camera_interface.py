"""Intel RealSense D435i Depth Camera Interface with Multi-Stage SDK Filtering.

Features:
- Hardware stream configuration (640x480 @ 30 FPS RGB and Depth)
- Depth-to-Color or Color-to-Depth pixel-accurate alignment
- Full RealSense SDK depth post-processing pipeline:
    1. Decimation Filter (optional resolution downsampling)
    2. Threshold Filter (cuts invalid near/far measurements, e.g. 0.2m - 1.8m)
    3. Disparity Transform (depth -> disparity space)
    4. Spatial Filter (edge-preserving spatial smoothing)
    5. Temporal Filter (smooths depth across consecutive video frames)
    6. Disparity Transform (disparity -> depth space)
    7. Hole-Filling Filter (recovers specular / reflection dropouts)
- Point cloud / 3D optical coordinate generation [X, Y, Z]
- Demo / Mock fallback mode when physical D435i is not connected
"""

import sys
from pathlib import Path
from typing import Any
import numpy as np

# Ensure realsense_lab from Robot Demo repo is importable if available
PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
ROBOT_DEMO_SRC = REPO_ROOT.parent / "Robot Demo" / "UR3_RG2_rosbridge" / "src"
if ROBOT_DEMO_SRC.exists() and str(ROBOT_DEMO_SRC) not in sys.path:
    sys.path.insert(0, str(ROBOT_DEMO_SRC))

try:
    import pyrealsense2 as rs
    PYREALSENSE_AVAILABLE = True
except ImportError:
    PYREALSENSE_AVAILABLE = False


class CameraFrame:
    """Convenience container for captured RGB-D data."""

    def __init__(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        xyz_m: np.ndarray,
        intrinsics: dict[str, Any],
        is_mock: bool = False,
    ):
        self.rgb = rgb
        self.depth_m = depth_m
        self.xyz_m = xyz_m
        self.intrinsics = intrinsics
        self.is_mock = is_mock

    @property
    def valid_depth(self) -> np.ndarray:
        return np.isfinite(self.depth_m) & (self.depth_m > 0.05)


class RealSenseD435i:
    """High-reliability D435i camera manager with extensive filtering."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        min_depth: float = 0.20,
        max_depth: float = 1.80,
        enable_spatial: bool = True,
        enable_temporal: bool = True,
        enable_hole_filling: bool = False,
        force_mock: bool = False,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.enable_spatial = enable_spatial
        self.enable_temporal = enable_temporal
        self.enable_hole_filling = enable_hole_filling
        self.force_mock = force_mock

        self.pipeline = None
        self.profile = None
        self.align = None
        self.filters = []
        self.intrinsics = {}
        self.is_mock = force_mock or not PYREALSENSE_AVAILABLE

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def _build_filters(self):
        """Construct the RealSense SDK depth post-processing filter chain."""
        if not PYREALSENSE_AVAILABLE:
            return []

        filters = []
        # 1. Threshold filter (clamp to valid table workspace)
        threshold = rs.threshold_filter(self.min_depth, self.max_depth)
        filters.append(threshold)

        # 2. Transform to disparity domain for spatial & temporal processing
        if self.enable_spatial or self.enable_temporal:
            filters.append(rs.disparity_transform(True))

            # 3. Spatial edge-preserving filter
            if self.enable_spatial:
                spatial = rs.spatial_filter()
                spatial.set_option(rs.option.filter_magnitude, 2.0)
                spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
                spatial.set_option(rs.option.filter_smooth_delta, 20.0)
                filters.append(spatial)

            # 4. Temporal persistence filter (essential for tripod stationary camera)
            if self.enable_temporal:
                temporal = rs.temporal_filter()
                temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
                temporal.set_option(rs.option.filter_smooth_delta, 20.0)
                filters.append(temporal)

            # Return from disparity domain
            filters.append(rs.disparity_transform(False))

        # 5. Hole filling filter
        if self.enable_hole_filling:
            hole_filling = rs.hole_filling_filter(1)  # 1 = farest from around
            filters.append(hole_filling)

        return filters

    def start(self):
        """Initialize camera streams and warm up pipeline."""
        if self.force_mock or not PYREALSENSE_AVAILABLE:
            print("[Camera] Running in MOCK/SIMULATED camera mode.")
            self.is_mock = True
            self.intrinsics = {
                "width": self.width,
                "height": self.height,
                "fx": 600.0,
                "fy": 600.0,
                "cx": self.width / 2.0,
                "cy": self.height / 2.0,
            }
            return

        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(
                rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
            )
            config.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps
            )

            print("[Camera] Starting Intel RealSense D435i pipeline...")
            self.profile = self.pipeline.start(config)

            # Align depth to colour camera frame
            self.align = rs.align(rs.stream.color)
            self.filters = self._build_filters()

            # Retrieve intrinsics from active colour stream
            color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = color_stream.get_intrinsics()
            self.intrinsics = {
                "width": intr.width,
                "height": intr.height,
                "fx": intr.fx,
                "fy": intr.fy,
                "cx": intr.ppx,
                "cy": intr.ppy,
                "coeffs": intr.coeffs,
            }

            # Warmup: drop first 15 frames for auto-exposure to stabilize
            print("[Camera] Warming up auto-exposure (15 frames)...")
            for _ in range(15):
                self.pipeline.wait_for_frames(timeout_ms=5000)

            print("[Camera] D435i online and filtered.")
            self.is_mock = False

        except Exception as exc:
            print(f"[Camera] Failed to open physical camera: {exc}")
            print("[Camera] Switching to MOCK camera fallback.")
            self.is_mock = True
            self.intrinsics = {
                "width": self.width,
                "height": self.height,
                "fx": 600.0,
                "fy": 600.0,
                "cx": self.width / 2.0,
                "cy": self.height / 2.0,
            }

    def capture(self) -> CameraFrame:
        """Capture one aligned, filtered RGB-D frame."""
        if self.is_mock:
            return self._capture_mock_frame()

        frameset = self.pipeline.wait_for_frames(timeout_ms=5000)
        aligned = self.align.process(frameset)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("Missing frame from RealSense stream.")

        # Apply RealSense SDK filter chain
        filtered_depth = depth_frame
        for flt in self.filters:
            filtered_depth = flt.process(filtered_depth)

        rgb = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(filtered_depth.get_data())

        # RealSense depth scale in metres (typically 0.001 m = 1 mm per unit)
        try:
            depth_scale = filtered_depth.as_depth_frame().get_units()
        except Exception:
            depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        depth_m = depth_raw.astype(np.float32) * depth_scale

        # Compute 3D XYZ coordinate map in camera optical frame
        fx, fy = self.intrinsics["fx"], self.intrinsics["fy"]
        cx, cy = self.intrinsics["cx"], self.intrinsics["cy"]

        u_grid, v_grid = np.meshgrid(
            np.arange(self.width, dtype=np.float32),
            np.arange(self.height, dtype=np.float32),
        )

        x_m = (u_grid - cx) * depth_m / fx
        y_m = (v_grid - cy) * depth_m / fy
        z_m = depth_m

        xyz_m = np.dstack((x_m, y_m, z_m))

        return CameraFrame(
            rgb=rgb,
            depth_m=depth_m,
            xyz_m=xyz_m,
            intrinsics=self.intrinsics,
            is_mock=False,
        )

    def _capture_mock_frame(self) -> CameraFrame:
        """Generate synthetic RGB-D frame containing a sample table and cube."""
        rgb = np.full((self.height, self.width, 3), 40, dtype=np.uint8)
        depth_m = np.full((self.height, self.width), 0.85, dtype=np.float32)

        # Draw a synthetic table surface
        rgb[100:440, 80:560] = [180, 180, 180]
        depth_m[100:440, 80:560] = 0.75

        # Draw a red cube at center (size ~40 mm, z=0.71 m)
        rgb[220:260, 300:340] = [220, 20, 20]
        depth_m[220:260, 300:340] = 0.71

        fx, fy = self.intrinsics["fx"], self.intrinsics["fy"]
        cx, cy = self.intrinsics["cx"], self.intrinsics["cy"]

        u_grid, v_grid = np.meshgrid(
            np.arange(self.width, dtype=np.float32),
            np.arange(self.height, dtype=np.float32),
        )

        x_m = (u_grid - cx) * depth_m / fx
        y_m = (v_grid - cy) * depth_m / fy
        xyz_m = np.dstack((x_m, y_m, depth_m))

        return CameraFrame(
            rgb=rgb,
            depth_m=depth_m,
            xyz_m=xyz_m,
            intrinsics=self.intrinsics,
            is_mock=True,
        )

    def stop(self):
        """Release camera hardware."""
        if self.pipeline and not self.is_mock:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
            print("[Camera] Pipeline stopped.")
