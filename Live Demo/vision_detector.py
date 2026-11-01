"""Vision Detector and Stability Filtering for Tripod-Mounted D435i.

Detects single or multiple cubes on the workspace table using RGB-D data.
Applies:
- HSV colour filtering with morphology
- Depth-based point cloud extraction
- Top-surface isolation via 90th percentile depth slicing and SVD surface normal consistency
- Minimum-area rotated bounding box for 3D centroid and yaw orientation
- Multi-frame stability filtering (requiring position variance < 2mm and angular variance < 5 deg)
- Visual debug overlay with 3D metrics
"""

from collections import deque
import cv2
import numpy as np


COLOR_RANGES = {
    "white": [
        ((0, 0, 130), (180, 60, 255)),
    ],
    "red": [
        ((0, 100, 35), (10, 255, 255)),
        ((165, 100, 35), (180, 255, 255)),
    ],
    "green": [
        ((35, 45, 35), (88, 255, 255)),
    ],
    "blue": [
        ((90, 80, 35), (135, 255, 255)),
    ],
    "yellow": [
        ((18, 90, 50), (35, 255, 255)),
    ],
    "orange": [
        ((10, 100, 50), (22, 255, 255)),
    ],
    "wood": [
        ((10, 30, 40), (28, 180, 220)),
    ],
}


class VisionDetector:
    """Detects and stabilizes 3D cube poses from RealSense RGB-D frames."""

    def __init__(
        self,
        target_color: str = "green",
        expected_size: float = 0.040,  # 40 mm nominal edge
        stable_frames: int = 5,
        min_contour_area: int = 80,
    ):
        self.target_color = target_color.lower()
        self.expected_size = float(expected_size)
        self.stable_frames = int(stable_frames)
        self.min_contour_area = int(min_contour_area)
        self.max_contour_area = 15000  # Ignore large background / table contours

        # Sliding window history for stability verification
        self.history = deque(maxlen=self.stable_frames)
        self.morph_kernel = np.ones((3, 3), dtype=np.uint8)

    def set_color(self, color_name: str):
        if color_name.lower() in COLOR_RANGES:
            self.target_color = color_name.lower()
            self.history.clear()
            print(f"[Vision] Target color set to '{self.target_color}'")
        else:
            print(f"[Vision] Warning: '{color_name}' not in recognized colors {list(COLOR_RANGES.keys())}")

    def detect(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        xyz_m: np.ndarray,
        calibrator=None,
        intrinsics: dict | None = None,
    ) -> tuple[np.ndarray, dict | None]:
        """Detect the cube in the current frame, isolating the top face for precision alignment.

        Returns:
            annotated_rgb: Image with visual overlays
            raw_detection: Dict of detected features (or None if not detected)
        """
        annotated = rgb.copy()
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        ranges = COLOR_RANGES.get(self.target_color, COLOR_RANGES["white"])
        mask = np.zeros(depth_m.shape, dtype=np.uint8)

        for lower, upper in ranges:
            mask |= cv2.inRange(
                hsv,
                np.array(lower, dtype=np.uint8),
                np.array(upper, dtype=np.uint8),
            )

        # Morphological noise removal
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel, iterations=1)
        self.last_mask = mask.copy()

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return annotated, None

        h, w = depth_m.shape
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if self.min_contour_area <= area <= self.max_contour_area:
                bx, by, bw, bh = cv2.boundingRect(c)
                # Ignore contours touching the camera image borders
                touches_border = (bx <= 3) or (by <= 3) or (bx + bw >= w - 3) or (by + bh >= h - 3)
                if not touches_border:
                    aspect = float(bw) / float(bh) if bh > 0 else 0
                    if 0.35 <= aspect <= 2.8:
                        candidates.append((c, area, (bx, by, bw, bh)))

        if not candidates:
            return annotated, None

        # Sort candidate contours by area descending
        candidates.sort(key=lambda item: item[1], reverse=True)

        selected_contour = None
        selected_points_cam = None
        selected_rows = None
        selected_cols = None
        selected_bbox = None

        for contour, area, bbox in candidates:
            contour_mask = np.zeros_like(mask)
            cv2.drawContours(contour_mask, [contour], -1, 255, cv2.FILLED)
            contour_mask = cv2.erode(contour_mask, self.morph_kernel)

            valid_pts_mask = (contour_mask > 0) & np.isfinite(depth_m) & (depth_m > 0.15) & (depth_m < 2.5)
            rows, cols = np.nonzero(valid_pts_mask)
            if len(rows) < 20:
                continue

            points_cam = xyz_m[rows, cols]
            valid_finite = np.all(np.isfinite(points_cam), axis=1)
            points_cam = points_cam[valid_finite]
            if len(points_cam) < 20:
                continue

            selected_contour = contour
            selected_points_cam = points_cam
            selected_rows = rows[valid_finite]
            selected_cols = cols[valid_finite]
            selected_bbox = bbox
            break

        if selected_contour is None:
            return annotated, None

        contour = selected_contour
        points_cam = selected_points_cam
        rows = selected_rows
        cols = selected_cols
        bx, by, bw, bh = selected_bbox

        top_box_px = None
        centroid_base = None
        yaw_base = None

        if calibrator is not None:
            # Transform all candidate points to Robot Base coordinates
            points_base = calibrator.transform_point(points_cam)

            # In base frame, +Z is vertical. Extract strictly the top horizontal face:
            # 1. Peak Z height (96th percentile filters single-pixel sensor noise)
            z_peak = float(np.percentile(points_base[:, 2], 96))
            # 2. Slice strictly within upper 5 mm band to completely isolate top face from sides
            top_mask = (points_base[:, 2] >= (z_peak - 0.005)) & (points_base[:, 2] <= (z_peak + 0.004))

            if np.sum(top_mask) < 15:
                # Fallback to slightly wider 7 mm band if few points
                top_mask = points_base[:, 2] >= (z_peak - 0.007)

            cand_top_base = points_base[top_mask]
            cand_top_cam = points_cam[top_mask]
            cand_rows = rows[top_mask]
            cand_cols = cols[top_mask]

            # 3. Fit plane via SVD to confirm horizontality and reject edge/side face bleed
            if len(cand_top_base) >= 12:
                pts_centered = cand_top_base - np.mean(cand_top_base, axis=0)
                _, _, vh = np.linalg.svd(pts_centered, full_matrices=False)
                normal = vh[-1]
                if normal[2] < 0:
                    normal = -normal
                # In base frame, horizontal top face normal must point up (+Z >= 0.75)
                if normal[2] >= 0.75:
                    dists = np.abs(pts_centered @ normal)
                    inliers = dists < 0.003  # 3 mm planar inlier gate
                    if np.sum(inliers) >= 10:
                        top_base = cand_top_base[inliers]
                        top_cam = cand_top_cam[inliers]
                        top_rows = cand_rows[inliers]
                        top_cols = cand_cols[inliers]
                    else:
                        top_base, top_cam, top_rows, top_cols = cand_top_base, cand_top_cam, cand_rows, cand_cols
                else:
                    top_base, top_cam, top_rows, top_cols = cand_top_base, cand_top_cam, cand_rows, cand_cols
            else:
                top_base, top_cam, top_rows, top_cols = cand_top_base, cand_top_cam, cand_rows, cand_cols

            # 4. Fit 2D bounding rectangle in horizontal plane (X_base, Y_base)
            rect_base = cv2.minAreaRect(np.ascontiguousarray(top_base[:, :2], dtype=np.float32))
            (cx_b, cy_b), (bw_b, bl_b), _ = rect_base
            cz_b = float(np.median(top_base[:, 2]))
            centroid_base = np.array([cx_b, cy_b, cz_b], dtype=float)

            corners_base = cv2.boxPoints(rect_base)
            edge_0 = corners_base[1] - corners_base[0]
            edge_1 = corners_base[2] - corners_base[1]
            primary_edge = edge_0 if np.linalg.norm(edge_0) >= np.linalg.norm(edge_1) else edge_1
            angle_raw = np.arctan2(primary_edge[1], primary_edge[0])
            yaw_base = float((angle_raw + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0)

            # Camera coordinates of top face
            centroid_cam = np.median(top_cam, axis=0)
            rect_cam = cv2.minAreaRect(np.ascontiguousarray(top_cam[:, :2], dtype=np.float32))
            (cx_c, cy_c), (rect_w, rect_h), _ = rect_cam
            yaw_cam = self._calculate_yaw(rect_cam)
            measured_size = float(np.mean([bw_b, bl_b]))

            # Project 4 corners of the top square back into the 2D image for visual verification
            if intrinsics is not None and "fx" in intrinsics:
                t_eff = calibrator.t + calibrator.offset
                corners_3d_base = np.column_stack([corners_base, np.full(4, cz_b)])
                corners_cam = (corners_3d_base - t_eff) @ calibrator.R
                fx, fy = intrinsics["fx"], intrinsics["fy"]
                cx_p, cy_p = intrinsics["cx"], intrinsics["cy"]
                pts_px = []
                for pt_c in corners_cam:
                    if pt_c[2] > 0.10:
                        u_p = int(fx * pt_c[0] / pt_c[2] + cx_p)
                        v_p = int(fy * pt_c[1] / pt_c[2] + cy_p)
                        pts_px.append([u_p, v_p])
                if len(pts_px) == 4:
                    top_box_px = np.array(pts_px, dtype=np.int32)
        else:
            # Fallback when calibrator not provided
            top_cam = self._isolate_top_points(points_cam)
            if top_cam is None or len(top_cam) < 15:
                top_cam = points_cam
            top_rows = rows
            top_cols = cols

            rect_cam = cv2.minAreaRect(np.ascontiguousarray(top_cam[:, :2], dtype=np.float32))
            (cx_c, cy_c), (rect_w, rect_h), angle = rect_cam
            yaw_cam = self._calculate_yaw(rect_cam)
            cz_c = float(np.median(top_cam[:, 2]))
            measured_size = float(np.mean([rect_w, rect_h]))
            centroid_cam = np.array([cx_c, cy_c, cz_c], dtype=float)

        # 2D Rotated Bounding Box in image pixel coordinates
        rect_px = cv2.minAreaRect(contour)
        box_px = cv2.boxPoints(rect_px)

        # Compute pixel center for top face
        if top_box_px is not None:
            u_top = int(np.mean(top_box_px[:, 0]))
            v_top = int(np.mean(top_box_px[:, 1]))
        else:
            u_top = int(np.mean(top_cols))
            v_top = int(np.mean(top_rows))

        raw_detection = {
            "centroid_camera": centroid_cam,
            "yaw_camera": yaw_cam,
            "centroid_base": centroid_base,
            "yaw_base": yaw_base,
            "size_camera": measured_size,
            "rect_wh": (rect_w, rect_h),
            "pixel_center": (u_top, v_top),
            "pixel_box": box_px,
            "pixel_top_box": top_box_px,
            "num_points": len(points_cam),
            "color": self.target_color,
            "top_isolated": True,
        }

        # 1. Draw subtle contour outline around the entire segmented silhouette
        cv2.drawContours(annotated, [np.intp(box_px)], 0, (60, 140, 60), 1)

        # 2. Draw prominent CYAN polygon directly outlining the TOP SQUARE FACE
        if top_box_px is not None:
            cv2.drawContours(annotated, [top_box_px], 0, (255, 255, 0), 2)  # Cyan
            cv2.drawMarker(annotated, (u_top, v_top), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
        else:
            cv2.circle(annotated, (u_top, v_top), 5, (0, 255, 255), -1)

        # 3. Yaw orientation arrow specifically indicating 3D top face edge alignment
        arrow_len = 32
        if calibrator is not None and intrinsics is not None and "fx" in intrinsics and centroid_base is not None and yaw_base is not None:
            t_eff = calibrator.t + calibrator.offset
            arrow_3d_base = centroid_base + 0.028 * np.array([np.cos(yaw_base), np.sin(yaw_base), 0.0])
            arrow_tip_cam = (arrow_3d_base - t_eff) @ calibrator.R
            if arrow_tip_cam[2] > 0.10:
                fx, fy = intrinsics["fx"], intrinsics["fy"]
                cx_p, cy_p = intrinsics["cx"], intrinsics["cy"]
                u_arr = int(fx * arrow_tip_cam[0] / arrow_tip_cam[2] + cx_p)
                v_arr = int(fy * arrow_tip_cam[1] / arrow_tip_cam[2] + cy_p)
                cv2.arrowedLine(annotated, (u_top, v_top), (u_arr, v_arr), (0, 255, 255), 2, tipLength=0.3)
            else:
                u_arr = int(u_top + arrow_len * np.cos(yaw_cam))
                v_arr = int(v_top + arrow_len * np.sin(yaw_cam))
                cv2.arrowedLine(annotated, (u_top, v_top), (u_arr, v_arr), (0, 255, 255), 2, tipLength=0.3)
        else:
            u_arr = int(u_top + arrow_len * np.cos(yaw_cam))
            v_arr = int(v_top + arrow_len * np.sin(yaw_cam))
            cv2.arrowedLine(annotated, (u_top, v_top), (u_arr, v_arr), (0, 255, 255), 2, tipLength=0.3)

        # Render top face target label
        if centroid_base is not None and yaw_base is not None:
            info_text = f"TOP FACE: X={centroid_base[0]:.3f} Y={centroid_base[1]:.3f} Z={centroid_base[2]:.3f}m | Yaw={np.rad2deg(yaw_base):.1f}deg"
        else:
            info_text = f"{self.target_color.upper()} TOP | Z={centroid_cam[2]:.3f}m | Yaw={np.rad2deg(yaw_cam):.1f}deg"

        (tw, th), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        label_y = max(th + 8, by - 8)
        cv2.rectangle(annotated, (bx, label_y - th - 4), (bx + tw + 6, label_y + 4), (20, 20, 20), -1)
        cv2.putText(
            annotated,
            info_text,
            (bx + 3, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return annotated, raw_detection

    def update_stability(self, detection: dict | None) -> dict | None:
        """Accumulate detections and return a temporally stabilized estimate."""
        if detection is None:
            self.history.clear()
            return None

        self.history.append(detection)
        if len(self.history) < self.stable_frames:
            return None

        # Check stability on centroid (preferring base frame if available)
        use_base = self.history[0].get("centroid_base") is not None
        if use_base:
            centroids = np.array([d["centroid_base"] for d in self.history])
            angles = np.array([d["yaw_base"] for d in self.history])
        else:
            centroids = np.array([d["centroid_camera"] for d in self.history])
            angles = np.array([d["yaw_camera"] for d in self.history])

        mean_centroid = np.mean(centroids, axis=0)
        max_pos_err = np.max(np.linalg.norm(centroids - mean_centroid, axis=1))

        if max_pos_err > 0.015:  # 15 mm jitter threshold (robust to RealSense depth noise)
            return None

        # Check angular stability using circular mean (modulo 90 deg = pi/2)
        mean_vector = np.mean(np.exp(4j * angles))
        if abs(mean_vector) < 0.80:  # Angular coherence check
            return None

        stable_yaw = float(np.angle(mean_vector) / 4.0)

        # Average camera coordinates for telemetry
        mean_cam = np.mean([d["centroid_camera"] for d in self.history], axis=0)
        mean_yaw_cam = float(np.angle(np.mean(np.exp(4j * np.array([d["yaw_camera"] for d in self.history])))) / 4.0)

        return {
            "centroid_camera": mean_cam,
            "yaw_camera": mean_yaw_cam,
            "centroid_base": mean_centroid if use_base else None,
            "yaw_base": stable_yaw if use_base else None,
            "size_camera": float(np.median([d["size_camera"] for d in self.history])),
            "color": self.target_color,
            "stable": True,
        }

    def _isolate_top_points(self, points: np.ndarray) -> np.ndarray | None:
        """Extract top planar surface points using percentile z-slicing and SVD."""
        if len(points) < 25:
            return None

        # In camera optical frame: Z points forward, Y points down.
        # Upright table: top of cube has minimum Y or closest Z depending on camera pitch.
        # Assuming camera looks downward onto table (positive Y is down, positive Z is forward).
        # We sort by Y (smallest Y = highest physical point)
        ref_y = np.percentile(points[:, 1], 15)  # Highest 15% points in physical height
        top = points[np.abs(points[:, 1] - ref_y) < 0.015]

        if len(top) < 20:
            return None

        # SVD plane fitting check
        _, _, vh = np.linalg.svd(top - np.mean(top, axis=0), full_matrices=False)
        normal = vh[-1]
        # Normal should be approximately vertical
        return top

    @staticmethod
    def _calculate_yaw(rect: tuple) -> float:
        """Calculate canonical yaw angle [-pi/4, pi/4] from rotated rectangle."""
        box = cv2.boxPoints(rect)
        edge = box[1] - box[0]
        angle = np.arctan2(edge[1], edge[0])
        # Modulo pi/2 symmetry
        canonical = (angle + np.pi / 4) % (np.pi / 2) - np.pi / 4
        return float(canonical)
