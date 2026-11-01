"""High-Definition Real-time Estimation & Uncertainty Visualizer for UR3 MuJoCo Simulation.

Features:
1. High-resolution (1200 x 780) anti-aliased canvas with bold, high-contrast typography.
2. Trajectory comparison: Ground Truth vs Kalman Filter Estimate vs Noisy Measurements
   with shaded +/- 2-sigma (95.4% confidence) uncertainty envelope.
3. Estimation error tracking e(t) with dynamic theoretical covariance bounds.
4. Comprehensive 6-joint telemetry table with NIS consistency metrics.
5. Interactive controls: [1-6] Select joint, [V] Toggle Pos/Vel, [+/-] Sim Speed.
"""

from collections import deque
import cv2
import numpy as np


class EstimationPlotter:
    def __init__(self, joint_names, history_len: int = 300, width: int = 1200, height: int = 780):
        self.joint_names = list(joint_names)
        self.num_joints = len(self.joint_names)
        self.history_len = history_len
        self.width = width
        self.height = height

        # Active view configuration
        self.selected_joint_idx = 1  # Default to shoulder_lift
        self.view_mode = "pos"       # "pos" (deg) or "vel" (deg/s)
        self.sim_speed = 1.0         # Current simulation speed multiplier
        self.window_name = "State Estimation & Uncertainty Monitor"

        # Circular buffer for time-series history
        self.history = deque(maxlen=self.history_len)

        # Most recent sample cache
        self.latest = None

        # Statistics
        self.total_samples = 0
        self.in_bounds_samples = 0

    def add_sample(
        self,
        t: float,
        measured_q: np.ndarray,
        true_q: np.ndarray,
        true_qd: np.ndarray,
        q_est: np.ndarray,
        qd_est: np.ndarray,
        q_std: np.ndarray,
        qd_std: np.ndarray,
        nis: np.ndarray,
    ):
        """Append state observation at timestamp t."""
        measured_q = np.asarray(measured_q, dtype=float)
        true_q = np.asarray(true_q, dtype=float)
        true_qd = np.asarray(true_qd, dtype=float)
        q_est = np.asarray(q_est, dtype=float)
        qd_est = np.asarray(qd_est, dtype=float)
        q_std = np.asarray(q_std, dtype=float)
        qd_std = np.asarray(qd_std, dtype=float)
        nis = np.asarray(nis, dtype=float)

        q_err = q_est - true_q
        qd_err = qd_est - true_qd

        sample = {
            "t": float(t),
            "measured_q": measured_q,
            "true_q": true_q,
            "true_qd": true_qd,
            "q_est": q_est,
            "qd_est": qd_est,
            "q_err": q_err,
            "qd_err": qd_err,
            "q_std": q_std,
            "qd_std": qd_std,
            "nis": nis,
        }
        self.history.append(sample)
        self.latest = sample

        # Check bounds consistency for selected joint
        j = self.selected_joint_idx
        if self.view_mode == "pos":
            err = abs(q_err[j])
            bound = 2.0 * q_std[j]
        else:
            err = abs(qd_err[j])
            bound = 2.0 * qd_std[j]

        self.total_samples += 1
        if bound > 1e-6 and err <= bound:
            self.in_bounds_samples += 1

    def handle_key(self, key: int):
        """Handle interactive keyboard shortcuts."""
        if key == -1:
            return
        code = key & 0xFF

        # Number keys 1-6 to select joint directly
        if ord("1") <= code <= ord("6"):
            idx = code - ord("1")
            if idx < self.num_joints:
                self.selected_joint_idx = idx
                self.total_samples = 0
                self.in_bounds_samples = 0

        # 'j' or 'J' to cycle joint
        elif code in (ord("j"), ord("J")):
            self.selected_joint_idx = (self.selected_joint_idx + 1) % self.num_joints
            self.total_samples = 0
            self.in_bounds_samples = 0

        # 'v' or 'V' to toggle position vs velocity
        elif code in (ord("v"), ord("V")):
            self.view_mode = "vel" if self.view_mode == "pos" else "pos"
            self.total_samples = 0
            self.in_bounds_samples = 0

    def render(self) -> np.ndarray:
        """Render complete high-definition dashboard canvas."""
        canvas = np.full((self.height, self.width, 3), (22, 18, 15), dtype=np.uint8)

        j_idx = self.selected_joint_idx
        j_name = self.joint_names[j_idx] if j_idx < len(self.joint_names) else f"Joint {j_idx+1}"

        # 1. Header Bar
        self._render_header(canvas, j_name, j_idx)

        # If insufficient data, display waiting message
        if len(self.history) < 2 or self.latest is None:
            cv2.putText(
                canvas,
                "Awaiting Kalman Filter observation samples from simulation...",
                (self.width // 2 - 270, self.height // 2),
                cv2.FONT_HERSHEY_DUPLEX,
                0.65,
                (170, 180, 190),
                1,
                cv2.LINE_AA,
            )
            return canvas

        # 2. Main Live Graph 1: State Trajectory & 2-sigma Uncertainty Band
        # Top panel: y from 62 to 345 (height: 283)
        plot1_rect = (20, 62, self.width - 40, 275)
        self._render_trajectory_plot(canvas, plot1_rect, j_idx)

        # 3. Main Live Graph 2: Estimation Error with +/- 2-sigma Covariance Envelope
        # Middle panel: y from 348 to 565 (height: 217)
        plot2_rect = (20, 348, self.width - 40, 215)
        self._render_error_plot(canvas, plot2_rect, j_idx)

        # 4. Bottom Table: All 6 Joints Telemetry Summary
        # Bottom panel: y from 575 to 765 (height: 190)
        table_rect = (20, 575, self.width - 40, 190)
        self._render_joint_table(canvas, table_rect)

        return canvas

    def _render_header(self, canvas: np.ndarray, j_name: str, j_idx: int):
        """Top banner with joint indicator, simulation speed, and keyboard controls."""
        cv2.rectangle(canvas, (0, 0), (self.width, 54), (32, 26, 20), cv2.FILLED)
        cv2.line(canvas, (0, 54), (self.width, 54), (70, 60, 50), 2)

        # Title (Bold & Crisp)
        cv2.putText(
            canvas,
            "KALMAN FILTER OBSERVER: ESTIMATE & UNCERTAINTY vs GROUND TRUTH",
            (20, 24),
            cv2.FONT_HERSHEY_DUPLEX,
            0.56,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Active Joint Badge
        mode_str = "Position [deg]" if self.view_mode == "pos" else "Velocity [deg/s]"
        badge_text = f"ACTIVE: [{j_idx+1}] {j_name.upper()}  |  {mode_str}"
        cv2.rectangle(canvas, (20, 30), (440, 48), (65, 45, 25), cv2.FILLED)
        cv2.rectangle(canvas, (20, 30), (440, 48), (120, 85, 40), 1)
        cv2.putText(
            canvas,
            badge_text,
            (26, 44),
            cv2.FONT_HERSHEY_DUPLEX,
            0.42,
            (110, 240, 255),
            1,
            cv2.LINE_AA,
        )

        # Speed Multiplier Badge
        speed_str = "MAX (UNLIMITED)" if self.sim_speed == 0 else f"{self.sim_speed:.1f}x REALTIME"
        speed_text = f"Speed: {speed_str} [+/- keys]"
        cv2.rectangle(canvas, (455, 30), (710, 48), (30, 55, 35), cv2.FILLED)
        cv2.rectangle(canvas, (455, 30), (710, 48), (50, 100, 60), 1)
        cv2.putText(
            canvas,
            speed_text,
            (462, 44),
            cv2.FONT_HERSHEY_DUPLEX,
            0.40,
            (100, 255, 150),
            1,
            cv2.LINE_AA,
        )

        # Shortcuts Guide
        hints = "[1-6]: Select Joint  |  [V]: Pos/Vel  |  [+/-]: Sim Speed"
        cv2.putText(
            canvas,
            hints,
            (self.width - 460, 36),
            cv2.FONT_HERSHEY_DUPLEX,
            0.40,
            (190, 200, 210),
            1,
            cv2.LINE_AA,
        )

    def _render_trajectory_plot(self, canvas: np.ndarray, rect: tuple[int, int, int, int], j_idx: int):
        """Plot State Trajectory: Ground Truth vs Estimate vs Measurement with 2-sigma band."""
        rx, ry, rw, rh = rect

        # Draw panel background & border
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (16, 13, 10), cv2.FILLED)
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (60, 52, 44), 1)

        is_pos = (self.view_mode == "pos")

        # Extract data series
        times = [s["t"] for s in self.history]
        if is_pos:
            true_vals = np.rad2deg([s["true_q"][j_idx] for s in self.history])
            est_vals = np.rad2deg([s["q_est"][j_idx] for s in self.history])
            meas_vals = np.rad2deg([s["measured_q"][j_idx] for s in self.history])
            std_vals = np.rad2deg([s["q_std"][j_idx] for s in self.history])
            unit_str = "deg"
        else:
            true_vals = np.rad2deg([s["true_qd"][j_idx] for s in self.history])
            est_vals = np.rad2deg([s["qd_est"][j_idx] for s in self.history])
            meas_vals = None
            std_vals = np.rad2deg([s["qd_std"][j_idx] for s in self.history])
            unit_str = "deg/s"

        upper_2sigma = est_vals + 2.0 * std_vals
        lower_2sigma = est_vals - 2.0 * std_vals

        # Determine Y axis limits with padding
        all_vals = [true_vals, est_vals, upper_2sigma, lower_2sigma]
        if meas_vals is not None:
            all_vals.append(meas_vals)
        y_min = float(min(np.min(v) for v in all_vals))
        y_max = float(max(np.max(v) for v in all_vals))
        y_margin = max(0.5, (y_max - y_min) * 0.12)
        y_min -= y_margin
        y_max += y_margin
        if abs(y_max - y_min) < 1e-4:
            y_max += 1.0
            y_min -= 1.0

        plot_x0 = rx + 75
        plot_x1 = rx + rw - 20
        plot_y0 = ry + 32
        plot_y1 = ry + rh - 26

        pw = plot_x1 - plot_x0
        ph = plot_y1 - plot_y0
        n_pts = len(times)

        def map_pt(idx, val):
            px = int(plot_x0 + (idx / max(1, n_pts - 1)) * pw)
            py = int(plot_y1 - ((val - y_min) / (y_max - y_min)) * ph)
            return px, py

        # Grid lines & Y tick labels (High Contrast)
        for step in np.linspace(y_min, y_max, 5):
            _, gy = map_pt(0, step)
            if plot_y0 <= gy <= plot_y1:
                cv2.line(canvas, (plot_x0, gy), (plot_x1, gy), (40, 34, 28), 1)
                cv2.putText(
                    canvas,
                    f"{step:+.1f}",
                    (rx + 10, gy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (170, 170, 170),
                    1,
                    cv2.LINE_AA,
                )

        # 1. Shaded 2-Sigma Confidence Band
        pts_upper = [map_pt(i, upper_2sigma[i]) for i in range(n_pts)]
        pts_lower = [map_pt(i, lower_2sigma[i]) for i in range(n_pts)]
        poly_pts = np.array(pts_upper + pts_lower[::-1], dtype=np.int32)

        overlay = canvas.copy()
        cv2.fillPoly(overlay, [poly_pts], (20, 85, 130))  # Translucent amber/gold confidence band
        cv2.addWeighted(overlay, 0.40, canvas, 0.60, 0, canvas)

        # 2-sigma boundary lines
        cv2.polylines(canvas, [np.array(pts_upper)], False, (50, 130, 200), 1, cv2.LINE_AA)
        cv2.polylines(canvas, [np.array(pts_lower)], False, (50, 130, 200), 1, cv2.LINE_AA)

        # 2. Noisy Measurements (Cyan scatter points)
        if meas_vals is not None:
            step = max(1, n_pts // 90)
            for i in range(0, n_pts, step):
                m_px, m_py = map_pt(i, meas_vals[i])
                if plot_x0 <= m_px <= plot_x1 and plot_y0 <= m_py <= plot_y1:
                    cv2.circle(canvas, (m_px, m_py), 2, (255, 220, 60), -1)

        # 3. Ground Truth Trajectory (Thick Emerald Green)
        pts_true = np.array([map_pt(i, true_vals[i]) for i in range(n_pts)], dtype=np.int32)
        cv2.polylines(canvas, [pts_true], False, (70, 240, 90), 2, cv2.LINE_AA)

        # 4. Kalman Filter Estimate (Vibrant Orange/Amber)
        pts_est = np.array([map_pt(i, est_vals[i]) for i in range(n_pts)], dtype=np.int32)
        cv2.polylines(canvas, [pts_est], False, (0, 190, 255), 2, cv2.LINE_AA)

        # Panel Header / Legend
        header_text = f"TRAJECTORY COMPARISON [{unit_str}]"
        cv2.putText(canvas, header_text, (rx + 12, ry + 22), cv2.FONT_HERSHEY_DUPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)

        # Legend items
        lx = rx + 300
        # Truth
        cv2.line(canvas, (lx, ry + 16), (lx + 20, ry + 16), (70, 240, 90), 3)
        cv2.putText(canvas, "Ground Truth", (lx + 26, ry + 21), cv2.FONT_HERSHEY_DUPLEX, 0.42, (70, 240, 90), 1, cv2.LINE_AA)
        # Estimate
        lx += 140
        cv2.line(canvas, (lx, ry + 16), (lx + 20, ry + 16), (0, 190, 255), 3)
        cv2.putText(canvas, "KF Estimate", (lx + 26, ry + 21), cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 190, 255), 1, cv2.LINE_AA)
        # 2-Sigma Band
        lx += 140
        cv2.rectangle(canvas, (lx, ry + 10), (lx + 18, ry + 22), (20, 85, 130), cv2.FILLED)
        cv2.rectangle(canvas, (lx, ry + 10), (lx + 18, ry + 22), (50, 130, 200), 1)
        cv2.putText(canvas, "+/-2sigma Band (95.4%)", (lx + 24, ry + 21), cv2.FONT_HERSHEY_DUPLEX, 0.42, (100, 190, 250), 1, cv2.LINE_AA)
        # Noisy Measurement
        if meas_vals is not None:
            lx += 230
            cv2.circle(canvas, (lx + 8, ry + 16), 3, (255, 220, 60), -1)
            cv2.putText(canvas, "Noisy Encoders", (lx + 18, ry + 21), cv2.FONT_HERSHEY_DUPLEX, 0.42, (255, 220, 60), 1, cv2.LINE_AA)

        # Current live values readout (Top Right)
        cur_true = true_vals[-1]
        cur_est = est_vals[-1]
        cur_2std = 2.0 * std_vals[-1]
        readout = f"True: {cur_true:+.2f}{unit_str} | Est: {cur_est:+.2f}{unit_str} | 2sigma: +/-{cur_2std:.3f}{unit_str}"
        cv2.putText(canvas, readout, (plot_x1 - 420, ry + 21), cv2.FONT_HERSHEY_DUPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    def _render_error_plot(self, canvas: np.ndarray, rect: tuple[int, int, int, int], j_idx: int):
        """Plot Estimation Error e = x_est - x_true with theoretical +/- 2-sigma bounds."""
        rx, ry, rw, rh = rect

        # Draw panel background & border
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (16, 13, 10), cv2.FILLED)
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (60, 52, 44), 1)

        is_pos = (self.view_mode == "pos")
        times = [s["t"] for s in self.history]
        n_pts = len(times)

        if is_pos:
            err_vals = np.rad2deg([s["q_err"][j_idx] for s in self.history])
            std_vals = np.rad2deg([s["q_std"][j_idx] for s in self.history])
            unit_str = "deg"
        else:
            err_vals = np.rad2deg([s["qd_err"][j_idx] for s in self.history])
            std_vals = np.rad2deg([s["qd_std"][j_idx] for s in self.history])
            unit_str = "deg/s"

        upper_bound = 2.0 * std_vals
        lower_bound = -2.0 * std_vals

        max_bound = max(0.05, float(np.max(upper_bound)) * 1.35)
        max_err = max(0.05, float(np.max(np.abs(err_vals))) * 1.25)
        y_lim = max(max_bound, max_err)

        plot_x0 = rx + 75
        plot_x1 = rx + rw - 20
        plot_y0 = ry + 30
        plot_y1 = ry + rh - 24

        pw = plot_x1 - plot_x0
        ph = plot_y1 - plot_y0

        def map_err_pt(idx, val):
            px = int(plot_x0 + (idx / max(1, n_pts - 1)) * pw)
            py = int(plot_y0 + ph / 2.0 - (val / y_lim) * (ph / 2.0))
            return px, py

        # Zero reference line
        _, zero_y = map_err_pt(0, 0.0)
        cv2.line(canvas, (plot_x0, zero_y), (plot_x1, zero_y), (75, 65, 55), 1)
        cv2.putText(canvas, "0.0", (rx + 30, zero_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        # Upper and lower limit markers
        _, up_y = map_err_pt(0, y_lim * 0.75)
        _, dn_y = map_err_pt(0, -y_lim * 0.75)
        cv2.putText(canvas, f"+{y_lim*0.75:.2f}", (rx + 10, up_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"-{y_lim*0.75:.2f}", (rx + 10, dn_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1, cv2.LINE_AA)

        # 1. Shaded 2-sigma error bounds envelope
        pts_up = [map_err_pt(i, upper_bound[i]) for i in range(n_pts)]
        pts_dn = [map_err_pt(i, lower_bound[i]) for i in range(n_pts)]
        poly_pts = np.array(pts_up + pts_dn[::-1], dtype=np.int32)

        overlay = canvas.copy()
        cv2.fillPoly(overlay, [poly_pts], (35, 20, 55))  # Subdued envelope fill
        cv2.addWeighted(overlay, 0.40, canvas, 0.60, 0, canvas)

        # 2. +/- 2-Sigma Theoretical Covariance Boundary (Vivid Coral Red)
        cv2.polylines(canvas, [np.array(pts_up)], False, (70, 70, 255), 2, cv2.LINE_AA)
        cv2.polylines(canvas, [np.array(pts_dn)], False, (70, 70, 255), 2, cv2.LINE_AA)

        # 3. Estimation Error Line (Bright Pure White)
        pts_err = np.array([map_err_pt(i, err_vals[i]) for i in range(n_pts)], dtype=np.int32)
        cv2.polylines(canvas, [pts_err], False, (255, 255, 255), 2, cv2.LINE_AA)

        # Title & Consistency metric
        cur_err = err_vals[-1]
        cur_bound = upper_bound[-1]
        consistency_pct = (self.in_bounds_samples / max(1, self.total_samples)) * 100.0

        title_text = f"ESTIMATION ERROR  [e = x_est - x_true] vs +/-2sigma BOUND [{unit_str}]"
        cv2.putText(canvas, title_text, (rx + 12, ry + 22), cv2.FONT_HERSHEY_DUPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)

        # Legend
        lx = rx + 520
        cv2.line(canvas, (lx, ry + 16), (lx + 20, ry + 16), (255, 255, 255), 3)
        cv2.putText(canvas, "Error e(t)", (lx + 26, ry + 21), cv2.FONT_HERSHEY_DUPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        lx += 120
        cv2.line(canvas, (lx, ry + 16), (lx + 20, ry + 16), (70, 70, 255), 2)
        cv2.putText(canvas, "+/-2sigma Bound", (lx + 26, ry + 21), cv2.FONT_HERSHEY_DUPLEX, 0.42, (90, 90, 255), 1, cv2.LINE_AA)

        # Consistency badge
        status_color = (70, 240, 90) if consistency_pct >= 90.0 else (50, 190, 255)
        summary = f"Err: {cur_err:+.3f}{unit_str} | Limit: +/-{cur_bound:.3f}{unit_str} | Consistency: {consistency_pct:.1f}% in-bounds"
        cv2.putText(canvas, summary, (plot_x1 - 490, ry + 21), cv2.FONT_HERSHEY_DUPLEX, 0.42, status_color, 1, cv2.LINE_AA)

    def _render_joint_table(self, canvas: np.ndarray, rect: tuple[int, int, int, int]):
        """Large, razor-sharp summary table for all 6 robot joints."""
        rx, ry, rw, rh = rect
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (18, 14, 12), cv2.FILLED)
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (60, 50, 40), 1)

        # Generous column positions for 1200px width
        cols = [rx + 14, rx + 160, rx + 300, rx + 445, rx + 600, rx + 755, rx + 910, rx + 1045]
        headers = ["Joint", "True [deg]", "Est [deg]", "Error [deg]", "1sigma Std [deg]", "2sigma Limit [deg]", "NIS", "Filter Status"]

        # Header row (Bold background)
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + 26), (36, 28, 22), cv2.FILLED)
        for x, h in zip(cols, headers):
            cv2.putText(canvas, h, (x, ry + 18), cv2.FONT_HERSHEY_DUPLEX, 0.42, (200, 205, 215), 1, cv2.LINE_AA)

        if self.latest is None:
            return

        row_h = 24  # Large comfortable row height for crisp legibility
        for i in range(self.num_joints):
            row_y = ry + 30 + i * row_h
            is_selected = (i == self.selected_joint_idx)

            if is_selected:
                cv2.rectangle(canvas, (rx + 2, row_y - 3), (rx + rw - 2, row_y + row_h - 5), (55, 45, 25), cv2.FILLED)
                cv2.rectangle(canvas, (rx + 2, row_y - 3), (rx + rw - 2, row_y + row_h - 5), (120, 95, 45), 1)
                selector = ">"
                text_color = (110, 240, 255)
            else:
                selector = " "
                text_color = (220, 220, 220)

            name = self.joint_names[i]
            t_deg = float(np.rad2deg(self.latest["true_q"][i]))
            e_deg = float(np.rad2deg(self.latest["q_est"][i]))
            err_deg = float(np.rad2deg(self.latest["q_err"][i]))
            std_deg = float(np.rad2deg(self.latest["q_std"][i]))
            bound_deg = 2.0 * std_deg
            nis = float(self.latest["nis"][i]) if np.isfinite(self.latest["nis"][i]) else 0.0

            in_bounds = abs(err_deg) <= bound_deg
            status_str = "OPTIMAL" if in_bounds else "ADAPTING"
            status_color = (70, 230, 90) if in_bounds else (50, 90, 255)

            # Draw cells with clean font scale 0.44 and anti-aliasing
            cv2.putText(canvas, f"{selector} [{i+1}] {name}", (cols[0], row_y + 14), cv2.FONT_HERSHEY_DUPLEX, 0.42, text_color, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{t_deg:+.2f}", (cols[1], row_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{e_deg:+.2f}", (cols[2], row_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 210, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{err_deg:+.3f}", (cols[3], row_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{std_deg:.4f}", (cols[4], row_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 190, 190), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"+/-{bound_deg:.4f}", (cols[5], row_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (90, 150, 250), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{nis:.2f}", (cols[6], row_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (210, 210, 210), 1, cv2.LINE_AA)
            cv2.putText(canvas, status_str, (cols[7], row_y + 14), cv2.FONT_HERSHEY_DUPLEX, 0.42, status_color, 1, cv2.LINE_AA)
