# UR3e + RealSense D435i Eye-to-Hand PBVS Live Demo

Autonomous 6-DOF Position-Based Visual Servoing (PBVS) pick-and-place system using an **Intel RealSense D435i depth camera** mounted on a tripod (eye-to-hand configuration) and a **Universal Robots UR3e** with an **OnRobot RG2** gripper.

---

## 1. High Distinction Engineering Highlights

### A. Eye-to-Hand PBVS Formulation
* **Tripod Extrinsic Geometry**: The camera is stationary and views the robot workspace. Visual target measurements in Camera Optical Coordinates are mapped to the Robot Base Frame via calibrated homogeneous transformation:
  $$\mathbf{p}_{base} = \mathbf{R}_{base}^{cam} \mathbf{p}_{cam} + \mathbf{t}_{base}^{cam}$$
* **Task-Space Closed-Loop Regulation**: Regulates Cartesian translation error $\mathbf{e}_p$ and rotation error $\mathbf{e}_{rot}$ directly:
  $$\mathbf{v}_{cmd} = K_p \mathbf{e}_p, \quad \boldsymbol{\omega}_{cmd} = K_r \mathbf{e}_{rot}$$
* **Adaptive Damped Least Squares (DLS)**: Dynamic differential kinematics:
  $$\dot{\mathbf{q}} = \mathbf{J}^T \left(\mathbf{J} \mathbf{J}^T + \lambda^2 \mathbf{I}\right)^{-1} \boldsymbol{\mathcal{V}}_{task}$$
  where $\lambda$ smoothly increases as the minimum singular value $\sigma_{min}(\mathbf{J})$ approaches kinematic singularities.

### B. Multi-Stage Filtering Pipeline
1. **RealSense SDK Hardware/Driver Filters**:
   * *Threshold Filter*: Isolates table workspace ($0.20\text{ m} \le z \le 1.80\text{ m}$), eliminating background clutter.
   * *Disparity Transform*: Prepares depth data for spatial/temporal operations.
   * *Spatial Filter*: Edge-preserving spatial smoothing to suppress pixel noise.
   * *Temporal Filter*: Exponential moving average across consecutive frames to eliminate depth flicker for stationary tripod mounting.
   * *Hole-Filling Filter*: Recovers dropouts from reflective table surfaces.
2. **Computer Vision Filtering**:
   * Color segmentation in HSV space with morphological opening and closing to remove camera noise.
   * Top-surface point isolation using 90th percentile depth slicing and SVD surface normal consistency.
   * Modulo $\pi/2$ rotated bounding box yaw estimation.
   * **Multi-frame stability filter**: Requires 8 consecutive frames with position jitter $< 2\text{ mm}$ and circular mean angular jitter $< 5^\circ$.
3. **Control Safety & Motion Filtering**:
   * Task-space velocity limiter: linear $\le 30\text{ mm/s}$, angular $\le 0.18\text{ rad/s}$.
   * Joint velocity limiter: $\le 0.30\text{ rad/s}$ with direction-preserving scaling.
   * Soft joint limit deceleration protection.

### C. Zero-Dependency OnRobot RG2 Driver
* Uses standard library HTTP XML-RPC directly to port `41414`, completely bypassing `pycurl` compilation issues on Windows while retaining full grip and width feedback.

---

## 2. Hardware Setup & Connection Checklist

### Step 1: Network Configuration (Laptop)
1. Connect Ethernet cable between laptop and UR3 control box.
2. In Windows Ethernet Adapter Settings, set manual IPv4:
   * **IP Address**: `192.168.0.100`
   * **Subnet Mask**: `255.255.255.0`
   * **Default Gateway**: `192.168.0.1`
3. Test connection in PowerShell / Command Prompt:
   ```powershell
   ping 192.168.0.X   # Replace with robot IP printed on the cart
   ```

### Step 2: UR3 Robot Pendant Setup
1. Turn on controller and release joint brakes (`ON` -> `START`).
2. Confirm OnRobot URCap is enabled under **Installation → Tool/IO**.
3. Set the pendant mode selector at the top to **Remote Control**.

### Step 3: Tripod Camera Placement
1. Mount the Intel RealSense D435i on a tripod facing the table workspace.
2. Connect camera via a **USB 3.0 port** (blue port / SS indicator).
3. Ensure the target table area and robot workspace are in clear view.

---

### Setting Your Custom Home and Deposit Positions (Quick & Easy)

You can set your preferred **Home observation view** and **Deposit/Drop location** directly from the robot:

1. **Set Home Observation Pose**:
   * In **Local mode**, hold Freedrive and move the robot arm so it looks down nicely at the table workspace.
   * Run:
     ```powershell
     ..\.venv\Scripts\python.exe live_demo_main.py --set-home
     ```
     *(Saves your current joint angles into `demo_config.json`).*

2. **Set Deposit / Drop Position**:
   * Move the gripper to where you want the cube placed down.
   * Run:
     ```powershell
     ..\.venv\Scripts\python.exe live_demo_main.py --set-drop
     ```
     *(Saves the current TCP coordinates into `demo_config.json`).*

---

## 3. Running the Live Demo

### Option A: Standard Autonomous Execution (Real Robot & Camera)
1. Place your white cube on the table within camera view.
2. On the UR teach pendant, switch the top toggle to **Remote Control**.
3. Run:
```powershell
cd "c:\Users\flynn\OneDrive\Documents\Uni Work\Year 4 sem 2\Sensors\Final Assignemnt\Live Demo"
..\.venv\Scripts\python.exe live_demo_main.py
```
*(Automatically connects to `192.168.0.197`, uses your saved calibration and `demo_config.json`, detects the white cube, and performs PBVS pick-and-place).*

### Option B: Dry-Run / Test Without Physical Robot (Mock Mode)
If verifying camera detection without moving the physical arm:
```powershell
..\.venv\Scripts\python.exe live_demo_main.py --mock-robot
```

---

## 4. Kinematic Joint-Pose Extrinsic Calibration

The calibration sequence maps the **Camera Optical Frame** directly to the **Robot Base (Absolute World Frame)** by sampling multiple distinct joint configurations $\mathbf{q} \in \mathbb{R}^6$:

```powershell
..\.venv\Scripts\python.exe calibrate_setup.py --robot-ip 192.168.0.197
```

### Calibration Procedure:
1. Keep robot in **Local mode** and hold the **Freedrive button** (black button on rear of teach pendant).
2. Move the arm into **4 or 5 diverse joint configurations** within the camera's field of view (vary base, shoulder, elbow, and wrist angles so points span 3D space).
3. The live HUD displays the real-time physical joint angles $[q_1, \dots, q_6]$ and the World Base TCP $[X, Y, Z]$.
4. For each joint configuration:
   * Press **`SPACE`** in the camera window.
   * Left-click the tool head / fingertip on the live RGB or Heatmap panel.
5. Once at least 4 diverse joint poses are sampled, press **`C`** on your keyboard.
6. The script computes the optimal transformation $\mathbf{T}_{base}^{cam} = (\mathbf{R}_{base}^{cam}, \mathbf{t}_{base}^{cam})$, prints an analytical verification report with residual errors per joint pose, and saves the complete dataset to [`camera_extrinsics.json`](file:///c:/Users/flynn/OneDrive/Documents/Uni%20Work/Year%204%20sem%202/Sensors/Final%20Assignemnt/Live%20Demo/camera_extrinsics.json).

*(To reset to nominal default settings: `python calibrate_setup.py --reset-default`).*

---

## 5. Live Demo Controls & Hotkeys

While the live OpenCV window is active:
* **`V` or `H`**: Toggle view modes:
  * **`SIDE_BY_SIDE`** (default): Dual-pane (Annotated RGB on Left + Filtered Depth Heatmap on Right).
  * **`BLEND`**: Translucent overlay of depth heatmap onto the color feed.
  * **`RGB`**: Annotated RGB feed with target bounding boxes.
  * **`HEATMAP`**: Full-screen filtered metric depth heatmap.
* **`Z`**: **Close Gripper** to cube clamp width (25.0 mm).
* **`X`**: **Open Gripper** fully (80.0 mm).
* **`G`**: **Toggle Gripper** (if open -> close, if closed -> open).
* **`SPACE`**: Immediately trigger autonomous PBVS Pick sequence on currently detected cube.
* **`A`**: **Auto-Calibrate Heatmap**: Instantly adapts the depth scale to your tripod height and workspace surface.
* **`[` / `]`**: Fine-tune minimum depth limit down/up (-/+ 0.05 m).
* **`-` / `=`**: Fine-tune maximum depth limit down/up (-/+ 0.05 m).
* **`S`**: Save current calibrated heatmap range into [`demo_config.json`](file:///c:/Users/flynn/OneDrive/Documents/Uni%20Work/Year%204%20sem%202/Sensors/Final%20Assignemnt/Live%20Demo/demo_config.json).
* **`M`**: Toggle HSV binary detection mask view.
* **`Q` or `ESC`**: Immediate safe stop and exit (halts robot motion).
* **`R`**: Reset state machine back to `OBSERVE_AND_FILTER`.
* **`1`**: Switch target color to **WHITE** (default).
* **`2`**: Switch target color to **RED**.
* **`3`**: Switch target color to **GREEN**.
* **`4`**: Switch target color to **BLUE**.

---

### Standalone Interactive Gripper Test Tool

To test the physical RG2 gripper before moving the robot:
```powershell
..\.venv\Scripts\python.exe test_gripper.py --robot-ip 192.168.0.197
```
* Press **`X`** to open (80 mm).
* Press **`Z`** to clamp (25 mm).
* Press **`G`** to toggle.
* Press **`+` / `-`** to adjust width in 5 mm steps.
* Press **`R`** to read live measured sensor width.

---

## 6. Architecture & Code Structure

```
Live Demo/
├── camera_interface.py     # RealSense D435i capture & SDK multi-stage filtering
├── vision_detector.py      # HSV color segmentation, 3D point cloud & stability filter
├── ur3e_kinematics.py      # Analytical UR3e kinematics & 6x6 geometric Jacobian
├── pbvs_controller.py      # Eye-to-Hand 6-DOF task-space PBVS servoing with DLS
├── robot_interface.py      # RTDE control, speedJ synchronization, XML-RPC RG2 gripper
├── extrinsic_calibrator.py # Umeyama SVD 3D rigid body transformation solver
├── calibrate_setup.py      # Interactive calibration wizard
├── live_demo_main.py       # Main live demo executable with visual diagnostics
└── camera_extrinsics.json  # Saved extrinsic calibration matrix
```
