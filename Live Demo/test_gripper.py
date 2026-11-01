"""Standalone Interactive RG2 Gripper Test & Offset Verification Tool.

Connects directly to the UR controller XML-RPC service on port 41414.
Allows testing open/close commands and inspecting physical gripper widths.

Controls:
  Z       - Close gripper to clamp width (e.g. 25 mm)
  X       - Open gripper fully (80 mm)
  G       - Toggle open / close
  [ / ]   - Decrease / increase width by 5 mm
  0-9     - Set width directly (e.g. '2' = 20 mm, '5' = 50 mm, '8' = 80 mm)
  R       - Read live measured width from gripper sensor
  Q / ESC - Quit
"""

import argparse
import sys
import time
import xmlrpc.client

try:
    from rtde_control import RTDEControlInterface as RTDEControl
    from rtde_receive import RTDEReceiveInterface as RTDEReceive
    RTDE_AVAILABLE = True
except ImportError:
    RTDE_AVAILABLE = False

from robot_interface import RG2Gripper


def check_discovery(robot_ip: str):
    """Query OnRobot URCap discovery status."""
    try:
        s = xmlrpc.client.ServerProxy(f"http://{robot_ip}:41414/")
        disc = s.get_discovery()
        return disc
    except Exception as exc:
        return f"Query error: {exc}"


def main():
    parser = argparse.ArgumentParser(description="Interactive OnRobot RG2 Gripper Controller")
    parser.add_argument("--robot-ip", default="192.168.0.197", help="UR controller IP address (default: 192.168.0.197)")
    parser.add_argument("--gripper-id", type=int, default=0, help="Gripper device ID (default: 0)")
    parser.add_argument("--mock", action="store_true", help="Simulate gripper without physical hardware")
    parser.add_argument("--rtde", action="store_true", default=True, help="Maintain active RTDE session while controlling gripper")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("       OnRobot RG2 Gripper Interactive Controller & Diagnostic")
    print("=" * 65)

    rtde_c = None
    rtde_r = None
    if not args.mock and args.rtde and RTDE_AVAILABLE:
        print(f">> Starting RTDE external control session on {args.robot_ip}...")
        try:
            rtde_c = RTDEControl(args.robot_ip)
            rtde_r = RTDEReceive(args.robot_ip)
            print("   RTDE session established successfully.")
        except Exception as exc:
            print(f"   [WARN] RTDE connection failed: {exc}. Proceeding with direct control.")

    print(f">> Checking OnRobot URCap discovery at http://{args.robot_ip}:41414 ...")
    disc = check_discovery(args.robot_ip)
    print(f"   Discovery status: {disc}")

    gripper = RG2Gripper(robot_ip=args.robot_ip, gripper_id=args.gripper_id, is_mock=args.mock)
    initial_w = gripper.get_width()
    print(f"   Current measured width: {initial_w:.1f} mm")

    if not args.mock and initial_w < 0:
        print("\n" + "!" * 65)
        print("  NOTICE: Measured width is -999.0 mm (NO COMMUNICATION WITH GRIPPER)")
        print("  Please check the following on the UR Teach Pendant (PolyScope):")
        print("  1. Tool I/O Power: Installation -> General -> Tool I/O")
        print("     Set 'Controlled by' to 'OnRobot' (or Tool Voltage to 24V).")
        print("  2. Cable & LED: Check the wrist M8 cable and confirm RG2 LED is lit.")
        print("  3. Device Scan: Installation -> URCaps -> OnRobot Setup -> tap 'Scan'.")
        print("!" * 65 + "\n")

    print("\nControls:")
    print("  [X]       - Open gripper (80.0 mm)")
    print("  [Z]       - Close gripper to cube clamp width (25.0 mm)")
    print("  [C]       - Fully close gripper (5.0 mm)")
    print("  [G]       - Toggle between Open and Close")
    print("  [+] / [-] - Step width up/down by 5 mm")
    print("  [R]       - Read real-time measured width & status")
    print("  [Q]       - Exit")
    print("=" * 65 + "\n")

    current_target = 80.0

    try:
        # Non-blocking terminal input on Windows using msvcrt
        import msvcrt

        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                # Handle special keys
                if ch in (b"\x1b", b"q", b"Q"):
                    print("\nExiting gripper test.")
                    break

                key = ch.decode("utf-8", errors="ignore").lower()

                if key == "x":
                    current_target = 80.0
                    print(f">> OPEN: Commanding width={current_target:.1f} mm...")
                    gripper.open()
                    time.sleep(0.3)
                    print(f"   Actual measured width: {gripper.get_width():.1f} mm")

                elif key == "z":
                    current_target = 25.0
                    print(f">> CLAMP (Cube): Commanding width={current_target:.1f} mm...")
                    gripper.close(width_mm=current_target)
                    time.sleep(0.3)
                    print(f"   Actual measured width: {gripper.get_width():.1f} mm")

                elif key == "c":
                    current_target = 5.0
                    print(f">> FULL CLOSE: Commanding width={current_target:.1f} mm...")
                    gripper.close(width_mm=current_target)
                    time.sleep(0.3)
                    print(f"   Actual measured width: {gripper.get_width():.1f} mm")

                elif key == "g":
                    cur_w = gripper.get_width()
                    if cur_w > 45.0:
                        current_target = 25.0
                        print(f">> TOGGLE -> CLOSE: Commanding width={current_target:.1f} mm...")
                        gripper.close(width_mm=current_target)
                    else:
                        current_target = 80.0
                        print(f">> TOGGLE -> OPEN: Commanding width={current_target:.1f} mm...")
                        gripper.open()
                    time.sleep(0.3)
                    print(f"   Actual measured width: {gripper.get_width():.1f} mm")

                elif key in ("+", "="):
                    current_target = min(110.0, current_target + 5.0)
                    print(f">> STEP UP: Commanding width={current_target:.1f} mm...")
                    gripper.grip(current_target, target_force_n=20.0)
                    time.sleep(0.3)
                    print(f"   Actual measured width: {gripper.get_width():.1f} mm")

                elif key in ("-", "_"):
                    current_target = max(0.0, current_target - 5.0)
                    print(f">> STEP DOWN: Commanding width={current_target:.1f} mm...")
                    gripper.grip(current_target, target_force_n=20.0)
                    time.sleep(0.3)
                    print(f"   Actual measured width: {gripper.get_width():.1f} mm")

                elif key == "r":
                    w = gripper.get_width()
                    print(f">> Live measured width: {w:.1f} mm")

                elif key.isdigit():
                    num = int(key)
                    current_target = float(num * 10)
                    print(f">> SET: Commanding width={current_target:.1f} mm...")
                    gripper.grip(current_target, target_force_n=20.0)
                    time.sleep(0.3)
                    print(f"   Actual measured width: {gripper.get_width():.1f} mm")

            time.sleep(0.05)

    except (KeyboardInterrupt, SystemExit):
        print("\nExiting.")
    finally:
        if rtde_c is not None:
            try:
                rtde_c.disconnect()
            except Exception:
                pass
        if rtde_r is not None:
            try:
                rtde_r.disconnect()
            except Exception:
                pass
        print("Gripper test session finished.")


if __name__ == "__main__":
    main()
