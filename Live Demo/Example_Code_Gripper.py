"""Control an OnRobot RG2 gripper without commanding the robot arm."""

import argparse

import onRobot.gripper as gripper
from pynput import keyboard
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive


CLOSED_WIDTH_MM = 10.0
OPEN_WIDTH_MM = 80.0
GRIP_FORCE_N = 20.0
RTDESession = tuple[RTDEControl, RTDEReceive]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control the OnRobot RG2 gripper from the keyboard.",
    )
    parser.add_argument(
        "--ip",
        required=True,
        help="IP address of the UR controller hosting the OnRobot URCap",
    )
    parser.add_argument(
        "--gripper-id",
        type=int,
        default=0,
        help="RG gripper index (default: 0)",
    )
    return parser.parse_args()


def setup_gripper(
    robot_ip: str,
    gripper_id: int,
    rtde_session: RTDESession,
) -> gripper.RG:
    """Create the RG2 interface using an active RTDE session."""
    rtde_c, rtde_r = rtde_session
    if not rtde_c.isConnected() or not rtde_r.isConnected():
        raise ConnectionError("An active RTDE session is required for RG2 control")

    print("Connecting to RG2 gripper via onRobot library...")
    rg2 = gripper.RG(robot_ip, gripper_id)
    initial_width = rg2.get_rg_width()
    print(f"RG2 gripper connected. Current width: {initial_width:.1f} mm")
    return rg2


def setup_rtde(robot_ip: str) -> RTDESession:
    """Start the external-control session required for RG2 write commands."""
    print("Connecting to robot via RTDE...")
    rtde_c = RTDEControl(robot_ip)

    try:
        rtde_r = RTDEReceive(robot_ip)
    except Exception:
        rtde_c.disconnect()
        raise

    if not rtde_c.isConnected() or not rtde_r.isConnected():
        rtde_c.disconnect()
        rtde_r.disconnect()
        raise ConnectionError("RTDE control or receive interface is not connected")

    print("RTDE external-control session connected.")
    return rtde_c, rtde_r


def main() -> None:
    args = parse_args()

    try:
        rtde_session = setup_rtde(args.ip)
    except Exception as exc:
        raise SystemExit(f"Failed to start RTDE external control: {exc}") from exc

    rtde_c, rtde_r = rtde_session

    try:
        try:
            rg2 = setup_gripper(args.ip, args.gripper_id, rtde_session)
        except Exception as exc:
            raise SystemExit(f"Failed to connect to RG2 gripper: {exc}") from exc

        print("\n--- Gripper Controls ---")
        print(f"Z   - Close to {CLOSED_WIDTH_MM:.0f} mm")
        print(f"X   - Open to {OPEN_WIDTH_MM:.0f} mm")
        print("ESC - Exit program\n")

        pressed_keys: set[str] = set()

        def command_gripper(width_mm: float, action: str) -> None:
            print(f"Commanding gripper: {action} ({width_mm:.0f} mm)")
            try:
                rg2.rg_grip(width_mm, GRIP_FORCE_N)
                actual_width = rg2.get_rg_width()
                print(f"Gripper width: {actual_width:.1f} mm")
            except Exception as exc:
                print(f"Gripper error: {exc}")

        def on_press(key):
            try:
                key_char = key.char.lower()
            except AttributeError:
                if key == keyboard.Key.esc:
                    return False
                return None

            # Ignore keyboard auto-repeat while a control key remains held down.
            if key_char in pressed_keys:
                return None

            pressed_keys.add(key_char)
            if key_char == "z":
                command_gripper(CLOSED_WIDTH_MM, "CLOSE")
            elif key_char == "x":
                command_gripper(OPEN_WIDTH_MM, "OPEN")

            return None

        def on_release(key):
            try:
                pressed_keys.discard(key.char.lower())
            except AttributeError:
                pass

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

        print("Gripper control stopped.")
    finally:
        rtde_r.disconnect()
        rtde_c.disconnect()
        print("RTDE disconnected.")


if __name__ == "__main__":
    main()