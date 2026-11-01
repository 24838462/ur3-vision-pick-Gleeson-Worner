import sys
import time

print("Testing RG2 gripper with active RTDE session...")

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
import xmlrpc.client
from urllib.request import Request, urlopen

robot_ip = "192.168.0.197"

print(f"Connecting RTDE to {robot_ip}...")
try:
    rtde_c = RTDEControl(robot_ip)
    rtde_r = RTDEReceive(robot_ip)
    print("RTDE connected successfully!")
    print(f"RTDE Control connected: {rtde_c.isConnected()}")
    print(f"RTDE Receive connected: {rtde_r.isConnected()}")
except Exception as e:
    print(f"Failed to connect RTDE: {e}")
    sys.exit(1)

def send_xmlrpc(method, params):
    call = xmlrpc.client.dumps(tuple(params), methodname=method)
    req = Request(f"http://{robot_ip}:41414", data=call.encode("utf-8"), headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urlopen(req, timeout=3.0) as resp:
        res, _ = xmlrpc.client.loads(resp.read().decode("utf-8"))
        return res[0]

print("\nWaiting 1s after RTDE connection...")
time.sleep(1.0)

for gid in [0, 1]:
    try:
        w = send_xmlrpc("rg_get_width", [gid])
        s = send_xmlrpc("rg_get_status", [gid])
        print(f"Gripper ID {gid}: width = {w}, status = {s}")
    except Exception as e:
        print(f"Gripper ID {gid} error: {e}")

print("\nDisconnecting RTDE...")
rtde_c.disconnect()
rtde_r.disconnect()
print("Done.")
