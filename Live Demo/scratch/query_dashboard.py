import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(3.0)
try:
    s.connect(("192.168.0.197", 29999))
    banner = s.recv(1024).decode("utf-8")
    print("Dashboard Banner:", banner.strip())
    
    for cmd in ["robotmode", "safetyStatus", "programState", "polyscopeVersion"]:
        s.sendall(f"{cmd}\n".encode("utf-8"))
        ans = s.recv(1024).decode("utf-8").strip()
        print(f"  {cmd}: {ans}")
    s.close()
except Exception as e:
    print("Dashboard error:", e)
