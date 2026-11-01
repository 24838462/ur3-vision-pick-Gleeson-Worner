import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(3.0)
s.connect(("192.168.0.197", 29999))
s.recv(1024)

# Get log or status
for cmd in ["get loaded program", "isProgramSaved", "running"]:
    s.sendall(f"{cmd}\n".encode("utf-8"))
    ans = s.recv(1024).decode("utf-8").strip()
    print(f"{cmd}: {ans}")

s.close()
