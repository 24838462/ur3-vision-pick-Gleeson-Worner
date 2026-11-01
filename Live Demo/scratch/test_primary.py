import socket
import time

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(3.0)
s.connect(("192.168.0.197", 30002))

# Read a packet
header = s.recv(4)
print("Primary interface connected! Packet header length:", int.from_bytes(header, "big"))
s.close()
