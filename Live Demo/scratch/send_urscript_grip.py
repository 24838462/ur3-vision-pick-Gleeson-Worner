import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('192.168.0.197', 30002))
script = """sec grip_prog():
  textmsg("Testing OnRobot grip via URScript")
  rg_grip(0, 10.0, 20.0)
end
grip_prog()
"""
s.sendall(script.encode('utf-8'))
s.close()
print("Sent URScript grip command.")
