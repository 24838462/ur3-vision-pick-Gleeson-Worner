from rtde_receive import RTDEReceiveInterface

r = RTDEReceiveInterface("192.168.0.197")
print("Robot Mode:", r.getRobotMode())
print("Safety Mode:", r.getSafetyMode())
print("Robot Status:", r.getRobotStatus())
print("Tool Output Voltage:", r.getActualToolOutputVoltage())
print("Tool Output Current:", r.getActualToolOutputCurrent())
print("Tool Analog Input 0:", r.getActualToolAnalogInput0())
print("Tool Analog Input 1:", r.getActualToolAnalogInput1())
print("Tool Digital Output 0:", r.getActualToolDigitalOutput0())
print("Tool Digital Output 1:", r.getActualToolDigitalOutput1())
r.disconnect()
