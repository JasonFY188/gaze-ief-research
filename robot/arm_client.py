"""Client for robot/arm_server.py. Works from Windows (WSL forwards localhost) or another PC (pass host).

    from robot.arm_client import Arm
    arm = Arm()                    # localhost:5055
    arm.status()
    arm.pantilt(20, -10)           # pan right 20 deg, tilt 10 deg (sign depends on mount, see README)
"""
import json
import socket


class ArmError(RuntimeError):
    pass


class Arm:
    def __init__(self, host="127.0.0.1", port=5055, timeout=30.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.f = self.sock.makefile("rwb")

    def call(self, cmd, **kw):
        self.f.write((json.dumps({"cmd": cmd, **kw}) + "\n").encode())
        self.f.flush()
        line = self.f.readline()
        if not line:
            raise ArmError("arm server closed the connection")
        rep = json.loads(line)
        if not rep.pop("ok"):
            raise ArmError(rep["error"])
        return rep

    def status(self): return self.call("status")
    def enable(self): return self.call("enable")
    def disable(self): return self.call("disable")
    def set_home(self): return self.call("set_home")
    def home(self, speed=10): return self.call("home", speed=speed)
    def joints(self, deg, speed=10): return self.call("joints", deg=list(deg), speed=speed)
    def jog(self, joint, delta, speed=10): return self.call("jog", joint=joint, delta=delta, speed=speed)
    def pantilt(self, pan, tilt, speed=10): return self.call("pantilt", pan=pan, tilt=tilt, speed=speed)

    def close(self):
        self.sock.close()
