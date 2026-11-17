"""PiPER arm server: runs in WSL (Ubuntu-22.04), owns the CAN connection, takes JSON commands over TCP.

Windows programs (gaze pipeline, jog tool) talk to it through robot/arm_client.py on localhost:5055.

Start (Ubuntu window, after usbipd attach + piper_can_up.sh):
    python3 /mnt/c/Users/ASUS/Desktop/piper_gaze_testing/gaze-ief-research/robot/arm_server.py

Protocol: one JSON object per line, one JSON reply per line.
    {"cmd": "status"}                                  -> joints (deg), enabled, can_ok, home, limits
    {"cmd": "enable"} / {"cmd": "disable"}             -> disable makes the arm go LIMP: hold it first
    {"cmd": "joints", "deg": [j1..j6], "speed": 10}    -> move all joints (deg), blocks until reached
    {"cmd": "jog", "joint": 1..6, "delta": 2.0}        -> move one joint by delta deg
    {"cmd": "set_home"}                                -> current pose = camera "looking straight ahead"
    {"cmd": "home"}                                    -> go to home pose
    {"cmd": "pantilt", "pan": 20, "tilt": -10}         -> home pose + pan on J1 + tilt on J5 (deg)
    {"cmd": "gripper", "mm": 0..70}
"""
import json
import os
import socketserver
import sys
import threading
import time

from piper_sdk import C_PiperInterface_V2

HOST, PORT = "0.0.0.0", int(os.environ.get("ARM_PORT", "5055"))
HOME_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm_home.json")

# Conservative software limits (deg), inside the PiPER's mechanical limits.
LIMITS = [(-150, 150), (0, 180), (-170, 0), (-100, 100), (-70, 70), (-120, 120)]
PAN_SIGN, TILT_SIGN = +1, +1     # flip to -1 if pan/tilt turn the wrong way (see README)
MAX_SPEED = 30                   # % of max speed
TOL_DEG = 1.0                    # "reached" tolerance

lock = threading.Lock()
piper = C_PiperInterface_V2(os.environ.get("CAN_PORT", "can0"))
piper.ConnectPort()
time.sleep(0.5)


def load_home():
    try:
        with open(HOME_FILE) as f:
            return json.load(f)["joints_deg"]
    except (OSError, KeyError, ValueError):
        return None


def joints_deg():
    j = piper.GetArmJointMsgs().joint_state
    return [v / 1000.0 for v in (j.joint_1, j.joint_2, j.joint_3, j.joint_4, j.joint_5, j.joint_6)]


def enabled():
    low = piper.GetArmLowSpdInfoMsgs()
    return all(getattr(low, f"motor_{i}").foc_status.driver_enable_status for i in range(1, 7))


def clamp(target):
    return [min(max(v, lo), hi) for v, (lo, hi) in zip(target, LIMITS)]


def enable(timeout=5.0):
    t0 = time.time()
    while not piper.EnablePiper():
        if time.time() - t0 > timeout:
            raise RuntimeError("could not enable motors in 5 s")
        time.sleep(0.05)


def move_joints(target, speed=10, timeout=15.0):
    if not piper.isOk():
        raise RuntimeError("no CAN feedback from arm (power? usbipd attach? piper_can_up.sh?)")
    if not enabled():
        enable()
    target = clamp(target)
    speed = int(min(max(speed, 1), MAX_SPEED))
    cmd = [int(round(v * 1000)) for v in target]
    t0 = time.time()
    while True:
        piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
        piper.JointCtrl(*cmd)
        cur = joints_deg()
        if max(abs(a - b) for a, b in zip(cur, target)) < TOL_DEG:
            return {"reached": True, "joints": cur, "target": target, "secs": round(time.time() - t0, 2)}
        if time.time() - t0 > timeout:
            return {"reached": False, "joints": cur, "target": target, "secs": round(time.time() - t0, 2)}
        time.sleep(0.01)


def pantilt_target(pan, tilt):
    home = load_home()
    if home is None:
        raise RuntimeError("no home pose yet: aim the camera straight ahead (jog tool), then set_home")
    t = list(home)
    t[0] = home[0] + PAN_SIGN * pan
    t[4] = home[4] + TILT_SIGN * tilt
    return t


def handle(req):
    c = req.get("cmd")
    with lock:
        if c == "status":
            return {"joints": joints_deg(), "enabled": enabled(), "can_ok": piper.isOk(),
                    "home": load_home(), "limits": LIMITS}
        if c == "enable":
            enable(); return {"enabled": True}
        if c == "disable":
            piper.DisableArm(7); return {"enabled": False}
        if c == "joints":
            return move_joints(req["deg"], req.get("speed", 10))
        if c == "jog":
            t = joints_deg(); t[int(req["joint"]) - 1] += float(req["delta"])
            return move_joints(t, req.get("speed", 10))
        if c == "set_home":
            home = joints_deg()
            with open(HOME_FILE, "w") as f:
                json.dump({"joints_deg": home, "saved": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
            return {"home": home}
        if c == "home":
            home = load_home()
            if home is None:
                raise RuntimeError("no home pose saved yet")
            return move_joints(home, req.get("speed", 10))
        if c == "pantilt":
            return move_joints(pantilt_target(float(req["pan"]), float(req["tilt"])), req.get("speed", 10))
        if c == "gripper":
            mm = min(max(float(req["mm"]), 0), 70)
            piper.GripperCtrl(int(mm * 1000), 1000, 0x01, 0); return {"gripper_mm": mm}
        raise ValueError(f"unknown cmd {c!r}")


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        print(f"client connected: {self.client_address}", flush=True)
        for line in self.rfile:
            try:
                req = json.loads(line)
                rep = {"ok": True, **handle(req)}
            except Exception as e:  # report errors to the client, keep serving
                rep = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            print(f"{line.decode().strip()} -> {rep}", flush=True)
            self.wfile.write((json.dumps(rep) + "\n").encode())


if __name__ == "__main__":
    if not piper.isOk():
        print("WARNING: no CAN feedback yet. Check arm power / usbipd attach / piper_can_up.sh", file=sys.stderr)
    print(f"home pose: {load_home()}")
    print(f"PiPER arm server listening on {HOST}:{PORT}  (Ctrl+C to stop; motors stay enabled)", flush=True)
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((HOST, PORT), Handler) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("stopped")
