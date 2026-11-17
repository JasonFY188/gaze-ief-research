"""Move the arm slowly to the start (home) pose: standing up, camera level and straight ahead.
Needs robot/arm_server.py running. Run from the repo root:  python robot/go_home.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot.arm_client import Arm  # noqa: E402

arm = Arm()
s = arm.status()
if not s["can_ok"]:
    sys.exit("No data from the arm (can_ok False): power-cycle the arm, then piper_can_up.sh (see HOW_TO_USE_PIPER.md).")
print("going to home", [round(j, 1) for j in s["home"]], "...")
r = arm.home(speed=10)
print("reached" if r["reached"] else "NOT reached", [round(j, 1) for j in r["joints"]], f"({r['secs']} s)")
