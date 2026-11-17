"""Make the camera level and straight ahead by itself, then save that pose as HOME.

Keeps the shoulder (J2) and elbow (J3) where they are, sets base J1 = 0, roll J4 = 0, J6 = 0, and solves the
wrist J5 with the PiPER kinematics (robot/piper_fk.py) so the camera's elevation is 0 deg.
Needs robot/arm_server.py running. Run from the repo root:
    python robot/level_home.py            # move + save home
    python robot/level_home.py --dry-run  # only print the target
"""
import argparse
import os
import sys

from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot.arm_client import Arm  # noqa: E402
from robot.piper_fk import camera  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    arm = Arm()
    cur = arm.status()["joints"]
    if cur[1] < 10:
        sys.exit("The arm looks folded (shoulder J2 < 10 deg). Raise it first with robot/jog.py (w/s, e/d).")
    j5 = brentq(lambda x: camera([0, cur[1], cur[2], 0, x, 0])["elevation_deg"], -69, 69)
    tgt = [0.0, cur[1], cur[2], 0.0, round(j5, 2), 0.0]
    print("now   ", [round(j, 1) for j in cur], f"camera elevation {camera(cur)['elevation_deg']:+.1f} deg")
    print("target", tgt, f"camera elevation {camera(tgt)['elevation_deg']:+.2f} deg, height {camera(tgt)['height']:.3f} m")
    if args.dry_run:
        return
    r = arm.joints(tgt, speed=10)
    if not r["reached"]:
        sys.exit(f"did not reach the target (joints {r['joints']}); home NOT saved")
    print("HOME saved:", [round(j, 1) for j in arm.set_home()["home"]])


if __name__ == "__main__":
    main()
