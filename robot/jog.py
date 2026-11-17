"""Jog the PiPER from Windows while watching the RealSense, and save the camera's "straight ahead" home pose.

Needs robot/arm_server.py running in WSL. Run from the repo root (gazecam env):
    python robot/jog.py

Keys (click the video window first):
    q/a  J1 (base, = pan)      w/s  J2      e/d  J3
    r/f  J4                    t/g  J5 (wrist, = tilt)   y/h  J6
    [ ]  smaller / bigger step (deg)
    H    (capital) save current pose as HOME = camera level and looking straight ahead
    o    go to home            arrows: pan/tilt test around home (left/right/up/down)
    p    print status          ESC   quit (motors stay enabled)
"""
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot.arm_client import Arm, ArmError  # noqa: E402
from robot.realsense_cam import RealSense  # noqa: E402

KEYS = {ord("q"): (1, +1), ord("a"): (1, -1), ord("w"): (2, +1), ord("s"): (2, -1),
        ord("e"): (3, +1), ord("d"): (3, -1), ord("r"): (4, +1), ord("f"): (4, -1),
        ord("t"): (5, +1), ord("g"): (5, -1), ord("y"): (6, +1), ord("h"): (6, -1)}
STEPS = [0.5, 1, 2, 5, 10]
ARROWS = {2424832: (-1, 0), 2555904: (+1, 0), 2490368: (0, +1), 2621440: (0, -1)}  # Windows waitKeyEx codes


def main():
    arm, cam = Arm(), RealSense()
    print(f"camera hfov={cam.hfov:.1f} vfov={cam.vfov:.1f}")
    print(arm.status())
    si, msg, pan, tilt = 2, "", 0.0, 0.0
    while True:
        img = cam.read()
        h, w = img.shape[:2]
        cv2.line(img, (w // 2 - 40, h // 2), (w // 2 + 40, h // 2), (0, 255, 0), 2)
        cv2.line(img, (w // 2, h // 2 - 40), (w // 2, h // 2 + 40), (0, 255, 0), 2)
        cv2.putText(img, f"step {STEPS[si]} deg | pan {pan:+.0f} tilt {tilt:+.0f} | {msg}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("PiPER jog (ESC quits)", img)
        k = cv2.waitKeyEx(1)
        if k == -1:
            continue
        try:
            if k == 27:
                break
            if (k & 0xFF) in KEYS and k < 256:
                j, sgn = KEYS[k & 0xFF]
                r = arm.jog(j, sgn * STEPS[si]); msg = f"J{j} -> {r['joints'][j - 1]:.1f}"
            elif k == ord("["):
                si = max(0, si - 1)
            elif k == ord("]"):
                si = min(len(STEPS) - 1, si + 1)
            elif k == ord("H"):
                r = arm.set_home(); pan = tilt = 0.0; msg = "HOME saved"; print("home:", r["home"])
            elif k == ord("o"):
                arm.home(); pan = tilt = 0.0; msg = "at home"
            elif k in ARROWS:
                dp, dt = ARROWS[k]; pan += dp * STEPS[si]; tilt += dt * STEPS[si]
                arm.pantilt(pan, tilt); msg = "pantilt"
            elif k == ord("p"):
                print(arm.status())
        except ArmError as e:
            msg = f"ERROR {e}"; print(msg)
    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
