"""Robot gaze demo: find the person -> face them -> (SPACE + 5 s) photo -> where to turn -> turn -> photo -> gaze target.

Run from the repo root with the gazecam env (Windows). The arm server must run in WSL for `scan` and `run`.

  python robot/gaze_demo.py scan                       # robot makes its own 6-photo room map (60 deg steps)
  python robot/gaze_demo.py run --brain <GPU PC IP> [--room DIR] [--delay 5]   # models on the lab PC
  python robot/gaze_demo.py run [--room DIR] [--delay 5]                       # models on this PC
  python robot/gaze_demo.py offline --room data/captures/20261116_1857_Room_60_degree \
         --person data/scenes/Person1/scenario_1_2m/testing_1scenario_2m/person_01.jpg --fov-short 55.9
         # no robot: tests the brain on saved iPhone photos (the 'turned' photo = closest room photo)

Everything is saved under data/ or outputs_robot/ (both git-ignored: person photos are never committed).
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

ROOM_PANS = [-150, -90, -30, 30, 90, 150]   # 6 photos, 60 deg apart; RealSense hfov 70 deg -> they overlap all round
PAN_LIMITS = (-145.0, 145.0)                 # robot pan the arm may use (J1 software limit is +-150)
OUT_ROOT = os.path.join(HERE, "outputs_robot")


def stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def K_of(intr):
    return np.array([[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1.0]])


def latest_robot_room():
    rooms = sorted(glob.glob(os.path.join(HERE, "data", "captures", "robot_*")))
    return rooms[-1] if rooms else None


def read_room(room_dir):
    rows = [json.loads(line) for line in open(os.path.join(room_dir, "poses.jsonl"), encoding="utf-8")]
    return [os.path.join(room_dir, r["file"]) for r in rows], [r.get("pan", r.get("yaw")) for r in rows], rows


# ---------------------------------------------------------------------------- scan
def scan(arm, cam, show=True):
    out = os.path.join(HERE, "data", "captures", f"robot_{stamp()}_Room_60_degree")
    os.makedirs(out, exist_ok=True)
    print(f"[scan] 6 photos -> {out}")
    with open(os.path.join(out, "poses.jsonl"), "w", encoding="utf-8") as f:
        for pan in ROOM_PANS:
            r = arm.pantilt(pan, 0, speed=15)
            time.sleep(1.0)                                   # let the arm settle (no motion blur)
            img = cam.read()
            name = f"robot_tilt0_pan{pan:+04d}.jpg"
            cv2.imwrite(os.path.join(out, name), img)
            f.write(json.dumps(dict(file=name, pan=pan, tilt=0, joints=r["joints"], reached=r["reached"])) + "\n")
            print(f"[scan] pan {pan:+4d}  reached={r['reached']}")
            if show:
                cv2.imshow("robot", img); cv2.waitKey(1)
    arm.home()
    json.dump(cam.intr, open(os.path.join(out, "camera_intrinsics.json"), "w"), indent=2)
    return out


# ---------------------------------------------------------------------------- live helpers
def overlay(img, lines, color=(0, 255, 255)):
    for i, t in enumerate(lines):
        cv2.putText(img, t, (15, 40 + 38 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(img, t, (15, 40 + 38 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return img


def find_and_face(arm, cam, brain, pan=0.0, tilt=0.0):
    """Turn until the largest face is in the middle of the image. Returns the (pan, tilt) that faces the person."""
    intr = cam.intr
    search = [0, -40, 40, -80, 80, -120, 120]
    si, lost = 0, 0
    for _ in range(40):
        img = cam.read()
        fs = brain.faces(img)
        if not fs:
            lost += 1
            cv2.imshow("robot", overlay(img, ["looking for a person ..."])); cv2.waitKey(1)
            if lost >= 3:                                    # nobody here: try the next search direction
                si = (si + 1) % len(search); pan, tilt, lost = float(search[si]), 0.0, 0
                arm.pantilt(pan, tilt, speed=15)
            continue
        lost = 0
        x1, y1, x2, y2, _ = fs[0]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dpan = math.degrees(math.atan((cx - intr["ppx"]) / intr["fx"]))      # face right of centre -> turn right
        dtilt = -math.degrees(math.atan((cy - intr["ppy"]) / intr["fy"]))    # face above centre -> look up
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.imshow("robot", overlay(img, [f"person found, centring ({dpan:+.0f}, {dtilt:+.0f} deg)"])); cv2.waitKey(1)
        if abs(dpan) < 3 and abs(dtilt) < 3:
            return pan, tilt
        pan = float(np.clip(pan + dpan, *PAN_LIMITS)); tilt = float(np.clip(tilt + dtilt, -40, 40))
        arm.pantilt(pan, tilt, speed=15)
        time.sleep(0.3)
    raise RuntimeError("could not centre on a face")


def countdown_photo(cam, delay):
    """Live view; SPACE starts a countdown; the photo is taken at 0. ESC aborts."""
    while True:
        img = cam.read()
        cv2.imshow("robot", overlay(img.copy(), ["facing you.  Press SPACE, then look at something",
                                                 f"(photo after {delay:.0f} s)   ESC = quit"]))
        k = cv2.waitKey(1)
        if k == 27:
            return None
        if k == 32:
            break
    t_end = time.time() + delay
    while time.time() < t_end:
        img = cam.read()
        cv2.imshow("robot", overlay(img.copy(), [f"look now ...  {math.ceil(t_end - time.time())}"], (0, 0, 255)))
        cv2.waitKey(1)
    img = cam.read()
    cv2.imshow("robot", overlay(img.copy(), ["photo taken - thinking ..."], (0, 255, 0))); cv2.waitKey(1)
    return img


# ---------------------------------------------------------------------------- modes
def run(args):
    from robot.arm_client import Arm
    from robot.realsense_cam import RealSense
    arm, cam = Arm(), RealSense()
    if args.brain:                                   # heavy models on the GPU PC (robot/brain_server.py)
        from robot.brain_client import RemoteBrain
        brain = RemoteBrain(args.brain)
    else:
        from robot.gaze_brain import Brain
        brain = Brain()
    room = args.room or latest_robot_room()
    if room is None or args.rescan:
        room = scan(arm, cam)
    room_paths, room_pans, _ = read_room(room)
    print(f"[run] room map: {room} ({len(room_paths)} photos)")
    K = K_of(cam.intr)
    arm.home()
    while True:
        pan0, tilt0 = find_and_face(arm, cam, brain)
        person = countdown_photo(cam, args.delay)
        if person is None:
            break
        out = os.path.join(OUT_ROOT, stamp())
        os.makedirs(out, exist_ok=True)
        cv2.imwrite(os.path.join(out, "person.jpg"), person)
        plan = brain.plan(room_paths, person, K, K, cam.hfov, cam.vfov, room_pans=room_pans, pan_limits=PAN_LIMITS)
        plan.update(person_robot_pan=pan0, person_robot_tilt=tilt0, room=room)
        print(f"[run] gaze yaw {plan['yaw']:+.0f} pitch {plan['pitch']:+.0f} cone ±{plan['cone_half']:.0f}  ->  turn to "
              f"pan {plan['robot_pan']:+.0f} tilt {plan['robot_tilt']:+.0f} (sees {plan['captured']:.0%} of the footprint; "
              f"{plan['secs_total']} s)")
        print(f"[run] check: VGGT vs robot angle of the room photos (deg): "
              f"{[round(x, 1) for x in plan['room_pan_residual_deg']]}")
        arm.pantilt(plan["robot_pan"], plan["robot_tilt"], speed=15)
        time.sleep(1.0)
        shot = cam.read()
        cv2.imwrite(os.path.join(out, "target.jpg"), shot)
        tgt, over = brain.target(shot, K, out_dir=out)
        plan.update(target=tgt)
        json.dump(plan, open(os.path.join(out, "result.json"), "w"), indent=1, default=float)
        print(f"[run] predicted gaze target at pixel {tgt['peak_px']} -> {out}")
        cv2.imshow("robot", overlay(over, ["predicted gaze target (green cross)", "any key = again, ESC = quit"]))
        if cv2.waitKey(0) == 27:
            break
        arm.home()
    cam.stop(); cv2.destroyAllWindows()


def offline(args):
    """No robot: plan on saved photos, then use the room photo closest to the chosen pan as the 'turned' photo."""
    from PIL import Image, ImageOps
    from robot.gaze_brain import Brain
    room_paths, yaws, rows = read_room(args.room)
    w, h = ImageOps.exif_transpose(Image.open(room_paths[0])).size
    fx = (min(w, h) / 2) / math.tan(math.radians(args.fov_short / 2))          # measured short-side FOV
    K_room = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1.0]])
    pim = Image.open(args.person)
    f35 = pim.getexif().get_ifd(0x8769).get(0xA405)
    pw, ph = ImageOps.exif_transpose(pim).size
    fp = float(f35) * math.hypot(pw, ph) / math.hypot(36, 24) if f35 else fx * pw / w
    K_person = np.array([[fp, 0, pw / 2], [0, fp, ph / 2], [0, 0, 1.0]])
    hfov = math.degrees(2 * math.atan(w / 2 / fx)); vfov = math.degrees(2 * math.atan(h / 2 / fx))
    brain = Brain()
    plan = brain.plan(room_paths, args.person, K_room, K_person, hfov, vfov)
    print(json.dumps({k: v for k, v in plan.items()}, indent=1, default=float))
    # stand-in for "turn and take a photo": the room photo whose VGGT direction is closest to the chosen pan
    import active_camera as A
    st = brain.state
    out = os.path.join(OUT_ROOT, f"offline_{stamp()}")
    v = brain._vggt(st["rgbs"][:st["nR"]], *plan["net_size"])
    fwd = np.stack([e[2, :3] for e in v["extr"]])
    _, _, vt = np.linalg.svd(fwd); up = vt[2] if vt[2] @ (-v["extr"][0][1, :3]) > 0 else -vt[2]
    pans = np.array([A.pan_tilt_of(f, fwd[0], up)[0] for f in fwd])
    k = int(np.argmin(np.abs((pans - plan["pan_vggt"] + 180) % 360 - 180)))
    print(f"[offline] chosen pan {plan['pan_vggt']:+.0f}; closest room photo {os.path.basename(room_paths[k])} "
          f"(pan {pans[k]:+.0f}) used as the 'turned' photo")
    tgt, _ = brain.target(room_paths[k], K_room, out_dir=out)
    plan.update(target=tgt, turned_photo=os.path.basename(room_paths[k]))
    json.dump(plan, open(os.path.join(out, "result.json"), "w"), indent=1, default=float)
    print(f"[offline] predicted gaze target pixel {tgt['peak_px']} ({tgt['secs']} s) -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("scan")
    r = sub.add_parser("run")
    r.add_argument("--room"); r.add_argument("--rescan", action="store_true"); r.add_argument("--delay", type=float, default=5.0)
    r.add_argument("--brain", help="IP of the GPU PC running robot/brain_server.py (default: run the models here)")
    o = sub.add_parser("offline")
    o.add_argument("--room", required=True); o.add_argument("--person", required=True)
    o.add_argument("--fov-short", type=float, default=55.9)
    args = ap.parse_args()
    if args.mode == "offline":
        offline(args)
    elif args.mode == "run":
        run(args)
    else:
        from robot.arm_client import Arm
        from robot.realsense_cam import RealSense
        arm, cam = Arm(), RealSense()
        print(scan(arm, cam, show=False)); cam.stop()


if __name__ == "__main__":
    main()
