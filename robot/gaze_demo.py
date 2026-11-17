"""Robot gaze demo: find the person -> face them -> (SPACE + 5 s) photo -> where to turn -> turn -> photo -> gaze target.

Run from the repo root with the gazecam env (Windows). The arm server must run in WSL for `scan` and `run`.

  python robot/gaze_demo.py face --brain <GPU PC IP>   # test: only find the person and turn to face them
  python robot/gaze_demo.py scan                       # robot makes its own 6-photo room map (60 deg steps)
  python robot/gaze_demo.py run --brain <GPU PC IP> [--room DIR] [--delay 5]   # models on the lab PC
  python robot/gaze_demo.py run [--room DIR] [--delay 5]                       # models on this PC
  python robot/gaze_demo.py run --brain <IP> --map data/captures/20261116_1857_Room_60_degree \
         --anchor data/scenes/Person1/scenario_1_2m/testing_1scenario_2m/person_01.jpg
         # saved iPhone room map, no robot scan; the robot camera is assumed to be where the anchor photo was taken
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
    """Newest robot room scan that is complete (6 photos listed in poses.jsonl); half-finished scans are skipped."""
    for d in sorted(glob.glob(os.path.join(HERE, "data", "captures", "robot_*")), reverse=True):
        pj = os.path.join(d, "poses.jsonl")
        if os.path.exists(pj) and sum(1 for line in open(pj, encoding="utf-8") if line.strip()) == len(ROOM_PANS):
            return d
    return None


def phone_K(path, fov_short=None):
    """Intrinsics (original pixels) of a phone photo: measured short-side FOV if given, else EXIF 35 mm focal."""
    from PIL import Image, ImageOps
    im = Image.open(path)
    w, h = ImageOps.exif_transpose(im).size
    if fov_short is not None:
        f = (min(w, h) / 2) / math.tan(math.radians(fov_short / 2))
    else:
        f35 = im.getexif().get_ifd(0x8769).get(0xA405)
        if not f35:
            raise ValueError(f"{path}: no EXIF focal length; pass a measured FOV")
        f = float(f35) * math.hypot(w, h) / math.hypot(36, 24)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def rel(p):
    return os.path.relpath(os.path.abspath(p), HERE).replace("\\", "/")


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
    si, lost, prev = 0, 0, None
    for _ in range(40):
        img = cam.read()
        fs = brain.faces(img)
        if not fs:
            lost += 1
            cv2.imshow("robot", overlay(img, ["looking for a person ..."])); cv2.waitKey(1)
            if lost >= 3:                                    # nobody here: try the next search direction
                si = (si + 1) % len(search); pan, tilt, lost, prev = float(search[si]), 0.0, 0, None
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
        # safety: after a correction the face must get closer to the centre; if it moved further away,
        # that axis is wired the wrong way round -> stop instead of turning away from the person
        if prev is not None:
            for name, now, before, step in (("pan", dpan, prev[0], prev[2]), ("tilt", dtilt, prev[1], prev[3])):
                if abs(step) >= 3 and abs(now) > abs(before) + 2 and np.sign(now) == np.sign(before):
                    raise RuntimeError(f"{name} turns the wrong way: the face moved further from the centre "
                                       f"({before:+.0f} -> {now:+.0f} deg). Flip {name.upper()}_SIGN in robot/arm_server.py.")
        step_p, step_t = float(np.clip(dpan, -20, 20)), float(np.clip(dtilt, -20, 20))   # gentle steps
        prev = (dpan, dtilt, step_p, step_t)
        pan = float(np.clip(pan + step_p, *PAN_LIMITS)); tilt = float(np.clip(tilt + step_t, -40, 40))
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
    arm = Arm()
    if arm.status()["home"] is None:
        sys.exit("No home pose yet: run robot/jog.py, aim the camera level and straight ahead, press Shift+H.")
    cam = RealSense()
    if args.brain:                                   # heavy models on the GPU PC (robot/brain_server.py)
        from robot.brain_client import RemoteBrain
        brain = RemoteBrain(args.brain)
    else:
        from robot.gaze_brain import Brain
        brain = Brain()
    K = K_of(cam.intr)
    if args.map:                                     # saved room map (e.g. iPhone circle): no robot scan
        from robot.piper_fk import camera as fk_camera
        map_dir = rel(args.map)
        map_photos = sorted(glob.glob(os.path.join(HERE, map_dir, "*.jpg")))
        K_map = phone_K(map_photos[0], args.map_fov_short)
        anchor = rel(args.anchor) if args.anchor else None
        K_anchor = phone_K(os.path.join(HERE, anchor)) if anchor else None
        print(f"[run] saved room map {map_dir} ({len(map_photos)} photos, short-side FOV {args.map_fov_short})"
              f"{'; extra structure: ' + anchor if anchor else ''}; robot camera position: from its own photo")
    else:
        room = args.room or latest_robot_room()
        if room is None or args.rescan:
            room = scan(arm, cam)
        room_paths, room_pans, _ = read_room(room)
        print(f"[run] room map: {room} ({len(room_paths)} photos)")
    arm.home()
    while True:
        pan0, tilt0 = find_and_face(arm, cam, brain)
        person = countdown_photo(cam, args.delay)
        if person is None:
            break
        out = os.path.join(OUT_ROOT, stamp())
        os.makedirs(out, exist_ok=True)
        cv2.imwrite(os.path.join(out, "person.jpg"), person)
        if args.map:
            fk = fk_camera(arm.status()["joints"])
            arm_info = {"radius_m": fk["radius"], "height_m": fk["height"], "cam_elevation_deg": fk["elevation_deg"]}
            if isinstance(brain, __import__("robot.brain_client", fromlist=["RemoteBrain"]).RemoteBrain):
                plan = brain.plan_map(map_dir, K_map, person, K, cam.hfov, cam.vfov, pan0=pan0, tilt0=tilt0,
                                      pan_limits=PAN_LIMITS, anchor=anchor, K_anchor=K_anchor, arm=arm_info, viz=True)
            else:
                plan = brain.plan_map(map_photos, K_map, person, K, cam.hfov, cam.vfov, pan0=pan0, tilt0=tilt0,
                                      pan_limits=PAN_LIMITS, anchor=os.path.join(HERE, anchor) if anchor else None,
                                      K_anchor=K_anchor, arm=arm_info, viz=True)
            viz = plan.pop("viz", None)                  # 3D room + footprint + search grid, for robot/report.py
            if viz:
                np.savez_compressed(os.path.join(out, "viz.npz"),
                                    **{k: np.asarray(v) for k, v in viz.items() if v is not None})
            plan.update(person_robot_pan=pan0, person_robot_tilt=tilt0, map=map_dir, anchor=anchor)
            print(f"[run] gaze yaw {plan['yaw']:+.0f} pitch {plan['pitch']:+.0f} cone ±{plan['cone_half']:.0f}  ->  turn by "
                  f"pan {plan['turn_pan']:+.0f} tilt {plan['turn_tilt']:+.0f} = robot pan {plan['robot_pan']:+.0f} tilt "
                  f"{plan['robot_tilt']:+.0f}; sees {plan['captured']:.0%} of the footprint "
                  f"(now {plan['captured_without_turning']:.0%}); {plan['secs_total']} s")
            if plan.get("robot_cam_vggt_to_anchor") is not None:
                print(f"[run] robot camera is {plan['robot_cam_vggt_to_anchor']:.2f} VGGT units from the anchor photo's "
                      f"spot (room size {plan['map_size']:.2f}, eye-camera {plan['eye_to_robot_cam']:.2f})")
            if plan["cone_half"] >= 45:
                print(f"[run] WARNING: gaze very uncertain (cone ±{plan['cone_half']:.0f} deg). Face the camera at ~1.5 m, "
                      f"camera at face height, turn your HEAD to the object.")
        else:
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
        k = cv2.waitKey(0)
        # record for the presentation: figures, 3D view, runs.csv (never stops the demo if drawing fails)
        if os.path.exists(os.path.join(out, "viz.npz")):
            try:
                note = input("[run] What was the person really looking at? (Enter = skip): ").strip()
                from robot import report
                report.make(out, note or None)
            except Exception as e:
                print(f"[run] report failed ({type(e).__name__}: {e}); the run is still saved in {out}")
        if k == 27:
            break
        arm.home()
    cam.stop(); cv2.destroyAllWindows()


def face(args):
    """Only: find the person and turn to face them (no room scan, no gaze). For testing the arm + brain link."""
    from robot.arm_client import Arm
    from robot.realsense_cam import RealSense
    arm, cam = Arm(), RealSense()
    if args.brain:
        from robot.brain_client import RemoteBrain
        brain = RemoteBrain(args.brain)
    else:
        from robot.gaze_brain import Brain
        brain = Brain()
    if arm.status()["home"] is None:
        sys.exit("No home pose yet: run robot/jog.py, aim the camera level and straight ahead, press Shift+H.")
    arm.home()
    try:
        while True:
            pan, tilt = find_and_face(arm, cam, brain)
            print(f"[face] facing you at pan {pan:+.1f}, tilt {tilt:+.1f}")
            while True:                                           # keep showing the view until a key is pressed
                img = cam.read()
                for x1, y1, x2, y2, _ in brain.faces(img)[:1]:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                cv2.imshow("robot", overlay(img, [f"facing you (pan {pan:+.0f}, tilt {tilt:+.0f})",
                                                  "SPACE = find again, ESC = quit"]))
                k = cv2.waitKey(1)
                if k in (27, 32):
                    break
            if k == 27:
                break
    finally:
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
    r.add_argument("--map", help="saved room map instead of a robot scan, e.g. data/captures/20261116_1857_Room_60_degree")
    r.add_argument("--map-fov-short", type=float, default=55.9, help="measured short-side FOV of the map photos (deg)")
    r.add_argument("--anchor", help="earlier photo taken from where the robot camera is now (e.g. the iPhone person photo)")
    fc = sub.add_parser("face")
    fc.add_argument("--brain", help="IP of the GPU PC running robot/brain_server.py")
    o = sub.add_parser("offline")
    o.add_argument("--room", required=True); o.add_argument("--person", required=True)
    o.add_argument("--fov-short", type=float, default=55.9)
    args = ap.parse_args()
    if args.mode == "offline":
        offline(args)
    elif args.mode == "run":
        run(args)
    elif args.mode == "face":
        face(args)
    else:
        from robot.arm_client import Arm
        from robot.realsense_cam import RealSense
        arm, cam = Arm(), RealSense()
        print(scan(arm, cam, show=False)); cam.stop()


if __name__ == "__main__":
    main()
