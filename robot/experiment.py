"""Experiment: is aligning the gaze-cone footprint with the camera FOV actually necessary?

Three ways of aiming the camera are compared, one after the other, for every gaze target:
  1 footprint  the current method: the view that covers the most of the gaze-cone footprint (FastSearch)
  2 parallel   the camera looks in the same direction as the gaze vector
  3 raycast    the camera aims at the point where the gaze ray meets the 3D map

For each method: arm goes home -> finds and faces the person -> SPACE -> photo -> arm turns -> photo -> you judge it.

Needs robot/arm_server.py (WSL) and robot/brain_server.py (GPU PC). Run from the repo root (gazecam env):
    python robot/experiment.py --subject S01

Saved in outputs_robot/experiments/<date_time>_<subject>/ (git-ignored):
    trial_001/m1_footprint/  person.jpg, face_input.jpg, face_surroundings.jpg  (input: around the person's face)
                             target.jpg, target_overlay.jpg                     (picture after the robot moved)
                             viz.npz, viz_target.npz, view3d.html, view3d_target.html, fig*.png, summary.png  (3D map)
                             result.json
    trials.csv    one row per method per trial
    summary.csv   how many times each method worked
"""
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from robot import gaze_demo as D  # noqa: E402

METHODS = ("footprint", "parallel", "raycast")       # same names as gaze_brain.METHODS (that module needs the GPU models)
TITLES = {"footprint": "Method 1/3: footprint x FOV", "parallel": "Method 2/3: camera parallel to gaze",
          "raycast": "Method 3/3: aim at gaze-ray hit point"}
DEFAULT_MAP = "data/captures/20261116_1857_Room_60_degree"
DEFAULT_ANCHOR = "data/scenes/Person1/scenario_1_2m/testing_1scenario_2m/person_01.jpg"
FIELDS = ["subject", "trial", "target", "method_no", "method", "time", "object_in_view", "cross_on_object",
          "footprint_before", "footprint_after", "focus", "turn_pan", "turn_tilt", "robot_pan", "robot_tilt",
          "limited_by_arm", "arm_reached", "gaze_yaw", "gaze_pitch", "cone_half", "gaze_hit_found", "folder"]


def save_face(person, bbox, out):
    """The input around the person's face: exactly the detected face box, and the box with its surroundings."""
    H, W = person.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1

    def crop(scale):
        a, b = max(int(cx - scale * bw / 2), 0), max(int(cy - scale * bh / 2), 0)
        return person[b:min(int(cy + scale * bh / 2), H), a:min(int(cx + scale * bw / 2), W)]
    cv2.imwrite(os.path.join(out, "face_input.jpg"), crop(1.0))
    cv2.imwrite(os.path.join(out, "face_surroundings.jpg"), crop(3.0))


def ask_key(img, lines, keys):
    """Show a question on the result image; returns the key pressed (one of `keys`, as a character)."""
    cv2.imshow("robot", D.overlay(img.copy(), lines))
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k < 128 and chr(k).lower() in keys:
            return chr(k).lower()


def write_tables(root, rows):
    with open(os.path.join(root, "trials.csv"), "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader(); wr.writerows(rows)
    out = []
    for no, m in enumerate(METHODS, 1):
        r = [x for x in rows if x["method"] == m]
        n = len(r)
        rate = lambda key: round(sum(x[key] for x in r) / n, 3) if n else ""                    # noqa: E731
        mean = lambda key: round(float(np.mean([x[key] for x in r])), 3) if n else ""           # noqa: E731
        out.append(dict(method_no=no, method=m, trials=n, object_in_view=sum(x["object_in_view"] for x in r),
                        object_in_view_rate=rate("object_in_view"), cross_on_object=sum(x["cross_on_object"] for x in r),
                        cross_on_object_rate=rate("cross_on_object"), mean_footprint_after=mean("footprint_after"),
                        mean_focus=mean("focus"), times_limited_by_arm=sum(x["limited_by_arm"] for x in r)))
    with open(os.path.join(root, "summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        wr.writeheader(); wr.writerows(out)
    return out


def print_summary(table):
    print("\n method      trials  object in view   cross on object   mean footprint seen   limited by arm")
    for s in table:
        if s["trials"]:
            print(f" {s['method']:10s}  {s['trials']:5d}  {s['object_in_view']:3d} ({s['object_in_view_rate']:.0%})        "
                  f"{s['cross_on_object']:3d} ({s['cross_on_object_rate']:.0%})          {s['mean_footprint_after']:.0%}"
                  f"                  {s['times_limited_by_arm']}")


def run_method(no, method, trial, target, trial_dir, ctx, delay):
    """One method once. Returns its table row; None if the operator pressed ESC."""
    arm, cam, brain, K, m = ctx["arm"], ctx["cam"], ctx["brain"], ctx["K"], ctx["map"]
    print(f"--- Trial {trial}, {TITLES[method]} (target: {target})")
    arm.home()
    pan0, tilt0 = D.find_and_face(arm, cam, brain)
    person = D.countdown_photo(cam, delay, f"Trial {trial}  {TITLES[method]}  (press SPACE)")
    if person is None:
        return None
    out = os.path.join(trial_dir, f"m{no}_{method}")
    os.makedirs(out, exist_ok=True)
    cv2.imwrite(os.path.join(out, "person.jpg"), person)

    joints = arm.status()["joints"]
    fk = ctx["fk"](joints)
    plan = brain.plan_map(m["dir"], m["K"], person, K, cam.hfov, cam.vfov, pan0=pan0, tilt0=tilt0,
                          pan_limits=D.PAN_LIMITS, anchor=m["anchor"], K_anchor=m["K_anchor"], viz=True, method=method,
                          arm={"radius_m": fk["radius"], "height_m": fk["height"], "cam_elevation_deg": fk["elevation_deg"]})
    viz = plan.pop("viz")
    np.savez_compressed(os.path.join(out, "viz.npz"), **{k: np.asarray(v) for k, v in viz.items() if v is not None})
    save_face(person, viz["bbox"], out)
    print(f"[{method}] gaze yaw {plan['yaw']:+.0f} pitch {plan['pitch']:+.0f} cone ±{plan['cone_half']:.0f} -> turn pan "
          f"{plan['turn_pan']:+.0f} tilt {plan['turn_tilt']:+.0f}; footprint {plan['captured_without_turning']:.0%} -> "
          f"{plan['captured']:.0%}{'  (LIMITED BY THE ARM RANGE)' if plan['limited_by_arm'] else ''}")

    moved = arm.pantilt(plan["robot_pan"], plan["robot_tilt"], speed=15)
    time.sleep(1.0)                                       # let the arm settle: no motion blur
    shot = cam.read()
    cv2.imwrite(os.path.join(out, "target.jpg"), shot)
    tgt, over = brain.target(shot, K, out_dir=out, viz=True)
    vt = tgt.pop("viz", None)
    if vt:
        np.savez_compressed(os.path.join(out, "viz_target.npz"), **{k: np.asarray(x) for k, x in vt.items()})
    cv2.imwrite(os.path.join(out, "target_overlay.jpg"), over)

    in_view = ask_key(shot, [TITLES[method], "Is the target object in this picture?  Y / N"], "yn") == "y"
    on_cross = in_view and ask_key(over, ["Is the green cross on the object?  Y / N"], "yn") == "y"
    plan.update(target=tgt, subject=ctx["subject"], trial=trial, looked_at=target, person_robot_pan=pan0,
                person_robot_tilt=tilt0, map=m["dir"], anchor=m["anchor"], arm_reached=bool(moved["reached"]),
                object_in_view=in_view, cross_on_object=on_cross)
    json.dump(plan, open(os.path.join(out, "result.json"), "w", encoding="utf-8"), indent=1, default=float, ensure_ascii=False)
    ctx["pending"].append((out, target))
    return dict(subject=ctx["subject"], trial=trial, target=target, method_no=no, method=method,
                time=time.strftime("%Y-%m-%d %H:%M:%S"), object_in_view=int(in_view), cross_on_object=int(on_cross),
                footprint_before=round(plan["captured_without_turning"], 3), footprint_after=round(plan["captured"], 3),
                focus=round(plan["focus"], 3), turn_pan=round(plan["turn_pan"], 1), turn_tilt=round(plan["turn_tilt"], 1),
                robot_pan=round(plan["robot_pan"], 1), robot_tilt=round(plan["robot_tilt"], 1),
                limited_by_arm=int(plan["limited_by_arm"]), arm_reached=int(moved["reached"]),
                gaze_yaw=round(plan["yaw"], 1), gaze_pitch=round(plan["pitch"], 1), cone_half=round(plan["cone_half"], 1),
                gaze_hit_found=int(plan["gaze_hit_found"]), folder=os.path.relpath(out, ctx["root"]).replace("\\", "/"))


def make_reports(pending):
    """Figures and 3D pages, drawn at the end so the subject does not wait between methods."""
    from robot import report
    for i, (out, target) in enumerate(pending, 1):
        print(f"[report {i}/{len(pending)}] {out}")
        try:
            report.make(out, target)
        except Exception as e:                            # the raw data is already saved: python robot/report.py <folder>
            print(f"   report failed ({type(e).__name__}: {e}); the data is still in {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S01", help="subject code (use a code, not a real name)")
    ap.add_argument("--brain", default="172.28.95.246", help="IP of the GPU PC running robot/brain_server.py")
    ap.add_argument("--map", default=DEFAULT_MAP)
    ap.add_argument("--map-fov-short", type=float, default=55.9)
    ap.add_argument("--anchor", default=DEFAULT_ANCHOR)
    ap.add_argument("--delay", type=float, default=0.5, help="seconds between SPACE and the photo")
    a = ap.parse_args()

    from robot.arm_client import Arm
    from robot.brain_client import RemoteBrain
    from robot.piper_fk import camera as fk_camera
    from robot.realsense_cam import RealSense
    arm = Arm()
    if arm.status()["home"] is None:
        sys.exit("No home pose yet: run robot/level_home.py first.")
    brain = RemoteBrain(a.brain)
    cam = RealSense()
    photos = sorted(f for f in os.listdir(os.path.join(HERE, a.map)) if f.lower().endswith(".jpg"))
    root = os.path.join(HERE, "outputs_robot", "experiments", f"{D.stamp()}_{a.subject}")
    os.makedirs(root, exist_ok=True)
    ctx = dict(arm=arm, cam=cam, brain=brain, K=D.K_of(cam.intr), fk=fk_camera, subject=a.subject, root=root, pending=[],
               map=dict(dir=D.rel(os.path.join(HERE, a.map)), K=D.phone_K(os.path.join(HERE, a.map, photos[0]), a.map_fov_short),
                        anchor=D.rel(os.path.join(HERE, a.anchor)), K_anchor=D.phone_K(os.path.join(HERE, a.anchor))))
    rows, trial, stop = [], 0, False
    print(f"Saving to {root}")
    try:
        while not stop:
            target = input(f"\nTrial {trial + 1}: which object will the subject look at? (q = finish): ").strip()
            if target.lower() == "q":
                break
            trial += 1
            trial_dir = os.path.join(root, f"trial_{trial:03d}")
            for no, method in enumerate(METHODS, 1):
                while True:                               # R = repeat this method after a failure
                    try:
                        row = run_method(no, method, trial, target or "the object", trial_dir, ctx, a.delay)
                        break
                    except Exception as e:
                        print(f"[{method}] failed: {type(e).__name__}: {e}")
                        blank = np.zeros((720, 1280, 3), np.uint8)
                        if ask_key(blank, [f"{method} failed (see PowerShell)", "R = repeat, S = skip this method"], "rs") == "s":
                            row = "skip"
                            break
                if row is None:                           # ESC
                    stop = True
                    break
                if row != "skip":
                    rows.append(row)
                    write_tables(root, rows)              # saved after every method: nothing is lost if something crashes
            if rows:
                print_summary(write_tables(root, rows))
    finally:
        try:
            arm.home()
        finally:
            cam.stop(); cv2.destroyAllWindows()
    make_reports(ctx["pending"])
    if rows:
        print_summary(write_tables(root, rows))
    print(f"\nAll saved in {root}")


if __name__ == "__main__":
    main()
