"""Presentation record of one robot run: figures, interactive 3D view, and a row in the runs table.

Writes into the run folder (outputs_robot/<date_time>/):
  fig1_gaze.png        person photo, face box, L2CS gaze arrow and cone size
  fig2_topview.png     room from above: gaze footprint (裾野, red), camera view BEFORE and AFTER turning
  fig3_search.png      share of the footprint seen for every pan/tilt; current view and chosen view
  fig4_before_after.png  what the camera saw before turning (person) and after (with the predicted target)
  summary.png          the four figures on one page
  view3d.html          interactive 3D: VGGT room, footprint, gaze cone, robot camera before/after, all photo cameras
and appends one line to outputs_robot/runs.csv.

  python robot/report.py outputs_robot/<run>                     # run saved with viz.npz
  python robot/report.py outputs_robot/<run> --replan <brain IP>  # older run: ask the brain again for the 3D data
  python robot/report.py outputs_robot/<run> --note "black tool cabinet"   # what the person really looked at
"""
import argparse
import csv
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import active_camera as A  # noqa: E402

for f in ("Yu Gothic", "Meiryo", "MS Gothic"):               # Japanese labels (裾野) on Windows
    if any(f in x.name for x in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13})
RUNS_CSV = os.path.join(HERE, "outputs_robot", "runs.csv")


def load_viz(run):
    z = np.load(os.path.join(run, "viz.npz"), allow_pickle=True)
    return {k: z[k] for k in z.files}


def replan(run, brain_ip):
    """Ask the brain to plan again from the saved person photo with the saved robot pose, keeping the 3D data."""
    from robot.brain_client import RemoteBrain
    from robot.gaze_demo import K_of, phone_K
    r = json.load(open(os.path.join(run, "result.json")))
    intr = json.load(open(os.path.join(HERE, "robot", "realsense_intrinsics.json")))
    K = K_of(intr)
    b = RemoteBrain(brain_ip)
    photos = sorted(f for f in os.listdir(os.path.join(HERE, r["map"])) if f.endswith(".jpg"))
    K_map = phone_K(os.path.join(HERE, r["map"], photos[0]), 55.9)
    K_anchor = phone_K(os.path.join(HERE, r["anchor"])) if r.get("anchor") else None
    p = b.plan_map(r["map"], K_map, os.path.join(run, "person.jpg"), K, intr["hfov"], intr["vfov"],
                   pan0=r["person_robot_pan"], tilt0=r["person_robot_tilt"], pan_limits=(-145.0, 145.0),
                   anchor=r.get("anchor"), K_anchor=K_anchor, arm=r.get("arm"), viz=True)
    print(f"[report] replanned: turn pan {p['turn_pan']:+.1f} tilt {p['turn_tilt']:+.1f} "
          f"(saved run: {r['turn_pan']:+.1f} {r['turn_tilt']:+.1f}); captured {p['captured']:.1%} (saved {r['captured']:.1%})")
    np.savez_compressed(os.path.join(run, "viz.npz"), **{k: np.asarray(v) for k, v in p["viz"].items() if v is not None})


def basis(up, ref):
    h0 = ref - (ref @ up) * up; h0 /= np.linalg.norm(h0)
    return h0, np.cross(h0, up)                     # forward (robot's view when it took the photo), right


def fig_gaze(run, r, V, ax):
    import final_form as F
    img = cv2.cvtColor(cv2.imread(os.path.join(run, "person.jpg")), cv2.COLOR_BGR2RGB)
    ax.imshow(img); ax.axis("off")
    x1, y1, x2, y2 = V["bbox"]
    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="lime", lw=2))
    g = F.gaze_vector_from_angles(np.deg2rad(r["yaw"]), np.deg2rad(r["pitch"]))
    cx, cy, L = (x1 + x2) / 2, (y1 + y2) / 2, 3 * (x2 - x1)
    ax.annotate("", (cx + L * g[0], cy + L * g[1]), (cx, cy), arrowprops=dict(color="red", width=3, headwidth=12))
    ax.set_title(f"① L2CS gaze: yaw {r['yaw']:+.0f}°, pitch {r['pitch']:+.0f}°, 95% cone ±{r['cone_half']:.0f}°")


def fig_top(r, V, ax):
    P, w, up, ref, cam, eye, gaze = V["P"], V["w"], V["up"], V["ref"], V["cam"], V["eye"], V["gaze"]
    h0, right = basis(up, ref)
    top = lambda X: np.stack([X @ right, X @ h0], -1)   # noqa: E731  x = right, y = robot's facing direction
    T = top(P)
    ax.scatter(T[:, 0], T[:, 1], s=0.4, c="0.8")
    f = w > 0.05
    ax.scatter(T[f, 0], T[f, 1], s=2, c=w[f], cmap="Reds", vmin=0, vmax=1)
    L = float(np.percentile(np.linalg.norm(P - cam, axis=1), 90))
    te, tc = top(eye), top(cam)
    half = r["cone_half"]
    for a in (-half, 0, half):
        d = A.forward_from_pan_tilt(a, 0, gaze, up)
        ax.plot(*np.stack([te, top(eye + L * d)]).T, "r-" if a == 0 else "r--", lw=2 if a == 0 else 1)
    ax.plot(*te, "ko", ms=7); ax.text(*te, "  eye", fontsize=11)
    hf = float(V["hfov"])
    for pan, col, lab in ((0.0, "tab:blue", f"before: {r['captured_without_turning']:.0%}"),
                          (r["turn_pan"], "tab:green", f"after turn {r['turn_pan']:+.0f}°: {r['captured']:.0%}")):
        for e in (-hf / 2, hf / 2):
            d = A.forward_from_pan_tilt(pan + e, 0, ref, up)
            ax.plot(*np.stack([tc, top(cam + 0.8 * L * d)]).T, color=col, lw=2)
        d = A.forward_from_pan_tilt(pan, 0, ref, up)
        ax.text(*top(cam + 0.45 * L * d), lab, color=col, fontsize=11, ha="center", fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    ax.plot(*tc, "s", color="navy", ms=9); ax.text(*tc, "  robot camera", color="navy", fontsize=11)
    for c in V["room_centers"]:
        ax.plot(*top(c), "^", color="tab:purple", ms=5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("② from above: gaze footprint (裾野, red) vs camera view (FOV)\n"
                 "blue = before turning, green = after; purple ▲ = iPhone map photos")


def fig_search(r, V, ax, fig):
    pans, tilts, G = V["grid_pans"], V["grid_tilts"], V["grid"]
    st = float(pans[1] - pans[0]) if len(pans) > 1 else 5.0
    im = ax.imshow(100 * G.T, origin="lower", aspect="auto", cmap="viridis",
                   extent=[pans[0] - st / 2, pans[-1] + st / 2, tilts[0] - st / 2, tilts[-1] + st / 2])
    fig.colorbar(im, ax=ax, label="% of the footprint inside the camera view")
    en = float(V["elev_now"])
    ax.plot(0, en, "wo", ms=11, mfc="none", mew=2.5, label=f"before (now): {r['captured_without_turning']:.0%}")
    ax.plot(r["turn_pan"], en + r["turn_tilt"], "r*", ms=18, label=f"chosen: {r['captured']:.0%}")
    ax.set_xlabel("pan (deg, + = right of where the robot faced)"); ax.set_ylabel("camera elevation (deg, + = up)")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_title(f"③ search: every direction scored ({r['secs_total']:.1f} s on the GPU PC incl. VGGT)")


def fig_before_after(run, axes):
    for ax, name, title in ((axes[0], "person.jpg", "before turning (robot faces the person)"),
                            (axes[1], "target_overlay.jpg", "after turning: heat map, cone (white), target (+)")):
        p = os.path.join(run, name)
        if os.path.exists(p):
            ax.imshow(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
        ax.axis("off"); ax.set_title(title)


def view3d(run, r, V):
    import plotly.graph_objects as go
    from view_3d import frustum, lines_trace
    P, w, rgb, up, ref, cam, eye, gaze = V["P"], V["w"], V["rgb"] / 255.0, V["up"], V["ref"], V["cam"], V["eye"], V["gaze"]
    hf, vf, en = float(V["hfov"]), float(V["vfov"]), float(V["elev_now"])
    L = float(np.percentile(np.linalg.norm(P - cam, axis=1), 90))
    a = np.clip(w / max(w.max(), 1e-9), 0, 1)[:, None] * 0.85
    col = (1 - a) * rgb + a * np.array([1.0, 0.05, 0.05])
    cs = [f"rgb({int(255 * x)},{int(255 * y)},{int(255 * z)})" for x, y, z in col]
    tr = [go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", name="VGGT room (photo colours, footprint red)",
                       marker=dict(size=1.8, color=cs), hoverinfo="skip"),
          go.Scatter3d(x=[eye[0]], y=[eye[1]], z=[eye[2]], mode="markers+text", text=["eye"], name="eye",
                       marker=dict(size=7, color="black")),
          lines_trace([(eye, eye + 1.1 * L * gaze)], "gaze ray", "red", 7)]
    u = np.cross(gaze, up); u /= np.linalg.norm(u); v = np.cross(u, gaze)
    h = np.radians(r["cone_half"])
    tr.append(lines_trace([(eye, eye + 1.1 * L * (np.cos(h) * gaze + np.sin(h) * (np.cos(t) * u + np.sin(t) * v)))
                           for t in np.radians(np.arange(0, 360, 30))], f"gaze cone ±{r['cone_half']:.0f}°",
                          "rgba(255,0,0,0.45)", 2, "dash"))
    tr.append(go.Scatter3d(x=[cam[0]], y=[cam[1]], z=[cam[2]], mode="markers+text", text=["robot camera"],
                           name="robot camera", marker=dict(size=7, color="navy", symbol="square")))
    for pan, tilt, c, lab in ((0.0, en, "blue", f"before: {r['captured_without_turning']:.0%} of footprint"),
                              (r["turn_pan"], en + r["turn_tilt"], "green", f"after turning: {r['captured']:.0%}")):
        R = A.rotation_looking_at(A.forward_from_pan_tilt(pan, tilt, ref, up), up)
        tr.append(lines_trace(frustum(cam, R, hf, vf, 0.8 * L), lab, c, 5))
    tr.append(lines_trace([(c, c + 0.25 * L * f) for c, f in zip(V["room_centers"], V["room_fwd"])],
                          "iPhone map photos (where VGGT put them)", "purple", 3))
    if "anchor_center" in V and V["anchor_center"] is not None and np.asarray(V["anchor_center"]).size == 3:
        ac = np.asarray(V["anchor_center"], float)
        tr.append(go.Scatter3d(x=[ac[0]], y=[ac[1]], z=[ac[2]], mode="markers+text", text=["iPhone person photo"],
                               name="anchor (iPhone person photo)", marker=dict(size=6, color="purple")))
    fig = go.Figure(tr)
    fig.update_layout(title=f"{os.path.basename(run)}: VGGT 3D room, gaze footprint, robot camera before/after turning",
                      scene=dict(aspectmode="data", xaxis=dict(visible=False), yaxis=dict(visible=False),
                                 zaxis=dict(visible=False),
                                 camera=dict(up=dict(x=float(up[0]), y=float(up[1]), z=float(up[2])))),
                      legend=dict(itemsizing="constant", bgcolor="rgba(255,255,255,0.8)"), margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(os.path.join(run, "view3d.html"), include_plotlyjs=True, full_html=True)


def append_csv(run, r, note):
    row = dict(run=os.path.basename(run), yaw=round(r["yaw"], 1), pitch=round(r["pitch"], 1),
               cone_half=round(r["cone_half"], 1), robot_pan_at_person=round(r["person_robot_pan"], 1),
               turn_pan=round(r["turn_pan"], 1), turn_tilt=round(r["turn_tilt"], 1),
               footprint_seen_before=round(r["captured_without_turning"], 3), footprint_seen_after=round(r["captured"], 3),
               focus_after=round(r["focus"], 3), robot_cam_to_anchor=round(r.get("robot_cam_vggt_to_anchor") or 0, 3),
               target_px=r.get("target", {}).get("peak_px"), plan_secs=r["secs_total"],
               target_secs=r.get("target", {}).get("secs"), looked_at=note or "")
    rows = []
    if os.path.exists(RUNS_CSV):
        rows = [x for x in csv.DictReader(open(RUNS_CSV, encoding="utf-8-sig")) if x["run"] != row["run"]]
    rows.append(row)
    with open(RUNS_CSV, "w", newline="", encoding="utf-8-sig") as f:      # utf-8-sig: Excel shows Japanese correctly
        wr = csv.DictWriter(f, fieldnames=list(row.keys()))
        wr.writeheader(); wr.writerows(rows)


def make(run, note=None):
    r = json.load(open(os.path.join(run, "result.json")))
    if note:
        r["looked_at"] = note
        json.dump(r, open(os.path.join(run, "result.json"), "w"), indent=1, default=float)
    note = note or r.get("looked_at")
    V = load_viz(run)
    for name, draw in (("fig1_gaze", lambda ax, fg: fig_gaze(run, r, V, ax)), ("fig2_topview", lambda ax, fg: fig_top(r, V, ax)),
                       ("fig3_search", lambda ax, fg: fig_search(r, V, ax, fg))):
        fg, ax = plt.subplots(figsize=(10, 7.5))
        draw(ax, fg); fg.tight_layout(); fg.savefig(os.path.join(run, name + ".png"), dpi=110); plt.close(fg)
    fg, axs = plt.subplots(1, 2, figsize=(18, 6))
    fig_before_after(run, axs); fg.tight_layout(); fg.savefig(os.path.join(run, "fig4_before_after.png"), dpi=100)
    plt.close(fg)
    fg = plt.figure(figsize=(22, 15))
    fig_gaze(run, r, V, fg.add_subplot(2, 2, 1)); fig_top(r, V, fg.add_subplot(2, 2, 2))
    fig_search(r, V, fg.add_subplot(2, 2, 3), fg)
    ax = fg.add_subplot(2, 2, 4); fig_before_after(run, [fg.add_subplot(4, 2, 6), fg.add_subplot(4, 2, 8)]); ax.axis("off")
    fg.suptitle(f"Active camera on PiPER: turn {r['turn_pan']:+.0f}° pan, {r['turn_tilt']:+.0f}° tilt  ->  footprint seen "
                f"{r['captured_without_turning']:.0%} before, {r['captured']:.0%} after"
                + (f"   (person looked at: {note})" if note else ""), fontsize=16)
    fg.tight_layout(); fg.savefig(os.path.join(run, "summary.png"), dpi=90); plt.close(fg)
    view3d(run, r, V)
    append_csv(run, r, note)
    print(f"[report] {run}: fig1-4, summary.png, view3d.html; row in {RUNS_CSV}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--replan", metavar="BRAIN_IP")
    ap.add_argument("--note", help="what the person really looked at")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.replan:
        replan(a.run, a.replan)
    make(a.run, a.note)


if __name__ == "__main__":
    main()
