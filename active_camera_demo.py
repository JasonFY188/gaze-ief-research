"""
active_camera_demo.py - synthetic room test of active_camera.py (no photos needed).

Room 6 x 3 x 5 m (y up), a table (occluder), a person's eye looking at the table top,
a camera high in a corner. Compares: camera not turned / pointed at the footprint centre /
searched pan-tilt / searched pan-tilt + zoom. Writes outputs_active/demo.png.
Usage:  ..\\.venv\\Scripts\\python.exe active_camera_demo.py      (from the sotsuron folder)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import active_camera as A  # noqa: E402

STEP = 0.04
UP = np.array([0.0, 1.0, 0.0])


def plane(x, y, z):
    g = np.meshgrid(*[np.arange(a, b + 1e-9, STEP) if a != b else np.array([a]) for a, b in (x, y, z)], indexing="ij")
    return np.stack([c.ravel() for c in g], 1)


def room():
    faces = [plane((0, 6), (0, 0), (0, 5)), plane((0, 6), (3, 3), (0, 5)),       # floor, ceiling
             plane((0, 0), (0, 3), (0, 5)), plane((6, 6), (0, 3), (0, 5)),       # left, right wall
             plane((0, 6), (0, 3), (0, 0)), plane((0, 6), (0, 3), (5, 5)),       # near, far wall
             plane((3, 4.5), (0.8, 0.8), (3, 4)),                                # table top
             plane((3, 4.5), (0, 0.8), (3, 3)), plane((3, 4.5), (0, 0.8), (4, 4)),  # table sides
             plane((3, 3), (0, 0.8), (3, 4)), plane((4.5, 4.5), (0, 0.8), (3, 4))]   # (closed box)
    return np.concatenate(faces).astype(np.float32)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = room()
    eye = np.array([1.5, 1.6, 1.0])
    target = np.array([3.9, 0.8, 3.6])                    # on the table top
    gaze = target - eye
    half = 12.0                                           # e.g. the q95 circle cone of L2CS
    w = A.gaze_weights(P, eye, gaze, half, z_edge=1.96, up=UP, device=dev)
    cam = np.array([5.8, 2.6, 0.2])                       # camera high in the corner
    ref = np.array([-1.0, 0.0, 1.0])                      # camera's current heading (towards room centre)
    W, H = 640, 480
    print(f"{len(P)} surface points, {int((w > 0.05).sum())} in the gaze footprint")
    reach = A.reachable_from(P, cam, UP, device=dev)
    reach_mass = float((w * reach).sum() / w.sum())
    print(f"share of the gaze footprint this camera can see from its position at all: {reach_mass:.1%}")

    Pt = torch.as_tensor(P, device=dev)
    wt = torch.as_tensor(w, device=dev)

    def evaluate(R, hfov):
        K = A.intrinsics_from_hfov(hfov, W, H)
        Rt = torch.as_tensor(R[None], dtype=torch.float32, device=dev)
        m, foc, iou = A._score(Pt, wt, torch.ones_like(wt), cam, K, W, H, Rt)
        return float(m[0]), float(foc[0]), float(iou[0])

    rows = []
    R0 = A.rotation_looking_at(A.forward_from_pan_tilt(0, 0, ref, UP), UP)
    rows.append(("not turned (60°)", 60, R0, *evaluate(R0, 60)))
    Rc = A.view_toward_centroid(P, w, cam, UP)
    rows.append(("point at footprint centre (60°)", 60, Rc, *evaluate(Rc, 60)))
    best = A.best_pan_tilt(P, w, cam, ref, UP, hfov_deg=60, width=W, height=H, device=dev)
    rows.append((f"searched pan {best.pan_deg:+.0f}°, tilt {best.tilt_deg:+.0f}° (60°)", 60, best.R_w2c,
                 best.captured_mass, best.focus, best.footprint_iou))
    zooms = A.best_view_with_zoom(P, w, cam, ref, UP, width=W, height=H, device=dev)
    for z in zooms:
        print(f"  zoom hfov {z.hfov_deg:4.0f}°: pan {z.pan_deg:+5.0f} tilt {z.tilt_deg:+4.0f}  "
              f"captured {z.captured_mass:6.1%}  focus {z.focus:5.1%}  IoU {z.footprint_iou:5.2f}")
    zb = A.pick_zoom(zooms, reach_mass, 0.95)
    rows.append((f"searched + zoom (hfov {zb.hfov_deg:.0f}°)", zb.hfov_deg, zb.R_w2c, zb.captured_mass, zb.focus,
                 zb.footprint_iou))

    print(f"\n{'camera view':42s} captured  focus   IoU")
    for name, _, _, m, foc, iou in rows:
        print(f"{name:42s} {m:7.1%}  {foc:6.1%}  {iou:5.2f}")

    # ---- figure: top view + what the chosen camera sees
    os.makedirs("outputs_active", exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    sub = np.random.default_rng(0).choice(len(P), 20000, replace=False)
    ax[0].scatter(P[sub, 0], P[sub, 2], s=1, c="0.85")
    f = w > 0.05
    ax[0].scatter(P[f, 0], P[f, 2], s=2, c=w[f], cmap="Reds", vmin=0, vmax=1)
    ax[0].plot(*eye[[0, 2]], "ko"); ax[0].annotate("eye", eye[[0, 2]] + [0.1, -0.25])
    ax[0].arrow(eye[0], eye[2], gaze[0] * 0.9, gaze[2] * 0.9, color="r", width=0.02)
    for name, hfov, R, *_ in rows[1:]:
        fwd = R[2]
        for s in (-1, 1):
            d = A.forward_from_pan_tilt(s * hfov / 2, 0, fwd, UP)
            ax[0].plot([cam[0], cam[0] + 7 * d[0]], [cam[2], cam[2] + 7 * d[2]], lw=1,
                       color={60: "tab:blue"}.get(hfov, "tab:green") if "centre" not in name else "0.5", alpha=0.8)
    ax[0].plot(*cam[[0, 2]], "bs"); ax[0].annotate("camera", cam[[0, 2]] + [-0.9, 0.2])
    ax[0].set_xlim(-0.2, 6.2); ax[0].set_ylim(-0.2, 5.2); ax[0].set_aspect("equal")
    ax[0].set_title("Top view: gaze footprint (red), camera field of view\n(grey = pointed at centre, blue = searched 60°, green = with zoom)")
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("z [m]")

    for k, (name, hfov, R, m, foc, iou) in zip((1, 2), (rows[2], rows[3])):
        K = A.intrinsics_from_hfov(hfov, W, H)
        vis = A.visible_from(Pt, cam, torch.as_tensor(R[None], dtype=torch.float32, device=dev), K, W, H)[0].cpu().numpy()
        Pc = (P - cam) @ R.T
        u = K[0, 0] * Pc[:, 0] / Pc[:, 2] + K[0, 2]
        v = K[1, 1] * Pc[:, 1] / Pc[:, 2] + K[1, 2]
        ax[k].scatter(u[vis], v[vis], s=1, c="0.8")
        fv = vis & (w > 0.05)
        ax[k].scatter(u[fv], v[fv], s=2, c=w[fv], cmap="Reds", vmin=0, vmax=1)
        ax[k].set_xlim(0, W); ax[k].set_ylim(H, 0); ax[k].set_aspect("equal")
        ax[k].set_title(f"Camera view: {name}\ncaptured {m:.0%} of gaze footprint, focus {foc:.0%}")
        ax[k].set_xticks([]); ax[k].set_yticks([])
    fig.tight_layout()
    fig.savefig("outputs_active/demo.png", dpi=110)
    print("-> outputs_active/demo.png")


if __name__ == "__main__":
    main()
