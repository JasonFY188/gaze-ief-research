"""
active_camera_room.py - real-room version of the active-camera idea.

Inputs: one person photo (gaze) + a Turn Capture circle (room photos taken while turning on one spot).
1. VGGT puts all photos in one 3D frame (room points, camera poses).
2. L2CS on the person photo -> eye position (VGGT depth at the face) + gaze direction + q95 cone.
3. Gaze footprint = room surface points inside the soft cone that the eye can see.
4. A camera standing at the Turn Capture spot: search pan/tilt (and zoom) that covers the footprint best.
5. Check with reality: footprint drawn onto the real photos taken at that spot.

Usage (from sotsuron):
  ..\\.venv\\Scripts\\python.exe -I active_camera_room.py --person data\\scenes\\Person1\\scenario_1_2m\\testing_1scenario_2m\\person_01.jpg ^
      --room ..\\data\\captures\\20261116_1904_Room_30_degree
Outputs: outputs_active\\room\\
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

import active_camera as A  # noqa: E402

NET_W, NET_H = 392, 518          # portrait 3:4, multiples of 14, NO cropping (VGGT's default crop cuts portrait photos)


def load_net_images(paths):
    ims, origs = [], []
    for p in paths:
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        if im.width > im.height:
            raise SystemExit(f"{p}: this script expects portrait photos")
        origs.append(np.array(im))
        ims.append(torch.from_numpy(np.array(im.resize((NET_W, NET_H), Image.Resampling.BICUBIC))).permute(2, 0, 1).float() / 255.0)
    return torch.stack(ims), origs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True)
    ap.add_argument("--room", required=True, help="Turn Capture session folder")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs_active", "room"))
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--stride", type=int, default=2, help="use every n-th VGGT point")
    ap.add_argument("--conf_keep", type=float, default=0.6, help="keep this share of most confident points per view")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"

    import final_form as F                                   # loads after argparse (heavy imports)
    from l2cs import Pipeline
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    room_paths = sorted(glob.glob(os.path.join(args.room, "*.jpg")))
    poses = {json.loads(l)["file"]: json.loads(l) for l in open(os.path.join(args.room, "poses.jsonl"), encoding="utf-8")}
    paths = room_paths + [args.person]                       # room yaw000 first = VGGT world frame
    nR, iP = len(room_paths), len(room_paths)
    print(f"{nR} room photos + 1 person photo")

    # ---------------- 1. VGGT
    images, origs = load_net_images(paths)
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(images.to(dev))
    extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    extr, intr = extr[0].float().cpu().numpy(), intr[0].float().cpu().numpy()
    wp = pred["world_points"][0].float().cpu().numpy()                 # (S,H,W,3)
    wconf = pred["world_points_conf"][0].float().cpu().numpy()          # (S,H,W)
    depth = pred["depth"][0, ..., 0].float().cpu().numpy()              # (S,H,W)
    del model, pred
    torch.cuda.empty_cache()
    centers = np.stack([-e[:3, :3].T @ e[:3, 3] for e in extr])
    fwd = np.stack([e[2, :3] for e in extr])                            # camera z axis in world
    spread = np.linalg.norm(centers[:nR] - centers[:nR].mean(0), axis=1)
    print(f"room camera centres spread (VGGT units): median {np.median(spread):.3f}")

    # up = axis the room cameras turned about (their forward vectors span the horizontal plane)
    _, _, vt = np.linalg.svd(fwd[:nR] - 0 * fwd[:nR].mean(0))
    up = vt[2] if vt[2] @ (-extr[0][1, :3]) > 0 else -vt[2]             # agree with image-up of photo 0
    cam = centers[:nR].mean(0)
    ref = fwd[0]                                                        # pan 0 = direction of yaw000
    hfov_room = float(np.degrees(2 * np.arctan(NET_W / (2 * intr[0][0, 0]))))
    print(f"room camera horizontal FOV from VGGT: {hfov_room:.1f}°")

    # ---------------- 2. L2CS on the person photo
    l2cs = Pipeline(weights=os.path.join(HERE, "models", "L2CSNet_gaze360.pkl"), arch="ResNet50", device=torch.device(dev))
    bgr = origs[iP][:, :, ::-1].copy()
    res = l2cs.step(bgr)
    bbox = np.array(res.bboxes[0], dtype=np.float32)
    face_t, _ = F.preprocess_face_for_l2cs_with_vis(bgr, bbox, torch.device(dev))
    with torch.inference_mode():
        yl, pl = l2cs.model(face_t)
    yp, pp = torch.softmax(yl, 1).cpu().numpy()[0], torch.softmax(pl, 1).cpu().numpy()[0]
    bins = np.arange(F.NUM_BINS, dtype=np.float32) * F.BIN_WIDTH_DEG + F.BIN_START_DEG
    yaw, pitch = F.expected_angle_deg(yp, bins), F.expected_angle_deg(pp, bins)
    half = F.cone_half_angle_from_probs(yp, pp, bins, 4000, args.quantile, F.CONE_MIN_DEG, F.CONE_MAX_DEG)
    eye, _ = F.estimate_eye_origin_world(bbox, wp[iP], origs[iP].shape[:2], NET_H, NET_W, intr[iP], extr[iP])
    d_cam = F.gaze_vector_from_angles(np.deg2rad(yaw), np.deg2rad(pitch))
    gaze = extr[iP][:3, :3].T @ d_cam
    gaze /= np.linalg.norm(gaze)
    print(f"L2CS: yaw {yaw:.1f}°, pitch {pitch:.1f}°, q{int(args.quantile * 100)} cone half-angle {half:.1f}°")

    # ---------------- 3. room points + gaze footprint
    s = args.stride
    P, area, src = [], [], []
    for v in range(nR):
        pts = wp[v, ::s, ::s].reshape(-1, 3)
        c = wconf[v, ::s, ::s].reshape(-1)
        keep = (c >= np.quantile(c, 1 - args.conf_keep)) & np.isfinite(pts).all(1)
        a = A.point_areas_from_depth(depth[v, ::s, ::s].reshape(-1), intr[v][0, 0] / s, intr[v][1, 1] / s)
        P.append(pts[keep]); area.append(a[keep]); src.append(np.full(keep.sum(), v))
    P, area, src = np.concatenate(P).astype(np.float32), np.concatenate(area).astype(np.float32), np.concatenate(src)
    w = A.gaze_weights(P, eye, gaze, half, z_edge=1.96, up=up, device=dev)
    print(f"{len(P)} room points, {int((w > 0.05).sum())} in the gaze footprint")
    if w.sum() <= 0:
        raise SystemExit("gaze cone hits no room point: check the eye position / gaze direction figure")

    reach = A.reachable_from(P, cam, up, device=dev)
    reach_mass = float((w * area * reach).sum() / (w * area).sum())
    print(f"share of the footprint visible from the camera spot at all: {reach_mass:.1%}")

    # ---------------- 4. search
    kw = dict(width=NET_W, height=NET_H, area=area, device=dev, tilt_range=(-45, 45))
    best = A.best_pan_tilt(P, w, cam, ref, up, hfov_deg=hfov_room, **kw)
    zooms = A.best_view_with_zoom(P, w, cam, ref, up, hfovs=[hfov_room, 40, 30, 20], **kw)
    zbest = A.pick_zoom(zooms, reach_mass, 0.95)

    # the real photos: pan = measured yaw, tilt ~ 0
    Pt, wt, at = (torch.as_tensor(x, device=dev) for x in (P, w, area))
    K = A.intrinsics_from_hfov(hfov_room, NET_W, NET_H)
    real = []
    for v, p in enumerate(room_paths):
        R = torch.as_tensor(extr[v][None, :3, :3], dtype=torch.float32, device=dev)   # the photo's real rotation
        m, foc, iou = A._score(Pt, wt, at, centers[v], K, NET_W, NET_H, R)
        real.append((os.path.basename(p), poses.get(os.path.basename(p), {}).get("yaw"), float(m[0]), float(foc[0])))
    print("\nreal photos taken at the spot (measured yaw -> share of gaze footprint in the photo):")
    for name, yw, m, foc in real:
        print(f"  {name:28s} yaw {yw:7.2f}  captured {m:6.1%}  focus {foc:5.1%}")
    print(f"\nsearched view ({hfov_room:.0f}°): pan {best.pan_deg:+.0f}°, tilt {best.tilt_deg:+.0f}°, "
          f"captured {best.captured_mass:.1%}, focus {best.focus:.1%}")
    print(f"with zoom: hfov {zbest.hfov_deg:.0f}°, pan {zbest.pan_deg:+.0f}°, tilt {zbest.tilt_deg:+.0f}°, "
          f"captured {zbest.captured_mass:.1%}, focus {zbest.focus:.1%}")
    best_real = max(real, key=lambda r: r[2])
    print(f"best real photo: {best_real[0]} (captured {best_real[2]:.1%})")

    # ---------------- 5. figures
    e1 = ref - (ref @ up) * up; e1 /= np.linalg.norm(e1); e2 = np.cross(e1, up)       # top view axes
    top = lambda X: np.stack([X @ e2, X @ e1], -1)                                     # noqa: E731
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.5))
    im = origs[iP]
    ax[0].imshow(im)
    x1, y1, x2, y2 = bbox
    ax[0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="lime", lw=2))
    ax[0].set_title(f"Person: L2CS yaw {yaw:.0f}°, pitch {pitch:.0f}°, cone ±{half:.0f}°"); ax[0].axis("off")

    sub = np.random.default_rng(0).choice(len(P), min(60000, len(P)), replace=False)
    T = top(P[sub]); ax[1].scatter(T[:, 0], T[:, 1], s=0.3, c="0.8")
    f = w > 0.05; Tf = top(P[f])
    ax[1].scatter(Tf[:, 0], Tf[:, 1], s=1, c=w[f], cmap="Reds", vmin=0, vmax=1)
    te, tc = top(eye), top(cam)
    ax[1].plot(*te, "ko"); ax[1].annotate("eye", te)
    tg = top(eye + gaze * np.median(np.linalg.norm(P - eye, axis=1)))
    ax[1].annotate("", tg, te, arrowprops=dict(color="r", width=1.5))
    ax[1].plot(*tc, "bs"); ax[1].annotate("camera spot", tc)
    for ch, col in ((best, "tab:blue"), (zbest, "tab:green")):
        for sgn in (-1, 1):
            d = A.forward_from_pan_tilt(ch.pan_deg + sgn * ch.hfov_deg / 2, 0, ref, up)
            L = np.percentile(np.linalg.norm(P - cam, axis=1), 90)
            ax[1].plot(*np.stack([tc, top(cam + L * d)]).T, color=col, lw=1)
    ax[1].set_aspect("equal"); ax[1].set_title("Top view (VGGT units): gaze footprint (red)\nblue = best view, green = with zoom")

    v = [os.path.basename(p) for p in room_paths].index(best_real[0])
    ax[2].imshow(origs[v])
    Pc = (P - centers[v]) @ extr[v][:3, :3].T
    sx, sy = origs[v].shape[1] / NET_W, origs[v].shape[0] / NET_H
    uu = (intr[v][0, 0] * Pc[:, 0] / Pc[:, 2] + intr[v][0, 2]) * sx
    vv = (intr[v][1, 1] * Pc[:, 1] / Pc[:, 2] + intr[v][1, 2]) * sy
    ok = (Pc[:, 2] > 0) & (w > 0.05) & (uu >= 0) & (uu < origs[v].shape[1]) & (vv >= 0) & (vv < origs[v].shape[0])
    ax[2].scatter(uu[ok], vv[ok], s=1, c=w[ok], cmap="Reds", vmin=0, vmax=1, alpha=0.5)
    ax[2].set_title(f"Real photo {best_real[0]}: gaze footprint drawn on it\n(captured {best_real[2]:.0%} of the footprint)")
    ax[2].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "room_result.png"), dpi=100)
    json.dump({"yaw": yaw, "pitch": pitch, "cone_half_deg": half, "reachable": reach_mass,
               "best": {k: getattr(best, k) for k in ("pan_deg", "tilt_deg", "hfov_deg", "captured_mass", "focus")},
               "zoom": {k: getattr(zbest, k) for k in ("pan_deg", "tilt_deg", "hfov_deg", "captured_mass", "focus")},
               "real_photos": real}, open(os.path.join(args.out, "room_result.json"), "w"), indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
