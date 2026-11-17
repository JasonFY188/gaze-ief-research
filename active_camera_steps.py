"""
active_camera_steps.py - the active-camera idea on the real room, one figure per step, for several
Turn Capture circles (e.g. 60°, 30°, 20°), plus a comparison.

Steps saved per circle (outputs_active/steps/<circle>/):
  step1_inputs.png    the photos that go in
  step2_vggt.png      VGGT: 3D room seen from above, where each photo was taken and which way it faced
  step3_l2cs.png      L2CS: face, predicted gaze drawn on the photo, yaw/pitch probabilities and the 95% cone
  step4_footprint.png gaze cone in 3D and its footprint (where the cone lands on surfaces the eye can see)
  step5_search.png    the search: captured share for every pan/tilt tried, and the chosen view
  step6_real.png      check with the real photos: captured share per real photo, best real photo with footprint
compare.png (outputs_active/steps/): the three circles side by side.

Usage (from sotsuron):
  ..\\.venv\\Scripts\\python.exe -I active_camera_steps.py --person data\\scenes\\Person1\\scenario_1_2m\\testing_1scenario_2m\\person_01.jpg ^
      --rooms ..\\data\\captures\\20261116_1857_Room_60_degree ..\\data\\captures\\20261116_1904_Room_30_degree ..\\data\\captures\\20261116_1906_Room_20_degree
"""
import argparse
import time
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

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13})
NET_W, NET_H = 392, 518                      # portrait 3:4, multiple of 14, no cropping
DEV = "cuda"
COARSE, FINE = 5.0, 1.0                      # search steps (degrees)
TILTS = (-45.0, 45.0)
AIM_SIGMA = 3.0                              # assumed aim error of the turned camera (deg, 1 sigma)


def load(paths):
    ims, origs = [], []
    for p in paths:
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        origs.append(np.array(im))
        ims.append(torch.from_numpy(np.array(im.resize((NET_W, NET_H), Image.Resampling.BICUBIC))).permute(2, 0, 1).float() / 255.0)
    return torch.stack(ims), origs


def person_K(path):
    """Network-size intrinsics of a phone photo from its EXIF 35 mm-equivalent focal length (None if missing)."""
    im = Image.open(path)
    f35 = im.getexif().get_ifd(0x8769).get(0xA405)
    if not f35:
        return None
    W, H = ImageOps.exif_transpose(im).size
    f_px = float(f35) * np.hypot(W, H) / np.hypot(36.0, 24.0)       # 35 mm equivalent is defined on the diagonal
    return np.array([[f_px * NET_W / W, 0, NET_W / 2], [0, f_px * NET_H / H, NET_H / 2], [0, 0, 1.0]])


def project(X, extr, intr, img_shape):
    """3D points -> pixel coords in the ORIGINAL photo (net intrinsics scaled up)."""
    Pc = (X - (-extr[:3, :3].T @ extr[:3, 3])) @ extr[:3, :3].T
    sx, sy = img_shape[1] / NET_W, img_shape[0] / NET_H
    z = np.maximum(Pc[:, 2], 1e-6)
    return (intr[0, 0] * Pc[:, 0] / z + intr[0, 2]) * sx, (intr[1, 1] * Pc[:, 1] / z + intr[1, 2]) * sy, Pc[:, 2]


def run_circle(room_dir, person, model, l2cs, F, out_root, quantile=0.95, stride=2, conf_keep=0.6, fov_short=None):
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    name = os.path.basename(room_dir.rstrip("/\\"))
    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    room_paths = sorted(glob.glob(os.path.join(room_dir, "*.jpg")))
    poses = {}
    pj = os.path.join(room_dir, "poses.jsonl")
    if os.path.exists(pj):
        for line in open(pj, encoding="utf-8"):
            d = json.loads(line)
            poses[d["file"]] = d
    paths = room_paths + [person]
    nR, iP = len(room_paths), len(room_paths)
    images, origs = load(paths)
    print(f"\n=== {name}: {nR} room photos + person photo")

    # ---------- step 1: inputs
    cols = min(nR, 9)
    rows_ = int(np.ceil(nR / cols)) + 1
    fig, ax = plt.subplots(rows_, cols, figsize=(1.9 * cols, 2.6 * rows_))
    ax = np.atleast_2d(ax)
    for a in ax.ravel():
        a.axis("off")
    for k, p in enumerate(room_paths):
        a = ax[k // cols, k % cols]
        a.imshow(origs[k]); a.set_title(f"yaw {poses.get(os.path.basename(p), {}).get('yaw', '?')}°", fontsize=10)
    ax[-1, 0].imshow(origs[iP]); ax[-1, 0].set_title("person", fontsize=10)
    fig.suptitle(f"Step 1 - inputs: {nR} room photos taken while turning on one spot + 1 person photo")
    fig.tight_layout(); fig.savefig(os.path.join(out, "step1_inputs.png"), dpi=90); plt.close(fig)

    # ---------- VGGT
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(images.to(DEV))
    extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    extr, intr = extr[0].float().cpu().numpy(), intr[0].float().cpu().numpy()
    wp = pred["world_points"][0].float().cpu().numpy()
    wconf = pred["world_points_conf"][0].float().cpu().numpy()
    depth = pred["depth"][0, ..., 0].float().cpu().numpy()
    del pred
    torch.cuda.empty_cache()
    centers = np.stack([-e[:3, :3].T @ e[:3, 3] for e in extr])
    fwd = np.stack([e[2, :3] for e in extr])
    _, _, vt = np.linalg.svd(fwd[:nR])
    up = vt[2] if vt[2] @ (-extr[0][1, :3]) > 0 else -vt[2]
    cam, ref = centers[:nR].mean(0), fwd[0]
    hfov_vggt = float(np.degrees(2 * np.arctan(NET_W / (2 * np.median([k[0, 0] for k in intr[:nR]])))))
    # VGGT's focal guess is ~70 deg for this phone, the measured value ~56 deg (fov_from_rotation.py);
    # VGGT's 3D points themselves span the right angle (fov_check_vggt.py), so only the camera model is replaced.
    hfov = float(fov_short) if fov_short is not None else hfov_vggt
    fx = NET_W / 2 / np.tan(np.deg2rad(hfov / 2))
    K_room = np.array([[fx, 0, NET_W / 2], [0, fx, NET_H / 2], [0, 0, 1.0]])
    vfov = float(np.degrees(2 * np.arctan(NET_H / 2 / fx)))
    K_person = person_K(person) if fov_short is not None else None
    K_person = intr[iP] if K_person is None else K_person
    print(f"room camera FOV {hfov:.1f} x {vfov:.1f} deg ({'measured' if fov_short is not None else 'VGGT guess'}; "
          f"VGGT guessed {hfov_vggt:.1f}); person photo FOV {np.degrees(2 * np.arctan(NET_W / 2 / K_person[0, 0])):.1f} deg")

    s = stride
    P, area, rgb = [], [], []
    for v in range(nR):
        pts = wp[v, ::s, ::s].reshape(-1, 3)
        c = wconf[v, ::s, ::s].reshape(-1)
        keep = (c >= np.quantile(c, 1 - conf_keep)) & np.isfinite(pts).all(1)
        a = A.point_areas_from_depth(depth[v, ::s, ::s].reshape(-1), K_room[0, 0] / s, K_room[1, 1] / s)
        col = images[v].permute(1, 2, 0).numpy()[::s, ::s].reshape(-1, 3)
        P.append(pts[keep]); area.append(a[keep]); rgb.append(col[keep])
    P, area = np.concatenate(P).astype(np.float32), np.concatenate(area).astype(np.float32)
    rgb = (np.concatenate(rgb) * 255).astype(np.uint8)

    e1 = ref - (ref @ up) * up; e1 /= np.linalg.norm(e1); e2 = np.cross(e1, up)
    top = lambda X: np.stack([X @ e2, X @ e1], -1)                                   # noqa: E731
    sub = np.random.default_rng(0).choice(len(P), min(60000, len(P)), replace=False)
    T = top(P[sub])
    L = float(np.percentile(np.linalg.norm(P - cam, axis=1), 90))

    def room_bg(a):
        a.scatter(T[:, 0], T[:, 1], s=0.3, c="0.8")
        a.set_aspect("equal"); a.set_xlabel("right  (VGGT units)"); a.set_ylabel("toward yaw 0°")

    # ---------- step 2: VGGT room and cameras
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    room_bg(ax)
    for v in range(nR):
        pan, _ = A.pan_tilt_of(fwd[v], ref, up)
        c0, c1 = top(centers[v]), top(centers[v] + 0.25 * L * fwd[v])
        ax.annotate("", c1, c0, arrowprops=dict(color="tab:blue", width=1, headwidth=6))
        yw = poses.get(os.path.basename(room_paths[v]), {}).get("yaw")
        ax.text(*c1, f"{yw:.0f}°" if yw is not None else str(v), fontsize=9, color="tab:blue")
    cp0, cp1 = top(centers[iP]), top(centers[iP] + 0.25 * L * fwd[iP])
    ax.annotate("", cp1, cp0, arrowprops=dict(color="tab:purple", width=1.5, headwidth=8))
    ax.text(*cp0, " person photo\n camera", color="tab:purple", fontsize=10)
    ax.set_title(f"Step 2 - VGGT: room from above, {nR} room cameras (blue, labelled with the\n"
                 f"phone's measured yaw) and the person-photo camera (purple). Room camera FOV {hfov:.0f}°")
    fig.tight_layout(); fig.savefig(os.path.join(out, "step2_vggt.png"), dpi=90); plt.close(fig)

    # ---------- step 3: L2CS
    np.random.seed(0)
    bgr = origs[iP][:, :, ::-1].copy()
    res = l2cs.step(bgr)
    bbox = np.array(res.bboxes[0], dtype=np.float32)
    face_t, face224 = F.preprocess_face_for_l2cs_with_vis(bgr, bbox, torch.device(DEV))
    with torch.inference_mode():
        yl, pl = l2cs.model(face_t)
    yp, pp = torch.softmax(yl, 1).cpu().numpy()[0], torch.softmax(pl, 1).cpu().numpy()[0]
    bins = np.arange(F.NUM_BINS, dtype=np.float32) * F.BIN_WIDTH_DEG + F.BIN_START_DEG
    yaw, pitch = F.expected_angle_deg(yp, bins), F.expected_angle_deg(pp, bins)
    half = F.cone_half_angle_from_probs(yp, pp, bins, 4000, quantile, F.CONE_MIN_DEG, F.CONE_MAX_DEG)
    eye, _ = F.estimate_eye_origin_world(bbox, wp[iP], origs[iP].shape[:2], NET_H, NET_W, K_person, extr[iP])
    gaze = extr[iP][:3, :3].T @ F.gaze_vector_from_angles(np.deg2rad(yaw), np.deg2rad(pitch))
    gaze /= np.linalg.norm(gaze)

    fig = plt.figure(figsize=(16, 6))
    a0 = fig.add_subplot(1, 4, 1); a0.imshow(face224[:, :, ::-1]); a0.axis("off"); a0.set_title("face crop (L2CS input)")
    a1 = fig.add_subplot(1, 4, 2); a1.imshow(origs[iP])
    ray = eye[None] + np.linspace(0, 0.6 * L, 30)[:, None] * gaze[None]
    u, v_, z = project(ray, extr[iP], K_person, origs[iP].shape)
    good = z > 0
    a1.plot(u[good], v_[good], "r-", lw=3)
    x1, y1, x2, y2 = bbox
    a1.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="lime", lw=2))
    a1.set_xlim(0, origs[iP].shape[1]); a1.set_ylim(origs[iP].shape[0], 0); a1.axis("off")
    a1.set_title("predicted gaze ray (red)")
    for k, (pr, mean, lab) in enumerate(((yp, yaw, "yaw"), (pp, pitch, "pitch"))):
        a = fig.add_subplot(1, 4, 3 + k)
        sel = np.abs(bins + 2 - mean) <= 60
        a.bar(bins[sel] + 2, pr[sel], width=3.6, color="0.5")
        a.axvline(mean, color="r", lw=2, label=f"mean {mean:.1f}°")
        a.axvspan(mean - half, mean + half, color="r", alpha=0.12, label=f"cone ±{half:.1f}° (q95)")
        a.set_xlabel(f"{lab} (deg, 4° bins)"); a.set_ylabel("probability"); a.legend(fontsize=10)
        a.set_title(f"L2CS {lab} probabilities")
    fig.suptitle(f"Step 3 - L2CS: gaze yaw {yaw:.1f}°, pitch {pitch:.1f}°; 95% of sampled gaze directions lie within {half:.1f}° of the mean")
    fig.tight_layout(); fig.savefig(os.path.join(out, "step3_l2cs.png"), dpi=90); plt.close(fig)

    # ---------- step 4: footprint
    w = A.gaze_weights(P, eye, gaze, half, z_edge=1.96, up=up, device=DEV)
    reach = A.reachable_from(P, cam, up, device=DEV)
    reach_mass = float((w * area * reach).sum() / max((w * area).sum(), 1e-12))
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    room_bg(ax[0])
    f = w > 0.05
    Tf = top(P[f]); ax[0].scatter(Tf[:, 0], Tf[:, 1], s=1.5, c=w[f], cmap="Reds", vmin=0, vmax=1)
    te = top(eye)
    for ang in (-half, 0, half):
        d = A.forward_from_pan_tilt(ang, 0, gaze, up)
        ax[0].plot(*np.stack([te, top(eye + L * d)]).T, "r-" if ang == 0 else "r--", lw=2 if ang == 0 else 1)
    ax[0].plot(*te, "ko", ms=8); ax[0].text(*te, "  eye", fontsize=12)
    ax[0].plot(*top(cam), "bs", ms=8); ax[0].text(*top(cam), "  camera spot", fontsize=12, color="b")
    ax[0].set_title(f"from above: gaze ray (red), cone ±{half:.0f}° (dashed),\nfootprint = surfaces in the cone the eye can see (red dots)")
    # side view: distance along gaze vs height
    hgt = P @ up
    along = (P - eye) @ (gaze - (gaze @ up) * up) / np.linalg.norm(gaze - (gaze @ up) * up)
    near_plane = np.abs((P - eye) @ np.cross(gaze, up) / np.linalg.norm(np.cross(gaze, up))) < 0.05 * L
    ax[1].scatter(along[near_plane], hgt[near_plane], s=0.5, c="0.75")
    ff = near_plane & f
    ax[1].scatter(along[ff], hgt[ff], s=2, c=w[ff], cmap="Reds", vmin=0, vmax=1)
    ax[1].plot(0, eye @ up, "ko", ms=8)
    for ang in (-half, 0, half):
        el = np.deg2rad(pitch + ang)
        dvec = np.array([np.cos(np.arcsin(gaze @ up) + np.deg2rad(ang)), np.sin(np.arcsin(gaze @ up) + np.deg2rad(ang))])
        ax[1].plot([0, L * dvec[0]], [eye @ up, eye @ up + L * dvec[1]], "r-" if ang == 0 else "r--", lw=2 if ang == 0 else 1)
    ax[1].set_aspect("equal"); ax[1].set_xlabel("distance along the gaze (VGGT units)"); ax[1].set_ylabel("height")
    ax[1].set_title("side view (thin slice along the gaze): the cone goes down\nand lands on the floor / desks")
    fig.suptitle(f"Step 4 - gaze footprint: {int(f.sum())} surface points; {reach_mass:.0%} of it can be seen "
                 f"from the camera spot at all")
    fig.tight_layout(); fig.savefig(os.path.join(out, "step4_footprint.png"), dpi=90); plt.close(fig)

    # ---------- step 5: search (coarse grid, then fine around the best)
    kw = dict(width=NET_W, height=NET_H, area=area, device=DEV)
    sync = torch.cuda.synchronize
    t0 = time.perf_counter()
    FS = A.FastSearch(P, w, cam, ref, up, area=area, device=DEV)
    sync(); t_prep = time.perf_counter() - t0
    t0 = time.perf_counter()
    b = FS.best(hfov, vfov, tilt_range=TILTS, coarse=COARSE, fine=FINE)
    sync(); t_fast = time.perf_counter() - t0
    br = FS.best(hfov, vfov, tilt_range=TILTS, coarse=COARSE, fine=FINE, aim_sigma_deg=AIM_SIGMA)
    pans, tilts, m_grid = b["coarse_grid"]
    i, j = np.unravel_index(np.argmax(m_grid), m_grid.shape)
    best_pan, best_tilt, best_m, best_foc = b["pan"], b["tilt"], b["mass"], b["focus"]
    # exact check: the old search renders a z-buffer for EVERY direction; same coarse grid
    t0 = time.perf_counter()
    m_old, _, _ = A.score_grid(P, w, cam, ref, up, hfov, pans, tilts, **kw)
    sync(); t_old = time.perf_counter() - t0
    io, jo = np.unravel_index(np.argmax(m_old), m_old.shape)
    check = dict(t_prep=t_prep, t_fast_search=t_fast, t_old_coarse_only=t_old, n_scored=b["n_scored"],
                 grid_max_abs_diff=float(np.abs(m_old - m_grid).max()),
                 grid_corr=float(np.corrcoef(m_old.ravel(), m_grid.ravel())[0, 1]),
                 old_best_coarse=(float(pans[io]), float(tilts[jo]), float(m_old[io, jo])),
                 new_best_coarse=(float(pans[i]), float(tilts[j]), float(m_grid[i, j])),
                 robust=dict(pan=br["pan"], tilt=br["tilt"], mass=br["mass"], expected=br["expected"]),
                 plain_expected=float(b["expected"]))
    # expected share of the plain maximum if the aim is off by AIM_SIGMA (same smoothing as the robust search)
    from scipy.ndimage import gaussian_filter
    fpn = np.arange(best_pan - 15, best_pan + 15 + 1e-6, FINE); ftn = np.arange(best_tilt - 15, best_tilt + 15 + 1e-6, FINE)
    Gn, _ = FS.grid(fpn, ftn, hfov, vfov)
    check["plain_expected"] = float(gaussian_filter(Gn, AIM_SIGMA / FINE, mode="nearest")[15, 15])
    fpn = np.arange(br["pan"] - 15, br["pan"] + 15 + 1e-6, FINE); ftn = np.arange(br["tilt"] - 15, br["tilt"] + 15 + 1e-6, FINE)
    Gn, _ = FS.grid(fpn, ftn, hfov, vfov)
    check["robust"]["expected"] = float(gaussian_filter(Gn, AIM_SIGMA / FINE, mode="nearest")[15, 15])
    print(f"fast search {1000 * t_fast:.0f} ms ({b['n_scored']} directions) + {1000 * t_prep:.0f} ms one-off occlusion; "
          f"old coarse grid alone {1000 * t_old:.0f} ms; grid agreement r={check['grid_corr']:.4f}, "
          f"max diff {100 * check['grid_max_abs_diff']:.1f} pts")
    real_pt = [A.pan_tilt_of(fwd[v], ref, up) for v in range(nR)]

    fig, ax = plt.subplots(1, 2, figsize=(17, 6.5), gridspec_kw={"width_ratios": [1.5, 1]})
    im = ax[0].imshow(100 * m_grid.T, origin="lower", aspect="auto", cmap="viridis",
                      extent=[pans[0] - COARSE / 2, pans[-1] + COARSE / 2, tilts[0] - COARSE / 2, tilts[-1] + COARSE / 2])
    fig.colorbar(im, ax=ax[0], label="% of gaze footprint inside the camera view")
    ax[0].plot([p for p, _ in real_pt], [t for _, t in real_pt], "wo", mfc="none", ms=9, label="real photos taken")
    ax[0].plot(pans[i], tilts[j], "c+", ms=16, mew=2, label=f"best on 5° grid ({100 * m_grid[i, j]:.1f}%)")
    ax[0].plot(best_pan, best_tilt, "r*", ms=16, label=f"best after 1° refine ({100 * best_m:.1f}%)")
    ax[0].plot(br["pan"], br["tilt"], "mD", ms=10, mfc="none", mew=2,
               label=f"safest if aim is off by ±{AIM_SIGMA:.0f}° ({100 * br['mass']:.1f}%)")
    ax[0].set_xlabel("pan (deg, + = turn right from yaw 0°)"); ax[0].set_ylabel("tilt (deg, + = up)")
    ax[0].legend(loc="lower left", fontsize=10)
    ax[0].set_title(f"{m_grid.size} directions on a 5° grid, then 1° around the 3 best peaks "
                    f"({b['n_scored']} in all, {1000 * t_fast:.0f} ms); camera FOV {hfov:.0f}°x{vfov:.0f}°")
    R = A.rotation_looking_at(A.forward_from_pan_tilt(best_pan, best_tilt, ref, up), up)
    K = K_room
    vis = A.visible_from(torch.as_tensor(P, device=DEV), cam, torch.as_tensor(R[None], dtype=torch.float32, device=DEV),
                         K, NET_W, NET_H)[0].cpu().numpy()
    Pc = (P - cam) @ R.T
    uu, vv = K[0, 0] * Pc[:, 0] / Pc[:, 2] + K[0, 2], K[1, 1] * Pc[:, 1] / Pc[:, 2] + K[1, 2]
    ax[1].scatter(uu[vis], vv[vis], s=0.5, c="0.75")
    fv = vis & f
    ax[1].scatter(uu[fv], vv[fv], s=2, c=w[fv], cmap="Reds", vmin=0, vmax=1)
    ax[1].set_xlim(0, NET_W); ax[1].set_ylim(NET_H, 0); ax[1].set_aspect("equal"); ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[1].set_title(f"chosen view: pan {best_pan:+.0f}°, tilt {best_tilt:+.0f}°\n"
                    f"sees {best_m:.0%} of the footprint; {best_foc:.0%} of the view is footprint")
    fig.suptitle("Step 5 - search: hidden surfaces found once from the camera spot; then for each pan/tilt "
                 "only 'is it inside the frame?' and add up the footprint")
    fig.tight_layout(); fig.savefig(os.path.join(out, "step5_search.png"), dpi=90); plt.close(fig)

    # ---------- step 6: real photos
    Pt, wt, at = (torch.as_tensor(x, device=DEV) for x in (P, w, area))
    real = []
    for v, p in enumerate(room_paths):
        Rv = torch.as_tensor(extr[v][None, :3, :3], dtype=torch.float32, device=DEV)
        m, foc, _ = A._score(Pt, wt, at, centers[v], K_room, NET_W, NET_H, Rv)
        real.append(dict(file=os.path.basename(p), yaw=poses.get(os.path.basename(p), {}).get("yaw"),
                         pan=real_pt[v][0], captured=float(m[0]), focus=float(foc[0])))
    bv = int(np.argmax([r["captured"] for r in real]))
    fig, ax = plt.subplots(1, 2, figsize=(17, 7), gridspec_kw={"width_ratios": [1.3, 1]})
    ax[0].bar([r["pan"] for r in real], [100 * r["captured"] for r in real], width=min(8, 360 / nR * 0.6), color="tab:blue",
              label="real photo")
    ax[0].plot(pans, 100 * m_grid.max(1), "k-", lw=1, label="search, best tilt per pan")
    ax[0].axvline(best_pan, color="r", ls="--", label=f"chosen pan {best_pan:+.0f}°")
    ax[0].set_xlabel("pan of the photo (deg, from VGGT)"); ax[0].set_ylabel("% of gaze footprint in view")
    ax[0].legend(); ax[0].set_title(f"captured share per real photo (mean {np.mean([r['captured'] for r in real]):.0%})")
    Rb = torch.as_tensor(extr[bv][None, :3, :3], dtype=torch.float32, device=DEV)
    visb = A.visible_from(Pt, centers[bv], Rb, K_room, NET_W, NET_H)[0].cpu().numpy()
    u, v_, _ = project(P, extr[bv], K_room, origs[bv].shape)
    ok = visb & f
    ax[1].imshow(origs[bv]); ax[1].scatter(u[ok], v_[ok], s=2, c=w[ok], cmap="Reds", vmin=0, vmax=1, alpha=0.45)
    ax[1].axis("off")
    ax[1].set_title(f"best real photo {real[bv]['file']}\nfootprint it can see (red, occlusion checked): {real[bv]['captured']:.0%}")
    fig.suptitle("Step 6 - check with the real photos taken at the spot")
    fig.tight_layout(); fig.savefig(os.path.join(out, "step6_real.png"), dpi=90); plt.close(fig)

    r = dict(circle=name, n_photos=nR, hfov=hfov, vfov=vfov, hfov_vggt=hfov_vggt,
             fov_source='measured' if fov_short is not None else 'vggt', speed_and_check=check, yaw=yaw, pitch=pitch, cone_half=half, reachable=reach_mass,
             search_pan=best_pan, search_tilt=best_tilt, search_captured=best_m, search_focus=best_foc,
             grid5_captured=float(m_grid[i, j]), best_real=real[bv]["file"], best_real_captured=real[bv]["captured"],
             mean_real_captured=float(np.mean([x["captured"] for x in real])), real=real,
             eye_to_cam=float(np.linalg.norm(eye - cam)))
    json.dump(r, open(os.path.join(out, "result.json"), "w"), indent=1)
    # everything the 3D viewer (view_3d.py) needs
    np.savez_compressed(os.path.join(out, "scene.npz"), P=P, rgb=rgb, w=w, eye=eye, gaze=gaze, half=half, cam=cam,
                        up=up, ref=ref, hfov=hfov, vfov=vfov, area=area,
                        best_pan=best_pan, best_tilt=best_tilt, centers=centers[:nR], fwd=fwd[:nR],
                        person_center=centers[iP], person_fwd=fwd[iP],
                        yaws=np.array([q["yaw"] if q["yaw"] is not None else np.nan for q in real], dtype=float),
                        captured=np.array([q["captured"] for q in real]))
    print(f"L2CS yaw {yaw:.1f} pitch {pitch:.1f} cone ±{half:.1f}; reachable {reach_mass:.1%}")
    print(f"search: pan {best_pan:+.0f} tilt {best_tilt:+.0f} captured {best_m:.1%} focus {best_foc:.1%}")
    print(f"best real photo {real[bv]['file']} {real[bv]['captured']:.1%}; mean real {r['mean_real_captured']:.1%}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True)
    ap.add_argument("--rooms", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "outputs_active", "steps"))
    ap.add_argument("--fov-short", type=float, default=None,
                    help="measured short-side FOV of the room camera in degrees (fov_from_rotation.py); "
                         "default: VGGT's guess")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(args.out, exist_ok=True)
    import final_form as F
    from l2cs import Pipeline
    from vggt.models.vggt import VGGT
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(DEV).eval()
    l2cs = Pipeline(weights=os.path.join(HERE, "models", "L2CSNet_gaze360.pkl"), arch="ResNet50", device=torch.device(DEV))
    results = [run_circle(r, args.person, model, l2cs, F, args.out, fov_short=args.fov_short) for r in args.rooms]

    labels = [f"{r['n_photos']} photos\n({360 // r['n_photos']}° steps)" for r in results]
    x = np.arange(len(results))
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for k, (key, lab) in enumerate((("reachable", "visible from the spot at all"), ("search_captured", "searched pan/tilt"),
                                    ("best_real_captured", "best real photo"), ("mean_real_captured", "random real photo (mean)"))):
        ax[0].bar(x + (k - 1.5) * 0.2, [100 * r[key] for r in results], 0.2, label=lab)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels); ax[0].set_ylabel("% of gaze footprint"); ax[0].legend(fontsize=10)
    ax[0].set_title("How much of the gaze footprint the camera sees")
    ax[1].plot(x, [r["search_pan"] for r in results], "r*-", ms=14, label="searched pan")
    ax[1].plot(x, [next(q["pan"] for q in r["real"] if q["file"] == r["best_real"]) for r in results], "bo-", ms=10,
               label="pan of best real photo")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels); ax[1].set_ylabel("pan (deg)"); ax[1].legend()
    ax[1].set_title("Chosen direction (should agree across circles)")
    fig.suptitle("Same person photo, three Turn Capture circles (60° / 30° / 20° steps)")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "compare.png"), dpi=90); plt.close(fig)

    print("\ncircle                         photos  reach  search(pan,tilt)        best real        mean real")
    for r in results:
        print(f"{r['circle']:30s} {r['n_photos']:4d}  {r['reachable']:5.1%}  {r['search_captured']:5.1%} "
              f"({r['search_pan']:+4.0f},{r['search_tilt']:+3.0f})   {r['best_real_captured']:5.1%} {r['best_real'][12:22]}  "
              f"{r['mean_real_captured']:5.1%}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
