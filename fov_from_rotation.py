r"""
fov_from_rotation.py - measure a camera's real field of view from photos taken while turning on one spot.

Premise: if the camera only ROTATES (no moving), two overlapping photos are related by
    x2 ~ K R K^-1 x1          (a homography; K holds the focal length f)
For the wrong f, K^-1 H K is not a rotation. So we search f that makes it a rotation
(Hartley 1997, "Self-calibration of stationary cameras", IJCV), and check that the rotation angle
agrees with the gyroscope yaw stored in poses.jsonl.
Needs no EXIF and no checkerboard; works for the Turn Capture circles and for any camera on the arm.

Usage (from sotsuron):  ..\\.venv\Scripts\python.exe -I fov_from_rotation.py ..\data\captures\20261116_1906_Room_20_degree
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

SCALE_W = 756                                       # work on 756 px wide copies (1/4 of 3024)


def load_gray(p):
    im = ImageOps.exif_transpose(Image.open(p)).convert("L")
    s = SCALE_W / im.width
    return np.array(im.resize((SCALE_W, round(im.height * s)), Image.Resampling.LANCZOS)), im.size


def pair_homography(a, b, sift):
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None:
        return None, 0
    m = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [x for x, y in (p for p in m if len(p) == 2) if x.distance < 0.75 * y.distance]
    if len(good) < 30:
        return None, len(good)
    pa = np.float32([ka[g.queryIdx].pt for g in good]); pb = np.float32([kb[g.trainIdx].pt for g in good])
    H, inl = cv2.findHomography(pa, pb, cv2.USAC_MAGSAC, 2.0)
    return (H, pa[inl.ravel() > 0], pb[inl.ravel() > 0]), int(inl.sum()) if H is not None else 0


def rot_error(H, f, cx, cy):
    """How far K^-1 H K is from a rotation (0 = perfect), and its rotation angle in degrees."""
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    M = np.linalg.inv(K) @ H @ K
    M /= np.cbrt(np.linalg.det(M))
    s = np.linalg.svd(M, compute_uv=False)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    return float(s[0] / s[2] - 1), ang


def reproj(H_unused, pa, pb, f, cx, cy):
    """Best rotation-only fit for focal f -> median transfer error in pixels (more robust than the SVD test)."""
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]]); Ki = np.linalg.inv(K)
    ra = (Ki @ np.c_[pa, np.ones(len(pa))].T).T; rb = (Ki @ np.c_[pb, np.ones(len(pb))].T).T
    ra /= np.linalg.norm(ra, axis=1, keepdims=True); rb /= np.linalg.norm(rb, axis=1, keepdims=True)
    U, _, Vt = np.linalg.svd(rb.T @ ra)                    # Kabsch: R a ~ b
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    q = (K @ (R @ ra.T)).T; q = q[:, :2] / q[:, 2:]
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    return float(np.median(np.linalg.norm(q - pb, axis=1))), ang


def main():
    d = sys.argv[1]
    files = sorted(glob.glob(os.path.join(d, "*.jpg")))
    poses = {}
    if os.path.exists(os.path.join(d, "poses.jsonl")):
        for line in open(os.path.join(d, "poses.jsonl"), encoding="utf-8"):
            j = json.loads(line); poses[j["file"]] = j
    files.sort(key=lambda p: poses.get(os.path.basename(p), {}).get("yaw", 0))
    sift = cv2.SIFT_create(4000)
    ims = [load_gray(p) for p in files]
    h, w = ims[0][0].shape
    cx, cy = w / 2, h / 2
    fs = np.linspace(0.4 * w, 2.0 * w, 321)               # short-side FOV from ~128 deg down to ~28 deg
    rows, curves = [], []
    for i in range(len(files)):
        j = (i + 1) % len(files)
        res, n = pair_homography(ims[i][0], ims[j][0], sift)
        if res is None or n < 40:
            continue
        H, pa, pb = res
        errs = np.array([reproj(H, pa, pb, f, cx, cy)[0] for f in fs])
        k = int(np.argmin(errs)); f = fs[k]
        _, ang = reproj(H, pa, pb, f, cx, cy)
        ya = poses.get(os.path.basename(files[i]), {}).get("yaw"); yb = poses.get(os.path.basename(files[j]), {}).get("yaw")
        gyro = None if ya is None or yb is None else abs(((yb - ya) + 180) % 360 - 180)
        rows.append(dict(a=os.path.basename(files[i]), b=os.path.basename(files[j]), inliers=n, f_px=f,
                         short_fov=float(np.degrees(2 * np.arctan(w / 2 / f))),
                         long_fov=float(np.degrees(2 * np.arctan(h / 2 / f))),
                         angle=ang, gyro=gyro, err_px=float(errs[k])))
        curves.append(errs / errs.min())
    if not rows:
        print("no overlapping pairs (need about 1/3 overlap: 20-30 deg steps for a phone)"); return
    f_med = float(np.median([r["f_px"] for r in rows]))
    print(f"{os.path.basename(d)}: {len(rows)} overlapping pairs, image {w}x{h} (scaled)")
    print(f"{'pair':44s} {'inl':>4s} {'shortFOV':>8s} {'longFOV':>7s} {'turn(img)':>9s} {'turn(gyro)':>10s} {'err px':>6s}")
    for r in rows:
        print(f"{r['a'][:20]:20s} -> {r['b'][:20]:20s} {r['inliers']:4d} {r['short_fov']:8.1f} {r['long_fov']:7.1f} "
              f"{r['angle']:9.1f} {r['gyro'] if r['gyro'] is not None else float('nan'):10.1f} {r['err_px']:6.2f}")
    sf = [r["short_fov"] for r in rows]
    print(f"MEDIAN short-side FOV {np.median(sf):.1f} deg (IQR {np.percentile(sf, 25):.1f}-{np.percentile(sf, 75):.1f}), "
          f"long-side {np.degrees(2 * np.arctan(h / 2 / f_med)):.1f} deg, fx = {f_med / w:.4f} x width")
    out = dict(source=d, width_scaled=w, height_scaled=h, f_over_width=f_med / w,
               short_fov_deg=float(np.median(sf)), long_fov_deg=float(np.degrees(2 * np.arctan(h / 2 / f_med))), pairs=rows)
    od = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs_fov")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, os.path.basename(d.rstrip("/\\")) + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
