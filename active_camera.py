"""
active_camera.py - turn a camera so that its field of view covers the gaze cone's footprint (裾野).

Idea
----
Both the gaze cone (apex = the person's eye) and the camera's view (apex = the camera) are cones.
Where each cone hits the room surfaces it leaves a footprint. We want the camera's footprint to
cover the gaze footprint as well as possible.

Everything is computed on 3D surface points (e.g. VGGT world points of the reference views):
  w_i  = gaze weight of point i  (soft cone, 0 outside; only points the eye can actually see)
  v_i  = 1 if point i is inside the camera's view AND not hidden behind something nearer
  a_i  = surface area that point i stands for (so dense and sparse regions count fairly)

  captured mass  = sum a_i w_i v_i / sum a_i w_i          (how much of the gaze footprint the camera sees)
  footprint IoU  = sum a_i min(w_i, v_i) / sum a_i max(w_i, v_i)   (how well the two footprints match)

The camera can only rotate (pan / tilt about its own centre), optionally zoom. We search pan/tilt
on a coarse grid and refine around the best one (GPU, batched, with a small z-buffer for occlusion).
With zoom, we pick the narrowest field of view that still captures >= `min_mass` of the gaze footprint,
which is where the two footprints match best.

Conventions: camera frame = OpenCV (x right, y down, z forward); R is world->camera.
`up` is the world's up direction (VGGT's world = first camera's frame, so up ~ (0,-1,0) if that photo was level).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch


# ----------------------------------------------------------------------------- geometry
def rotation_looking_at(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """World->camera rotation for a camera looking along `forward` with image 'down' = -up."""
    f = forward / (np.linalg.norm(forward) + 1e-12)
    down = -up / (np.linalg.norm(up) + 1e-12)
    y = down - (down @ f) * f
    if np.linalg.norm(y) < 1e-6:                       # looking straight up/down: pick any y
        y = np.cross(f, [1.0, 0.0, 0.0]) if abs(f[0]) < 0.9 else np.cross(f, [0.0, 1.0, 0.0])
    y /= np.linalg.norm(y)
    x = np.cross(y, f)
    return np.stack([x, y, f], axis=0)                 # rows = camera axes in world coords


def forward_from_pan_tilt(pan_deg, tilt_deg, ref_forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Pan (+ = turn right) and tilt (+ = look up) relative to `ref_forward`, about the world up axis."""
    u = up / np.linalg.norm(up)
    h0 = ref_forward - (ref_forward @ u) * u
    h0 /= np.linalg.norm(h0)
    right = np.cross(h0, u)
    p, t = np.deg2rad(pan_deg), np.deg2rad(tilt_deg)
    return np.cos(t) * (np.cos(p) * h0 + np.sin(p) * right) + np.sin(t) * u


def intrinsics_from_hfov(hfov_deg: float, width: int, height: int) -> np.ndarray:
    fx = 0.5 * width / np.tan(0.5 * np.deg2rad(hfov_deg))
    return np.array([[fx, 0, width / 2.0], [0, fx, height / 2.0], [0, 0, 1.0]])


def point_areas_from_depth(depth: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """Area a pixel's 3D point stands for, ~ (depth/fx)*(depth/fy). Use for VGGT points (per view)."""
    return (depth / fx) * (depth / fy)


# ----------------------------------------------------------------------------- visibility
def visible_from(
    P: torch.Tensor,             # (N,3) world points
    center: np.ndarray,          # (3,)
    R_w2c: torch.Tensor,         # (B,3,3)
    K: np.ndarray,               # (3,3)
    width: int,
    height: int,
    zbuf_cell_deg: float = 0.75,
    near: float = 0.05,
    depth_tol: float = 0.03,
) -> torch.Tensor:
    """(B,N) bool: point is inside the image AND is the nearest surface in its z-buffer cell.
    Cells have a fixed ANGULAR size, so wide and zoomed views (and the eye) judge occlusion alike."""
    B = R_w2c.shape[0]
    zbuf_cell_px = max(1, int(round(float(K[0, 0]) * np.deg2rad(zbuf_cell_deg))))
    Pc = torch.einsum("bij,nj->bni", R_w2c, P - torch.as_tensor(center, dtype=P.dtype, device=P.device))
    z = Pc[..., 2]
    zs = z.clamp(min=near)
    u = K[0, 0] * Pc[..., 0] / zs + K[0, 2]
    v = K[1, 1] * Pc[..., 1] / zs + K[1, 2]
    inside = (z > near) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    Wz, Hz = max(1, width // zbuf_cell_px), max(1, height // zbuf_cell_px)
    cell = (v / height * Hz).long().clamp(0, Hz - 1) * Wz + (u / width * Wz).long().clamp(0, Wz - 1)
    cell = torch.where(inside, cell, torch.full_like(cell, Wz * Hz))      # dummy slot for outside points
    zbuf = torch.full((B, Wz * Hz + 1), float("inf"), device=P.device, dtype=P.dtype)
    zbuf.scatter_reduce_(1, cell, torch.where(inside, z, torch.full_like(z, float("inf"))), reduce="amin")
    front = z <= zbuf.gather(1, cell) * (1.0 + depth_tol) + 1e-4
    return inside & front


# ----------------------------------------------------------------------------- gaze footprint
def gaze_weights(
    P: np.ndarray,
    eye: np.ndarray,
    gaze_dir: np.ndarray,
    half_angle_deg: float,
    z_edge: float = 1.96,
    up: np.ndarray = np.array([0.0, 1.0, 0.0]),
    device: str = "cuda",
    width: int = 640,
) -> np.ndarray:
    """
    Soft circular gaze cone, only on surfaces the eye can see (first surface hit).
    w = exp(-0.5 (theta/sigma)^2) with sigma = half_angle / z_edge, cut at 2 x half-angle.
    (For the ellipse cone, replace theta/sigma by m * z_edge from final_form.ellipse_radius.)
    """
    g = gaze_dir / np.linalg.norm(gaze_dir)
    V = P - eye
    cos = (V @ g) / (np.linalg.norm(V, axis=1) + 1e-12)
    theta = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    sigma = half_angle_deg / z_edge
    w = np.exp(-0.5 * (theta / sigma) ** 2)
    w[theta > 2 * half_angle_deg] = 0.0
    # occlusion from the eye: a virtual camera at the eye, looking along the gaze
    hfov = min(170.0, 4.0 * half_angle_deg + 10.0)
    K = intrinsics_from_hfov(hfov, width, width)
    R = torch.as_tensor(rotation_looking_at(g, up)[None], dtype=torch.float32, device=device)
    # finer cells + looser depth tolerance: surfaces seen at a grazing angle must not hide themselves
    vis = visible_from(torch.as_tensor(P, dtype=torch.float32, device=device), eye, R, K, width, width,
                       zbuf_cell_deg=0.4, depth_tol=0.08)[0]
    return (w * vis.cpu().numpy()).astype(np.float32)


def reachable_from(P: np.ndarray, cam_center: np.ndarray, up: np.ndarray, device: str = "cuda",
                   width: int = 512) -> np.ndarray:
    """Points the camera could see from where it stands, in ANY direction (6 cube faces of 90°).
    Gaze footprint outside this set can only be reached by moving the camera, not by turning it."""
    Pt = torch.as_tensor(P, dtype=torch.float32, device=device)
    K = intrinsics_from_hfov(90.0, width, width)
    u = up / np.linalg.norm(up)
    a = np.cross(u, [1.0, 0, 0]) if abs(u[0]) < 0.9 else np.cross(u, [0, 0, 1.0])
    a /= np.linalg.norm(a)
    b = np.cross(u, a)
    dirs = [a, -a, b, -b, u, -u]
    Rs = np.stack([rotation_looking_at(d, b if abs(d @ u) > 0.9 else u) for d in dirs])
    vis = visible_from(Pt, cam_center, torch.as_tensor(Rs, dtype=torch.float32, device=device), K, width, width)
    return vis.any(0).cpu().numpy()


# ----------------------------------------------------------------------------- search
@dataclass
class ViewChoice:
    pan_deg: float
    tilt_deg: float
    hfov_deg: float
    captured_mass: float      # share of the gaze footprint inside the camera view
    focus: float              # share of the camera view that is gaze footprint (precision)
    footprint_iou: float      # soft IoU of the two footprints
    R_w2c: np.ndarray


def _score(P, w, a, cam_center, K, width, height, R_batch, chunk=48):
    masses, focus, ious = [], [], []
    wa_sum = (w * a).sum()
    for s in range(0, R_batch.shape[0], chunk):
        vis = visible_from(P, cam_center, R_batch[s:s + chunk], K, width, height).float()
        seen = (a * w * vis).sum(1)
        masses.append(seen / (wa_sum + 1e-12))
        focus.append(seen / ((a * vis).sum(1) + 1e-12))
        ious.append((a * torch.minimum(w, vis)).sum(1) / ((a * torch.maximum(w, vis)).sum(1) + 1e-12))
    return torch.cat(masses), torch.cat(focus), torch.cat(ious)


def best_pan_tilt(
    P: np.ndarray, w: np.ndarray, cam_center: np.ndarray, ref_forward: np.ndarray, up: np.ndarray,
    hfov_deg: float = 60.0, width: int = 640, height: int = 480, area: Optional[np.ndarray] = None,
    objective: str = "mass", pan_range=(-180.0, 180.0), tilt_range=(-60.0, 60.0),
    coarse_step: float = 5.0, fine_step: float = 1.0, device: str = "cuda",
) -> ViewChoice:
    """Best camera rotation for a fixed field of view. objective: 'mass' (tie-break IoU) or 'iou'."""
    Pt = torch.as_tensor(P, dtype=torch.float32, device=device)
    wt = torch.as_tensor(w, dtype=torch.float32, device=device)
    at = torch.ones_like(wt) if area is None else torch.as_tensor(area, dtype=torch.float32, device=device)
    K = intrinsics_from_hfov(hfov_deg, width, height)

    def run(pans, tilts):
        grid = [(p, t) for p in pans for t in tilts]
        R = np.stack([rotation_looking_at(forward_from_pan_tilt(p, t, ref_forward, up), up) for p, t in grid])
        m, foc, iou = _score(Pt, wt, at, cam_center, K, width, height,
                             torch.as_tensor(R, dtype=torch.float32, device=device))
        score = m + 1e-3 * iou if objective == "mass" else iou
        k = int(torch.argmax(score))
        return grid[k], R[k], float(m[k]), float(foc[k]), float(iou[k])

    (p0, t0), *_ = run(np.arange(pan_range[0], pan_range[1], coarse_step),
                       np.arange(tilt_range[0], tilt_range[1] + 1e-6, coarse_step))
    (p1, t1), R1, m1, f1, iou1 = run(np.arange(p0 - coarse_step, p0 + coarse_step + 1e-6, fine_step),
                                     np.arange(t0 - coarse_step, t0 + coarse_step + 1e-6, fine_step))
    return ViewChoice(float(p1), float(t1), hfov_deg, m1, f1, iou1, R1)


def score_grid(P, w, cam_center, ref_forward, up, hfov_deg, pans, tilts, width=640, height=480,
               area=None, device="cuda"):
    """Captured share / focus / IoU for every (pan, tilt) in the grid -> 3 arrays (len(pans), len(tilts)).
    Used to draw the whole search as a heat map."""
    Pt = torch.as_tensor(P, dtype=torch.float32, device=device)
    wt = torch.as_tensor(w, dtype=torch.float32, device=device)
    at = torch.ones_like(wt) if area is None else torch.as_tensor(area, dtype=torch.float32, device=device)
    K = intrinsics_from_hfov(hfov_deg, width, height)
    R = np.stack([rotation_looking_at(forward_from_pan_tilt(p, t, ref_forward, up), up) for p in pans for t in tilts])
    m, foc, iou = _score(Pt, wt, at, cam_center, K, width, height, torch.as_tensor(R, dtype=torch.float32, device=device))
    shape = (len(pans), len(tilts))
    return (m.cpu().numpy().reshape(shape), foc.cpu().numpy().reshape(shape), iou.cpu().numpy().reshape(shape))


def pan_tilt_of(forward: np.ndarray, ref_forward: np.ndarray, up: np.ndarray):
    """Inverse of forward_from_pan_tilt: (pan, tilt) in degrees of a viewing direction."""
    u = up / np.linalg.norm(up)
    h0 = ref_forward - (ref_forward @ u) * u
    h0 /= np.linalg.norm(h0)
    right = np.cross(h0, u)
    f = forward / np.linalg.norm(forward)
    return float(np.degrees(np.arctan2(f @ right, f @ h0))), float(np.degrees(np.arcsin(np.clip(f @ u, -1, 1))))


def best_view_with_zoom(
    P, w, cam_center, ref_forward, up, hfovs: Sequence[float] = (70, 60, 50, 40, 30, 20, 15),
    **kw,
) -> list:
    """Best pan/tilt for each zoom level (field of view). Choose among them with pick_zoom()."""
    out = [best_pan_tilt(P, w, cam_center, ref_forward, up, hfov_deg=h, **kw) for h in hfovs]
    return out


def pick_zoom(choices: list, reachable_mass: float = 1.0, min_share: float = 0.95) -> ViewChoice:
    """Narrowest view (best footprint match) that still captures >= min_share of what is reachable."""
    ok = [c for c in choices if c.captured_mass >= min_share * reachable_mass]
    return max(ok, key=lambda c: c.footprint_iou) if ok else max(choices, key=lambda c: c.captured_mass)


def view_toward_centroid(P, w, cam_center, up) -> np.ndarray:
    """Simple baseline: point the camera at the weighted centre of the gaze footprint."""
    c = (P * w[:, None]).sum(0) / (w.sum() + 1e-12)
    return rotation_looking_at(c - cam_center, up)


# ----------------------------------------------------------------------------- fast exact search
def rotations_pan_tilt(pans, tilts, ref_forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """(len(pans)*len(tilts), 3, 3) world->camera rotations, pan-major order (vectorised rotation_looking_at)."""
    u = up / np.linalg.norm(up)
    h0 = ref_forward - (ref_forward @ u) * u
    h0 /= np.linalg.norm(h0)
    right = np.cross(h0, u)
    p, t = np.meshgrid(np.deg2rad(np.asarray(pans, float)), np.deg2rad(np.asarray(tilts, float)), indexing="ij")
    p, t = p.ravel()[:, None], t.ravel()[:, None]
    f = np.cos(t) * (np.cos(p) * h0 + np.sin(p) * right) + np.sin(t) * u
    y = -u[None] + (f @ u)[:, None] * f                       # image down = -up, made orthogonal to f
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    x = np.cross(y, f)
    return np.stack([x, y, f], axis=1)


class FastSearch:
    """
    Same answer as best_pan_tilt, much less work.

    Premise: when the camera only ROTATES about its optical centre, the set of surfaces hidden behind
    other surfaces does not change - only which of them fall inside the frame. So:
      1. occlusion test ONCE for the whole sphere around the camera (6 cube faces, angular z-buffer),
      2. keep the footprint points that are visible (w > 0) as unit directions,
      3. for every candidate pan/tilt only test "inside the rectangle?" (a 3x3 rotation + 2 comparisons
         per point, no z-buffer) and add up the footprint weight.
    A real camera on an arm does not turn exactly about its optical centre: pass center_fn(pan, tilt)
    to re-score the best candidates exactly (full z-buffer at the moved centre).

    The field of view is given as hfov and vfov separately (portrait phone: hfov = short side;
    RealSense landscape: hfov = long side). Use MEASURED values (fov_from_rotation.py or the
    RealSense SDK intrinsics), not VGGT's focal guess.
    """

    def __init__(self, P, w, cam_center, ref_forward, up, area=None, device="cuda", w_min=1e-3):
        self.P, self.w, self.cam, self.ref, self.up, self.dev = P, w, np.asarray(cam_center, float), ref_forward, up, device
        self.a = np.ones_like(w) if area is None else area
        vis = reachable_from(P, self.cam, up, device=device)
        aw = self.a * w
        self.total = float(aw.sum())
        self.reachable = float((aw * vis).sum() / max(self.total, 1e-12))
        D = P - self.cam
        D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
        keep = vis & (w > w_min)
        self.Df = torch.as_tensor(D[keep], dtype=torch.float32, device=device)
        self.mf = torch.as_tensor(aw[keep] / max(self.total, 1e-12), dtype=torch.float32, device=device)
        self.Dv = torch.as_tensor(D[vis], dtype=torch.float32, device=device)       # for focus only
        self.av = torch.as_tensor(self.a[vis], dtype=torch.float32, device=device)

    @staticmethod
    def _inside(D, R, tx, ty):
        q = torch.einsum("bij,nj->bni", R, D)
        z = q[..., 2]
        return (z > 1e-6) & (q[..., 0].abs() <= tx * z) & (q[..., 1].abs() <= ty * z)

    @torch.no_grad()
    def mass(self, R: np.ndarray, hfov: float, vfov: float, chunk: int = 256) -> np.ndarray:
        """Share of the WHOLE gaze footprint seen, for each rotation in R (B,3,3)."""
        tx, ty = np.tan(np.deg2rad(hfov / 2)), np.tan(np.deg2rad(vfov / 2))
        Rt = torch.as_tensor(R, dtype=torch.float32, device=self.dev)
        out = [self._inside(self.Df, Rt[s:s + chunk], tx, ty).float() @ self.mf for s in range(0, len(Rt), chunk)]
        return torch.cat(out).cpu().numpy()

    @torch.no_grad()
    def focus(self, R: np.ndarray, hfov: float, vfov: float) -> float:
        """Share of the visible surface area in the frame that is gaze footprint (single view)."""
        tx, ty = np.tan(np.deg2rad(hfov / 2)), np.tan(np.deg2rad(vfov / 2))
        Rt = torch.as_tensor(R[None], dtype=torch.float32, device=self.dev)
        seen = float(self._inside(self.Df, Rt, tx, ty).float()[0] @ self.mf) * self.total
        area_in = float(self._inside(self.Dv, Rt, tx, ty).float()[0] @ self.av)
        return seen / max(area_in, 1e-12)

    def grid(self, pans, tilts, hfov, vfov):
        R = rotations_pan_tilt(pans, tilts, self.ref, self.up)
        return self.mass(R, hfov, vfov).reshape(len(pans), len(tilts)), R

    def best(self, hfov, vfov, pan_range=(-180.0, 180.0), tilt_range=(-45.0, 45.0), coarse=5.0, fine=1.0,
             aim_sigma_deg=0.0, n_peaks=3, center_fn=None, n_exact=20, width=640):
        """
        1) coarse grid, 2) fine grid around the n_peaks best coarse cells,
        3) optional: aim_sigma_deg > 0 -> pick the direction whose EXPECTED share under a Gaussian aim error
           is highest (stays good if the arm/pose is off by a few degrees), not the knife-edge maximum,
        4) optional: center_fn(pan, tilt) -> camera centre for that pose (arm offset): re-score the
           n_exact best fine candidates with the full z-buffer at their own centre.
        """
        from scipy.ndimage import gaussian_filter
        pans = np.arange(pan_range[0], pan_range[1], coarse)
        tilts = np.arange(tilt_range[0], tilt_range[1] + 1e-6, coarse)
        G, _ = self.grid(pans, tilts, hfov, vfov)
        Gs = gaussian_filter(G, aim_sigma_deg / coarse, mode=("wrap", "nearest")) if aim_sigma_deg > 0 else G
        peaks = []
        for k in np.argsort(Gs.ravel())[::-1]:          # n_peaks coarse cells that are not neighbours
            i, j = np.unravel_index(k, G.shape)
            if all(abs(((pans[i] - p + 180) % 360) - 180) > 2 * coarse or abs(tilts[j] - t) > 2 * coarse for p, t in peaks):
                peaks.append((pans[i], tilts[j]))
            if len(peaks) == n_peaks:
                break
        r = coarse + 3 * aim_sigma_deg
        cands = []
        for p0, t0 in peaks:
            fp = np.arange(p0 - r, p0 + r + 1e-6, fine)
            ft = np.arange(max(tilt_range[0], t0 - r), min(tilt_range[1], t0 + r) + 1e-6, fine)
            F_, R_ = self.grid(fp, ft, hfov, vfov)
            Fs = gaussian_filter(F_, aim_sigma_deg / fine, mode="nearest") if aim_sigma_deg > 0 else F_
            for k in range(F_.size):
                i, j = np.unravel_index(k, F_.shape)
                cands.append(dict(pan=float(((fp[i] + 180) % 360) - 180), tilt=float(ft[j]), mass=float(F_[i, j]),
                                  expected=float(Fs[i, j]), R=R_[k]))
        cands.sort(key=lambda c: (c["expected"], c["mass"]), reverse=True)
        if center_fn is not None:
            th, tv = np.tan(np.deg2rad(hfov / 2)), np.tan(np.deg2rad(vfov / 2))
            height = int(round(width * tv / th))
            K = intrinsics_from_hfov(hfov, width, height)
            Pt = torch.as_tensor(self.P, dtype=torch.float32, device=self.dev)
            wt = torch.as_tensor(self.w, dtype=torch.float32, device=self.dev)
            at = torch.as_tensor(self.a, dtype=torch.float32, device=self.dev)
            for c in cands[:n_exact]:
                c["center"] = np.asarray(center_fn(c["pan"], c["tilt"]), float)
                m, _, _ = _score(Pt, wt, at, c["center"], K, width, height,
                                 torch.as_tensor(c["R"][None], dtype=torch.float32, device=self.dev))
                c["mass_exact"] = float(m[0])
            cands = sorted(cands[:n_exact], key=lambda c: c["mass_exact"], reverse=True) + cands[n_exact:]
        b = dict(cands[0])
        b["focus"] = self.focus(b["R"], hfov, vfov)
        b["coarse_grid"] = (pans, tilts, G)
        b["n_scored"] = int(G.size + len(cands))
        return b
