"""The 'brain' for the robot demo: the active_camera_steps.py + final_form.py method, split into calls.

    plan(room photos, person photo)  -> where to turn (pan/tilt), using VGGT + L2CS + gaze footprint + FastSearch
    target(photo taken after turning) -> where the person is looking in that photo (gaze cone x DeepGaze IIE)

Main pipeline only (final_form.py), no IEF. Same constants as active_camera_steps.py.
Angles: pan + = turn right, tilt + = look up (active_camera.py convention), degrees.
"""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # Xet downloader crashed on this laptop

import active_camera as A  # noqa: E402
import final_form as F  # noqa: E402

DEV = "cuda"
QUANTILE, STRIDE, CONF_KEEP = 0.95, 2, 0.6       # as active_camera_steps.run_circle
COARSE, FINE, TILTS = 5.0, 1.0, (-45.0, 45.0)


def net_size(w, h, long_side=518):
    """VGGT input size: long side 518, both sides multiples of 14, no cropping."""
    s = long_side / max(w, h)
    return int(round(w * s / 14) * 14), int(round(h * s / 14) * 14)


def load_rgb(path_or_bgr):
    if isinstance(path_or_bgr, np.ndarray):
        return path_or_bgr[:, :, ::-1].copy()
    return np.array(ImageOps.exif_transpose(Image.open(path_or_bgr)).convert("RGB"))


def load_vggt():
    """VGGT-1B built directly on the GPU and filled tensor by tensor from the cached safetensors file.
    VGGT.from_pretrained() first builds the 4.7 GB model in RAM and then reads the 4.7 GB file next to it;
    on this 16 GB laptop that ran out of Windows commit memory ('paging file is too small' / random crashes)."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from vggt.models.vggt import VGGT
    path = hf_hub_download("facebook/VGGT-1B", "model.safetensors")
    with torch.device(DEV):
        model = VGGT()
    params = dict(model.named_parameters()) | dict(model.named_buffers())
    seen = set()
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k in params:
                with torch.no_grad():
                    params[k].copy_(f.get_tensor(k))
                seen.add(k)
    missing = [k for k, _ in model.named_parameters() if k not in seen]
    if missing:
        raise RuntimeError(f"VGGT weights missing for {len(missing)} parameters, e.g. {missing[:3]}")
    # some buffers (e.g. aggregator._resnet_mean/_std) are made with torch.FloatTensor(...), which ignores the
    # torch.device context and stays on the CPU; they are tiny, so moving the whole model costs nothing
    return model.to(DEV).eval()


def K_net(fx, fy, cx, cy, w, h, W, H):
    """Scale pixel intrinsics of a w x h image to the W x H network size."""
    return np.array([[fx * W / w, 0, cx * W / w], [0, fy * H / h, cy * H / h], [0, 0, 1.0]])


SQ = 518                                         # square network size when photos of different shapes are mixed


def pad_square(rgb, K):
    """Centre the photo on a white square (VGGT's own 'pad' mode uses white) and shift the principal point.
    Returns the square image, its K (original pixels), the (x, y) offset and the valid-content mask at SQ x SQ."""
    h, w = rgb.shape[:2]
    S = max(h, w)
    ox, oy = (S - w) // 2, (S - h) // 2
    out = np.full((S, S, 3), 255, np.uint8)
    out[oy:oy + h, ox:ox + w] = rgb
    K2 = np.array(K, float).copy()
    K2[0, 2] += ox
    K2[1, 2] += oy
    m = np.zeros((SQ, SQ), bool)
    s = SQ / S
    m[int(np.ceil(oy * s)):int((oy + h) * s), int(np.ceil(ox * s)):int((ox + w) * s)] = True
    return out, K2, (ox, oy), m


class Brain:
    def __init__(self):
        from l2cs import Pipeline
        import deepgaze_pytorch
        t0 = time.time()
        print("[brain] loading VGGT, L2CS, DeepGaze IIE ...", flush=True)
        self.vggt = load_vggt()
        self.l2cs = Pipeline(weights=os.path.join(HERE, "models", "L2CSNet_gaze360.pkl"), arch="ResNet50",
                             device=torch.device(DEV))
        self.dg = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEV).eval().half()
        self.centerbias = np.load(os.path.join(HERE, "centerbias_mit1003.npy"))
        self.bins = np.arange(F.NUM_BINS, dtype=np.float32) * F.BIN_WIDTH_DEG + F.BIN_START_DEG
        self.state = {}
        print(f"[brain] ready in {time.time() - t0:.0f} s", flush=True)

    def warmup(self, w=1280, h=720, n_room=6):
        """The FIRST call for each image size is slow (CUDA picks/compiles kernels): measured on the RTX 5090,
        plan 33 s then 2.2 s, target 61 s then 1.8 s. Run every model once at the robot's image size so the
        robot does not wait. Grey dummy images: no face is found, which is fine for warming up."""
        t0 = time.time()
        W, H = net_size(w, h)
        grey = np.full((h, w, 3), 127, np.uint8)
        self._vggt([grey] * (n_room + 1), W, H)          # plan: room photos + person photo
        self._vggt([grey] * (n_room + 2), W, H)          # target: + the photo after turning
        sq = np.full((SQ, SQ, 3), 127, np.uint8)         # plan_map/target with a saved map: room + anchor + person (+ target)
        self._vggt([sq] * (n_room + 2), SQ, SQ)
        self._vggt([sq] * (n_room + 3), SQ, SQ)
        self.faces(grey)
        with torch.inference_mode():
            self.l2cs.model(torch.zeros(1, 3, 224, 224, device=DEV))
            s = min(1.0, F.DEEPGAZE_MAX_DIM / max(h, w))
            Hd, Wd = int(h * s), int(w * s)
            self.dg(torch.zeros(1, 3, Hd, Wd, device=DEV).half(), torch.zeros(1, Hd, Wd, device=DEV).half())
        torch.cuda.synchronize()
        print(f"[brain] warmed up for {w}x{h} images in {time.time() - t0:.0f} s", flush=True)

    # ------------------------------------------------------------------ faces
    def faces(self, bgr, min_score=0.9):
        """[(x1, y1, x2, y2, score)] most confident first; [] when nobody is visible.
        First robot run (2026-11-17): RetinaFace saw the door as a face (score 0.64, box 77% of the image) while
        the real face scored 1.00; with min_score 0.6 and 'largest first' the door was used as the person.
        So: score >= 0.9, boxes taller than half the image dropped, most confident first."""
        H = bgr.shape[0]
        out = []
        for box, _lm, score in (self.l2cs.detector(bgr) or []):
            if score >= min_score and (box[3] - box[1]) <= 0.5 * H:
                out.append((*map(float, box[:4]), float(score)))
        return sorted(out, key=lambda b: (-round(b[4], 2), -(b[2] - b[0]) * (b[3] - b[1])))

    def gaze(self, bgr):
        """L2CS on the largest face: bbox, yaw, pitch, 95% cone half-angle, bin probabilities."""
        fs = self.faces(bgr)
        if not fs:
            raise RuntimeError("no face in the person photo")
        bbox = np.array(fs[0][:4], dtype=np.float32)
        face_t, _ = F.preprocess_face_for_l2cs_with_vis(bgr, bbox, torch.device(DEV))
        with torch.inference_mode():
            yl, pl = self.l2cs.model(face_t)
        yp, pp = torch.softmax(yl, 1).cpu().numpy()[0], torch.softmax(pl, 1).cpu().numpy()[0]
        yaw, pitch = F.expected_angle_deg(yp, self.bins), F.expected_angle_deg(pp, self.bins)
        np.random.seed(0)        # the cone half-angle is sampled (4000 draws): fixed seed -> same answer every run
        half = F.cone_half_angle_from_probs(yp, pp, self.bins, 4000, QUANTILE, F.CONE_MIN_DEG, F.CONE_MAX_DEG)
        return dict(bbox=bbox, yaw=float(yaw), pitch=float(pitch), half=float(half))

    # ------------------------------------------------------------------ VGGT
    def _vggt(self, rgbs, W, H):
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        ims = torch.stack([torch.from_numpy(np.array(Image.fromarray(r).resize((W, H), Image.Resampling.BICUBIC)))
                           .permute(2, 0, 1).float() / 255.0 for r in rgbs])
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred = self.vggt(ims.to(DEV))
        extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], ims.shape[-2:])
        out = dict(extr=extr[0].float().cpu().numpy(), intr=intr[0].float().cpu().numpy(),
                   wp=pred["world_points"][0].float().cpu().numpy(),
                   wconf=pred["world_points_conf"][0].float().cpu().numpy(),
                   depth=pred["depth"][0, ..., 0].float().cpu().numpy(), ims=ims)
        del pred
        torch.cuda.empty_cache()
        return out

    def _eye_gaze(self, g, v, iP, rgb_person, K_person, W, H):
        eye, _ = F.estimate_eye_origin_world(g["bbox"], v["wp"][iP], rgb_person.shape[:2], H, W, K_person, v["extr"][iP])
        gaze = v["extr"][iP][:3, :3].T @ F.gaze_vector_from_angles(np.deg2rad(g["yaw"]), np.deg2rad(g["pitch"]))
        return eye, gaze / np.linalg.norm(gaze)

    # ------------------------------------------------------------------ step A: where to turn
    def plan(self, room, person, K_room, K_person, hfov, vfov, room_pans=None, pan_limits=None):
        """room: list of paths/BGR arrays; person: path/BGR; K_*: 3x3 at original pixel size (fx, fy, cx, cy used).
        room_pans: the robot pan (deg, + = right) of each room photo, to turn the answer into a robot command.
        pan_limits: (lo, hi) robot pan the arm can reach."""
        t0 = time.time()
        rgbs = [load_rgb(p) for p in room] + [load_rgb(person)]
        h0, w0 = rgbs[0].shape[:2]
        W, H = net_size(w0, h0)
        nR = iP = len(room)
        Kr = K_net(K_room[0, 0], K_room[1, 1], K_room[0, 2], K_room[1, 2], w0, h0, W, H)
        hp, wpx = rgbs[iP].shape[:2]
        Kp = K_net(K_person[0, 0], K_person[1, 1], K_person[0, 2], K_person[1, 2], wpx, hp, W, H)
        v = self._vggt(rgbs, W, H)
        extr, wp, wconf, depth = v["extr"], v["wp"], v["wconf"], v["depth"]
        t_vggt = time.time() - t0

        fwd = np.stack([e[2, :3] for e in extr])
        _, _, vt = np.linalg.svd(fwd[:nR])
        up = vt[2] if vt[2] @ (-extr[0][1, :3]) > 0 else -vt[2]
        centers = np.stack([-e[:3, :3].T @ e[:3, 3] for e in extr])
        cam, ref = centers[:nR].mean(0), fwd[0]

        P, area = [], []
        s = STRIDE
        for k in range(nR):
            pts = wp[k, ::s, ::s].reshape(-1, 3)
            c = wconf[k, ::s, ::s].reshape(-1)
            keep = (c >= np.quantile(c, 1 - CONF_KEEP)) & np.isfinite(pts).all(1)
            a = A.point_areas_from_depth(depth[k, ::s, ::s].reshape(-1), Kr[0, 0] / s, Kr[1, 1] / s)
            P.append(pts[keep]); area.append(a[keep])
        P, area = np.concatenate(P).astype(np.float32), np.concatenate(area).astype(np.float32)

        bgr_person = rgbs[iP][:, :, ::-1].copy()
        g = self.gaze(bgr_person)
        eye, gaze = self._eye_gaze(g, v, iP, rgbs[iP], Kp, W, H)
        w = A.gaze_weights(P, eye, gaze, g["half"], z_edge=1.96, up=up, device=DEV)
        reach = A.reachable_from(P, cam, up, device=DEV)
        reach_mass = float((w * area * reach).sum() / max((w * area).sum(), 1e-12))

        # robot pan of photo 0 = the VGGT reference direction; check the other photos agree with the robot angles
        vggt_pans = np.array([A.pan_tilt_of(fwd[k], ref, up)[0] for k in range(nR)])
        p0, resid = 0.0, None
        pan_range = (-180.0, 180.0)
        if room_pans is not None:
            room_pans = np.asarray(room_pans, float)
            p0 = float(room_pans[0])
            d = (vggt_pans - (room_pans - p0) + 180) % 360 - 180
            resid = d.tolist()
            if pan_limits is not None:
                pan_range = (pan_limits[0] - p0, pan_limits[1] - p0)

        FS = A.FastSearch(P, w, cam, ref, up, area=area, device=DEV)
        b = FS.best(hfov, vfov, pan_range=pan_range, tilt_range=TILTS, coarse=COARSE, fine=FINE)
        person_pan = float(A.pan_tilt_of(fwd[iP], ref, up)[0])
        res = dict(yaw=g["yaw"], pitch=g["pitch"], cone_half=g["half"], reachable=reach_mass,
                   pan_vggt=float(b["pan"]), tilt=float(b["tilt"]), captured=float(b["mass"]), focus=float(b["focus"]),
                   robot_pan=p0 + float(b["pan"]), robot_tilt=float(b["tilt"]),
                   person_pan_vggt=person_pan, room_pan_residual_deg=resid,
                   secs_vggt=round(t_vggt, 2), secs_total=round(time.time() - t0, 2), net_size=(W, H))
        self.state = dict(rgbs=rgbs, K_room=K_room, K_person=K_person, gaze=g, nR=nR, plan=res)
        return res

    # ------------------------------------------------------------------ step A': where to turn, using a saved room map
    def plan_map(self, room, K_room, person, K_person, hfov, vfov, pan0=0.0, tilt0=0.0, pan_limits=None,
                 tilt_limits=(-40.0, 40.0), anchor=None, K_anchor=None, arm=None, viz=False):
        """The room map comes from photos taken BEFORE (e.g. the iPhone Turn Capture circle); the robot only took the
        person photo. VGGT puts that photo in the same map, which gives the eyes AND the robot camera's position and
        direction. The search runs from the robot camera's position, relative to the direction it faces now:
            robot pan  = pan0 + search pan            (pan0, tilt0 = robot pose when the person photo was taken)
            robot tilt = tilt0 + (search tilt - elevation of the person photo)
        Photos of different shapes (iPhone portrait, RealSense landscape) are padded to SQ x SQ squares.

        anchor (optional): an earlier photo of the scene (e.g. the iPhone person photo), added to the VGGT run as extra
        room structure. The robot camera position is where VGGT places the robot's own photo; the result reports
        how far that is from the anchor's camera (robot_cam_vggt_to_anchor).

        arm (optional): {"radius_m": camera distance from the J1 axis (robot/piper_fk.py),
                         "m_per_unit": metres per VGGT unit}. With both, the camera centre swings around the base
        axis when the base turns (FastSearch center_fn); without a metric scale it is assumed to turn in place."""
        t0 = time.time()
        rooms = [pad_square(load_rgb(p), K_room) for p in room]
        extra = [pad_square(load_rgb(anchor), K_anchor)] if anchor is not None else []
        pers = pad_square(load_rgb(person), K_person)
        allp = rooms + extra + [pers]
        rgbs = [r[0] for r in allp]
        masks = [r[3] for r in allp]
        Ks = [K_net(k[0, 0], k[1, 1], k[0, 2], k[1, 2], im.shape[1], im.shape[0], SQ, SQ)
              for im, k in zip(rgbs, [r[1] for r in allp])]
        nR = len(room)                      # room photos: indices 0..nR-1 (their points make the map)
        iA = nR if extra else None          # anchor photo
        iP = len(allp) - 1                  # the robot's person photo (last)
        v = self._vggt(rgbs, SQ, SQ)
        extr, wp, wconf, depth = v["extr"], v["wp"], v["wconf"], v["depth"]
        t_vggt = time.time() - t0

        fwd = np.stack([e[2, :3] for e in extr])
        _, _, vt = np.linalg.svd(fwd[:nR])                       # the room circle was taken level -> its normal is "up"
        up = vt[2] if vt[2] @ (-extr[0][1, :3]) > 0 else -vt[2]
        centers = np.stack([-e[:3, :3].T @ e[:3, 3] for e in extr])
        # search FROM the robot camera where VGGT places the robot's own photo, relative to the robot's view.
        # (First robot run, 2026-11-17: the robot stood ~1/4 of the room away from the anchor spot; using the anchor
        # centre turned the arm to the wrong side. The anchor now only adds room structure.)
        cam = centers[iP]
        ref = fwd[iP]

        P, area, rgb = [], [], []
        s = STRIDE
        for k in range(nR):
            pts = wp[k, ::s, ::s].reshape(-1, 3)
            c = wconf[k, ::s, ::s].reshape(-1)
            valid = masks[k][::s, ::s].reshape(-1)                # drop the white padding
            ok = valid & np.isfinite(pts).all(1)
            keep = ok & (c >= np.quantile(c[ok], 1 - CONF_KEEP))
            a = A.point_areas_from_depth(depth[k, ::s, ::s].reshape(-1), Ks[k][0, 0] / s, Ks[k][1, 1] / s)
            col = v["ims"][k].permute(1, 2, 0).numpy()[::s, ::s].reshape(-1, 3)
            P.append(pts[keep]); area.append(a[keep]); rgb.append(col[keep])
        P, area = np.concatenate(P).astype(np.float32), np.concatenate(area).astype(np.float32)
        rgb = (np.concatenate(rgb) * 255).astype(np.uint8)

        rgb_p = load_rgb(person)
        g = self.gaze(rgb_p[:, :, ::-1].copy())                   # L2CS on the real (unpadded) photo
        ox, oy = pers[2]
        g["bbox"] = g["bbox"] + np.array([ox, oy, ox, oy], np.float32)   # face box in padded-square pixels
        eye, gaze = self._eye_gaze(g, v, iP, rgbs[iP], Ks[iP], SQ, SQ)
        w = A.gaze_weights(P, eye, gaze, g["half"], z_edge=1.96, up=up, device=DEV)
        reach = A.reachable_from(P, cam, up, device=DEV)
        reach_mass = float((w * area * reach).sum() / max((w * area).sum(), 1e-12))

        elev_p = float(np.degrees(np.arcsin(np.clip(fwd[iP] @ up, -1, 1))))
        pl = pan_limits or (-180.0, 180.0)
        pan_range = (pl[0] - pan0, pl[1] - pan0)
        tilt_range = (max(TILTS[0], tilt_limits[0] - tilt0 + elev_p), min(TILTS[1], tilt_limits[1] - tilt0 + elev_p))
        center_fn = None
        arm = arm or {}
        if arm.get("radius_m") and arm.get("m_per_unit"):
            r_u = float(arm["radius_m"]) / float(arm["m_per_unit"])            # radius in VGGT units
            h0 = ref - (ref @ up) * up; h0 /= np.linalg.norm(h0)
            right = np.cross(h0, up)
            axis = cam - r_u * h0                                              # camera faces outward from the J1 axis

            def center_fn(p, t, axis=axis, h0=h0, right=right, r_u=r_u):
                q = np.deg2rad(p)
                return axis + r_u * (np.cos(q) * h0 + np.sin(q) * right)
        FS = A.FastSearch(P, w, cam, ref, up, area=area, device=DEV)
        b = FS.best(hfov, vfov, pan_range=pan_range, tilt_range=tilt_range, coarse=COARSE, fine=FINE, center_fn=center_fn)
        if not (pan_range[0] <= b["pan"] <= pan_range[1]):       # the 1-deg refinement can step past the arm's range
            b["pan"] = float(np.clip(b["pan"], *pan_range))
            b["R"] = A.rotation_looking_at(A.forward_from_pan_tilt(b["pan"], b["tilt"], ref, up), up)
            b["mass"] = float(FS.mass(b["R"][None], hfov, vfov)[0])
        # how much of the footprint the robot sees WITHOUT turning (the current view)
        R_now = A.rotation_looking_at(A.forward_from_pan_tilt(0.0, elev_p, ref, up), up)
        now = float(FS.mass(R_now[None], hfov, vfov)[0])
        res = dict(yaw=g["yaw"], pitch=g["pitch"], cone_half=g["half"], reachable=reach_mass,
                   turn_pan=float(b["pan"]), turn_tilt=float(b["tilt"]) - elev_p,
                   robot_pan=pan0 + float(b["pan"]), robot_tilt=tilt0 + float(b["tilt"]) - elev_p,
                   captured=float(b["mass"]), focus=float(b["focus"]), captured_without_turning=now,
                   person_photo_elevation=elev_p, robot_cam_to_map_centre=float(np.linalg.norm(cam - centers[:nR].mean(0))),
                   map_size=float(np.percentile(np.linalg.norm(P - centers[:nR].mean(0), axis=1), 90)),
                   # check of the "robot stands where the anchor photo was taken" assumption (VGGT units)
                   robot_cam_vggt_to_anchor=None if iA is None else float(np.linalg.norm(centers[iP] - centers[iA])),
                   eye_to_robot_cam=float(np.linalg.norm(eye - cam)),
                   used_anchor=iA is not None, arm=arm, used_center_fn=center_fn is not None,
                   secs_vggt=round(t_vggt, 2), secs_total=round(time.time() - t0, 2), net_size=(SQ, SQ))
        self.state = dict(mode="map", rgbs=rgbs, Ks=Ks, gaze=g, nR=nR, iP=iP, plan=res, masks=masks)
        if viz:                       # everything robot/report.py needs to draw the presentation figures
            rng = np.random.default_rng(0)
            sub = rng.choice(len(P), min(40000, len(P)), replace=False)
            pans, tilts, G = b["coarse_grid"]
            res["viz"] = dict(P=P[sub].round(4), w=w[sub].round(4), rgb=rgb[sub],
                              cam=cam, eye=eye, gaze=gaze, up=up, ref=ref, hfov=hfov, vfov=vfov, elev_now=elev_p,
                              room_centers=centers[:nR], room_fwd=fwd[:nR],
                              anchor_center=None if iA is None else centers[iA],
                              grid_pans=pans, grid_tilts=tilts, grid=G.round(4),
                              bbox=g["bbox"] - np.array([ox, oy, ox, oy], np.float32))   # face box in the real photo
        return res

    # ------------------------------------------------------------------ step B: where is the person looking
    def target(self, photo, K_target, out_dir=None, viz=False):
        """photo taken after turning -> fused heat map (gaze cone x DeepGaze IIE), peak pixel, overlay image.
        viz=True also returns the 3D view of this step (room + the turned photo coloured by the heat map, eye, gaze,
        the predicted target as a 3D point); all in this VGGT run's own frame."""
        st = self.state
        t0 = time.time()
        rgbT = load_rgb(photo)
        bgrT = rgbT[:, :, ::-1].copy()
        H0, W0 = bgrT.shape[:2]
        iP, iT = st["nR"], st["nR"] + 1
        if st.get("mode") == "map":           # saved room map (+ anchor) + robot photos, all padded to squares
            iP, iT = st["iP"], len(st["rgbs"])
            sqT, _KT, (ox, oy), _m = pad_square(rgbT, K_target)
            v = self._vggt(st["rgbs"] + [sqT], SQ, SQ)
            eye, gaze = self._eye_gaze(st["gaze"], v, iP, st["rgbs"][iP], st["Ks"][iP], SQ, SQ)
            dir_cam = F.gaze_dir_cam_from_dir_world(gaze, v["extr"][iT])
            theta_net = F.theta_map_from_eye_and_worldpoints(v["wp"][iT], v["extr"][iT], eye, dir_cam)
            S0 = sqT.shape[0]
            theta_full = cv2.resize(theta_net, (S0, S0), interpolation=cv2.INTER_LINEAR)[oy:oy + H0, ox:ox + W0]
            Wn = Hn = SQ
            to_orig = lambda u, vv: (u * S0 / SQ - ox, vv * S0 / SQ - oy)          # noqa: E731  net px -> photo px
            to_net = lambda x, y: ((x + ox) * SQ / S0, (y + oy) * SQ / S0)         # noqa: E731
            room_masks = st.get("masks")
        else:
            rgbs = st["rgbs"] + [rgbT]
            h0, w0 = rgbs[0].shape[:2]
            W, H = net_size(w0, h0)
            hp, wpx = rgbs[iP].shape[:2]
            Kp = K_net(st["K_person"][0, 0], st["K_person"][1, 1], st["K_person"][0, 2], st["K_person"][1, 2], wpx, hp, W, H)
            v = self._vggt(rgbs, W, H)        # new run: the target photo gets a 3D point per pixel, same frame as the person
            eye, gaze = self._eye_gaze(st["gaze"], v, iP, rgbs[iP], Kp, W, H)
            dir_cam = F.gaze_dir_cam_from_dir_world(gaze, v["extr"][iT])
            theta_net = F.theta_map_from_eye_and_worldpoints(v["wp"][iT], v["extr"][iT], eye, dir_cam)
            theta_full = cv2.resize(theta_net, (W0, H0), interpolation=cv2.INTER_LINEAR)
            Wn, Hn = W, H
            to_orig = lambda u, vv: (u * W0 / W, vv * H0 / H)                      # noqa: E731
            to_net = lambda x, y: (x * W / W0, y * H / H0)                         # noqa: E731
            room_masks = None
        half = st["gaze"]["half"]
        sigma = F.sigma_from_cone_quantile(np.deg2rad(half), F.CONE_QUANTILE)
        # DeepGaze on the cone-conditioned photo (as final_form.py: outside the cone darkened)
        mask_u8 = ((theta_full <= np.deg2rad(half)) * 255).astype(np.uint8)
        img = F.blend_image_with_mask(bgrT, mask_u8, outside_dark=F.MASK_OUTSIDE_DARK)
        if max(H0, W0) > F.DEEPGAZE_MAX_DIM:
            sc = F.DEEPGAZE_MAX_DIM / max(H0, W0)
            img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        Hd, Wd = img.shape[:2]
        cb = cv2.resize(self.centerbias, (Wd, Hd), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        cb -= logsumexp(cb)
        with torch.inference_mode():
            ld = self.dg(torch.from_numpy(img.transpose(2, 0, 1).copy()).unsqueeze(0).to(DEV).half(),
                         torch.from_numpy(cb).unsqueeze(0).to(DEV).half())[0, 0].float().cpu().numpy()
        S = F.normalize_prob_map(np.exp(ld - ld.max()))
        theta = cv2.resize(theta_full, (Wd, Hd), interpolation=cv2.INTER_LINEAR)
        G = F.normalize_prob_map(F.geometry_likelihood(theta, sigma))
        Hf = F.fuse_poe(G, S, F.ALPHA, F.BETA)
        _Hobj, (py, px), _region = F.extract_single_object_heatmap(Hf, F.SINGLEOBJ_REL_THRESH, F.SINGLEOBJ_SMOOTH_SIGMA,
                                                                   F.SINGLEOBJ_MIN_AREA, F.SINGLEOBJ_MORPH_CLOSE)
        peak = (int(px * W0 / Wd), int(py * H0 / Hd))
        cone_share = float((mask_u8 > 0).mean())

        heat = cv2.resize(Hf / Hf.max(), (W0, H0))
        over = cv2.addWeighted(bgrT, 0.45, cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET), 0.55, 0)
        cnt, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cnt, -1, (255, 255, 255), 2)
        cv2.drawMarker(over, peak, (0, 255, 0), cv2.MARKER_CROSS, 60, 4)
        cv2.putText(over, "predicted gaze target", (peak[0] + 20, peak[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        res = dict(peak_px=peak, cone_share_of_photo=cone_share, secs=round(time.time() - t0, 2))
        if viz:
            wpT, s = v["wp"][iT], 2
            vv, uu = np.mgrid[0:Hn:s, 0:Wn:s]
            xo, yo = to_orig(uu + 0.5, vv + 0.5)
            ok = (xo >= 0) & (xo < W0) & (yo >= 0) & (yo < H0) & np.isfinite(wpT[::s, ::s]).all(-1)
            xi, yi = np.clip(xo[ok].astype(int), 0, W0 - 1), np.clip(yo[ok].astype(int), 0, H0 - 1)
            pu, pv = to_net(peak[0], peak[1])
            peak3d = wpT[int(np.clip(pv, 0, Hn - 1)), int(np.clip(pu, 0, Wn - 1))]
            # room from the map photos of this same VGGT run (for context)
            nR, Pr, Cr = st["nR"], [], []
            for k in range(nR):
                pk, ck = v["wp"][k, ::4, ::4].reshape(-1, 3), v["wconf"][k, ::4, ::4].reshape(-1)
                good = np.isfinite(pk).all(1)
                if room_masks is not None:
                    good &= room_masks[k][::4, ::4].reshape(-1)
                good &= ck >= np.quantile(ck[good], 1 - CONF_KEEP)
                Pr.append(pk[good]); Cr.append(v["ims"][k].permute(1, 2, 0).numpy()[::4, ::4].reshape(-1, 3)[good])
            extr = v["extr"]
            fwd = np.stack([e[2, :3] for e in extr])
            _, _, vt = np.linalg.svd(fwd[:nR])
            up = vt[2] if vt[2] @ (-extr[0][1, :3]) > 0 else -vt[2]
            centers = np.stack([-e[:3, :3].T @ e[:3, 3] for e in extr])
            res["viz"] = dict(PT=wpT[::s, ::s][ok].round(4), heatT=heat[yi, xi].round(4), rgbT=rgbT[yi, xi],
                              PR=np.concatenate(Pr).round(4), rgbR=(np.concatenate(Cr) * 255).astype(np.uint8),
                              eye=eye, gaze=gaze, half=half, up=up, peak3d=peak3d,
                              cam_person=centers[iP], fwd_person=fwd[iP], cam_target=centers[iT], fwd_target=fwd[iT],
                              R_target=extr[iT][:3, :3])
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, "target_overlay.jpg"), over)
            cv2.imwrite(os.path.join(out_dir, "target_cone_mask.png"), mask_u8)
            np.save(os.path.join(out_dir, "target_heat.npy"), Hf)
        return res, over
