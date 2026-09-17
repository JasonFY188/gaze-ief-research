"""
IEF per-fold pipeline — VGGT + IEF-ensemble + DeepGaze IIE

Runs each of the 15 training folds INDEPENDENTLY for both IEF variants
(ief_gaze360, ief_mpiigaze) plus one ENSEMBLE condition (averaged probs).

Per-fold output goes to:
  data_root/outputs_ief/<variant>/fold<N>/<subscenario>/<primary>/

Summary Excel files (one per person × distance) go to:
  ~/Documents/output of ief/

At the end a mean ± std table of angular_error_peak_deg is printed for
every variant × distance combination.
"""

from __future__ import annotations

import gc
import os
import re
import sys
import csv
import json
import math
import warnings
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import openpyxl
import torch
import torch.nn as nn
import torchvision
from openpyxl.styles import Font, PatternFill
from PIL import Image
from scipy.special import logsumexp

# Load machine-specific paths from paths.py (edit that file when moving machines).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from paths import (
    BASE_ROOT, IEF_REPO, L2CS_WEIGHTS, CENTERBIAS_NPY,
    GAZE360_BACKBONE, MPIIGAZE_BACKBONE_DIR, SUMMARY_OUT_DIR,
    VGGT_SOURCE, DEEPGAZE_CACHE,
)
del _HERE

# l2cs is installed from l2cs_net/ into the environment (see README), so no sys.path hack is needed.
from l2cs import Pipeline                                          # type: ignore
from l2cs.model import L2CS                                        # type: ignore
from l2cs.wrapper import L2CSWrapper                               # type: ignore
from l2cs.refinement_head import RefinementConfig, IEFGazeModel   # type: ignore

from vggt.models.vggt import VGGT                                  # type: ignore
from vggt.utils.load_fn import load_and_preprocess_images          # type: ignore
from vggt.utils.pose_enc import pose_encoding_to_extri_intri       # type: ignore
from blob_projection_utils import project_blob_to_images           # type: ignore
import deepgaze_pytorch                                             # type: ignore

warnings.filterwarnings("ignore", message=".*flash attention.*", category=UserWarning)

# =============================================================================
# CONFIG  (machine-specific paths come from paths.py imported above)
# =============================================================================

ONLY_SCENARIO_NAME: Optional[str]    = None
ONLY_SUBSCENARIO_NAME: Optional[str] = None

RESUME_AUTO       = True
ENSEMBLE_ONLY     = False  # True → skip per-fold DeepGaze, run only the ensemble (16× faster)
SAVE_INTERMEDIATES = False  # True → save .npy heatmaps + .png visualisations (~12 MB/condition)
DONE_MARKER       = "_DONE.json"
SKIP_MARKER       = "_SKIPPED.json"

REF_IMAGE_NAMES    = ["view1.jpg", "view2.jpg"]
DEEPGAZE_VIEW_IDXS = [2]
GAZE_VIEW_IDX      = 0

NUM_FOLDS = 15

IEF_VARIANTS: Dict[str, Dict] = {
    "ief_gaze360": {
        "backbone": GAZE360_BACKBONE,
        "ckpts": [f"{IEF_REPO}/checkpoints/ief_gaze360/best_fold{i}.pt" for i in range(NUM_FOLDS)],
        "shared_backbone": True,
    },
    "ief_mpiigaze": {
        "backbone_pattern": f"{MPIIGAZE_BACKBONE_DIR}/fold{{fold}}.pkl",
        "ckpts": [f"{IEF_REPO}/checkpoints/ief_mpiigaze/best_fold{i}.pt" for i in range(NUM_FOLDS)],
        "shared_backbone": False,
    },
}

# Conditions to run: individual folds + one ensemble.
# "ensemble" averages the softmax probs of all 15 folds before computing the cone.
FOLD_IDS: List[str] = [f"fold{i}" for i in range(NUM_FOLDS)] + ["ensemble"]

CONE_QUANTILE = 0.95
CONE_SAMPLES  = 2000
CONE_MIN_DEG  = 8.0
CONE_MAX_DEG  = 60.0

DEEPGAZE_MAX_DIM  = 1024
MASK_OUTSIDE_DARK = 0.0
BETA  = 1.0
ALPHA = 1.0
EPS   = 1e-12

SINGLEOBJ_ENABLE       = True
SINGLEOBJ_REL_THRESH   = 0.50
SINGLEOBJ_SMOOTH_SIGMA = 0.0
SINGLEOBJ_MIN_AREA     = 50
SINGLEOBJ_MORPH_CLOSE  = True

ANNOTATE_IF_MISSING = True
GT_FILENAME = "ground_truth_polygons.json"
DO_EVAL     = True

USE_SOFT_GT_FOR_METRICS = True
GT_SOFT_SIGMA_MODE   = "min_dim_frac"
GT_SOFT_SIGMA_VALUE  = 0.03
GT_SOFT_SIGMA_MIN_PX = 1.0
GT_SOFT_SIGMA_MAX_PX = 40.0


PERSON_LABEL_MAP: Dict[str, str] = {
    "1Senpai":  "senpai",
    "Moriyama": "moriyama",
    "Yoneyama": "yoneyama",
}

SUMMARY_AVG_COLS = [
    "cone_half_angle_deg",
    "auc", "ap",
    "dist_px_to_gt_centroid",
    "dist_px_to_gt_centroid_ief",
    "angular_error_ief_deg",
    "angular_error_peak_deg",
]

# =============================================================================
# DEVICE
# =============================================================================
device_str = "cuda" if torch.cuda.is_available() else "cpu"
torch_dev  = torch.device(device_str)
dtype      = torch.float16 if device_str == "cuda" else torch.float32

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# =============================================================================
# IEF MODEL LOADING
# =============================================================================
def _load_l2cs_backbone(pkl_path: str, num_bins: int) -> L2CS:
    base  = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins=num_bins)
    state = torch.load(pkl_path, map_location="cpu", weights_only=False)
    # Strip DataParallel "module." prefix WITHOUT using nn.DataParallel —
    # nn.DataParallel() auto-moves to GPU even when we want CPU-only loading.
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    base.load_state_dict(state)
    return base


def load_ief_variant(variant_name: str, vcfg: Dict, device: torch.device):
    """
    Returns (models, cfgs) — one IEFGazeModel per fold.

    ief_gaze360  : shared backbone on GPU, all 15 heads on GPU  (~250 MB VRAM total)
    ief_mpiigaze : each fold on CPU (different backbone per fold), moved to GPU during
                   inference to avoid 1.5 GB VRAM overhead.
    """
    models: List[IEFGazeModel] = []
    cfgs:   List[RefinementConfig] = []

    if vcfg["shared_backbone"]:
        print(f"  [{variant_name}] loading shared backbone …", flush=True)
        backbone = _load_l2cs_backbone(vcfg["backbone"], num_bins=90)
        wrapper  = L2CSWrapper(backbone).to(device).eval()
        for p in wrapper.parameters():
            p.requires_grad_(False)

        for fold, ckpt_path in enumerate(vcfg["ckpts"]):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg: RefinementConfig = ckpt["cfg"]
            m = IEFGazeModel(wrapper, cfg).to(device)
            m.head.load_state_dict(ckpt["head_state_dict"])
            m.eval()
            for p in m.head.parameters():
                p.requires_grad_(False)
            models.append(m); cfgs.append(cfg)
        print(f"  [{variant_name}] {len(models)} folds on GPU (shared backbone)", flush=True)
    else:
        for fold, ckpt_path in enumerate(vcfg["ckpts"]):
            backbone_path = vcfg["backbone_pattern"].format(fold=fold)
            if not os.path.isfile(backbone_path):
                print(f"  [{variant_name}] fold{fold}: backbone not found, skipping", flush=True)
                models.append(None); cfgs.append(None)   # placeholder so fold indices stay aligned
                continue
            backbone = _load_l2cs_backbone(backbone_path, num_bins=28)
            wrapper  = L2CSWrapper(backbone)  # stays on CPU
            for p in wrapper.parameters():
                p.requires_grad_(False)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg  = ckpt["cfg"]
            m    = IEFGazeModel(wrapper, cfg)
            m.head.load_state_dict(ckpt["head_state_dict"])
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            models.append(m); cfgs.append(cfg)
        n_loaded = sum(1 for m in models if m is not None)
        print(f"  [{variant_name}] {n_loaded}/{len(models)} folds loaded on CPU (diff backbone per fold)", flush=True)

    return models, cfgs


def _infer_fold(
    model: IEFGazeModel,
    cfg:   RefinementConfig,
    face_t: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run one IEFGazeModel and return (yaw_probs, pitch_probs).

    gaze360 models are pre-loaded on GPU → face_t goes to GPU.
    mpiigaze models stay on CPU → face_t runs on CPU (~100 ms per fold, acceptable).
    No model device movement is done here to avoid device-mismatch bugs.
    """
    model_device = next(model.parameters()).device
    with torch.no_grad():
        steps = model(face_t.to(model_device))
    pitch_logits, yaw_logits = steps[-1]
    yp = torch.softmax(yaw_logits,   dim=1)[0].cpu().numpy().astype(np.float32)
    pp = torch.softmax(pitch_logits, dim=1)[0].cpu().numpy().astype(np.float32)
    return yp, pp


# =============================================================================
# GEOMETRY + FUSION (same as final_form.py)
# =============================================================================
def gaze_vector_from_angles(yaw_rad, pitch_rad):
    x = np.sin(yaw_rad) * np.cos(pitch_rad)
    y = np.sin(pitch_rad)
    z = np.cos(pitch_rad) * np.cos(yaw_rad)
    g = np.array([x, y, z], dtype=np.float32)
    return -g / (np.linalg.norm(g) + 1e-9)


def _clip_bbox_xyxy(bbox, h, w):
    x1, y1, x2, y2 = bbox.astype(int).tolist()
    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h, y2))
    if x2 <= x1+2 or y2 <= y1+2:
        cx, cy = w//2, h//2; r = min(w,h)//4
        x1, x2 = max(0, cx-r), min(w, cx+r)
        y1, y2 = max(0, cy-r), min(h, cy+r)
    return x1, y1, x2, y2


def preprocess_face(img_bgr, bbox, dev, out_size=224):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = _clip_bbox_xyxy(bbox, h, w)
    face_vis = cv2.resize(img_bgr[y1:y2, x1:x2].copy(), (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    face_rgb = cv2.cvtColor(face_vis, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t - mean) / std, face_vis


def expected_angle_deg(probs, bin_angles_deg):
    p = probs.astype(np.float64); p /= p.sum() + 1e-12
    return float((p * bin_angles_deg.astype(np.float64)).sum())


def cone_half_angle_from_probs(yaw_probs, pitch_probs, bin_angles_deg,
                                num_samples=CONE_SAMPLES, quantile=CONE_QUANTILE,
                                min_deg=CONE_MIN_DEG, max_deg=CONE_MAX_DEG):
    yp = yaw_probs.astype(np.float64);   yp /= yp.sum() + 1e-12
    pp = pitch_probs.astype(np.float64); pp /= pp.sum() + 1e-12
    ys = np.random.choice(bin_angles_deg, size=num_samples, p=yp)
    ps = np.random.choice(bin_angles_deg, size=num_samples, p=pp)
    yr, pr = np.deg2rad(ys), np.deg2rad(ps)
    vecs = np.stack([-np.sin(yr)*np.cos(pr), -np.sin(pr), -np.cos(pr)*np.cos(yr)], axis=1).astype(np.float32)
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    mean = vecs.mean(axis=0); mean /= np.linalg.norm(mean) + 1e-9
    ang  = np.rad2deg(np.arccos(np.clip(vecs @ mean, -1.0, 1.0)))
    return float(np.clip(np.quantile(ang, quantile), min_deg, max_deg))


def estimate_eye_origin_world(bbox, world_points_view, orig_size, H_net, W_net, intrinsic, extrinsic):
    x1, y1, x2, y2 = map(float, bbox)
    H_i, W_i = orig_size
    cx_o = 0.5*(x1+x2); cy_o = 0.5*(y1+y2)
    ui = int(np.clip(round(cx_o*(W_net/W_i)), 0, W_net-1))
    vi = int(np.clip(round(cy_o*(H_net/H_i)), 0, H_net-1))
    pw = world_points_view[vi, ui]
    if not np.isfinite(pw).all():
        r = 2
        patch = world_points_view[max(0,vi-r):min(H_net,vi+r+1),
                                  max(0,ui-r):min(W_net,ui+r+1)].reshape(-1,3)
        fin = np.isfinite(patch).all(axis=1)
        pw = np.median(patch[fin], axis=0) if fin.any() else pw
    R = extrinsic[:3,:3]; t = extrinsic[:3,3]
    depth = float((R @ pw + t)[2])
    fx = float(intrinsic[0,0]); fy = float(intrinsic[1,1])
    cx = float(intrinsic[0,2]); cy = float(intrinsic[1,2])
    fxo = fx*(W_i/W_net); fyo = fy*(H_i/H_net)
    cxo = cx*(W_i/W_net); cyo = cy*(H_i/H_net)
    pc  = np.array([(cx_o-cxo)/(fxo+1e-12)*depth,
                    (cy_o-cyo)/(fyo+1e-12)*depth, depth], dtype=np.float32)
    return (R.T @ (pc - t)).astype(np.float32), (int(cx_o), int(cy_o))


def compute_cone_mask_for_points(pts, origin3d, dir3d, cone_deg, min_dist=0.20):
    vecs = pts - origin3d[None,:]; dist = np.linalg.norm(vecs, axis=1)
    vn = vecs / (dist[:,None] + 1e-9)
    ang = np.rad2deg(np.arccos(np.clip(vn @ dir3d, -1.0, 1.0)))
    return (ang <= cone_deg) & (dist > min_dist), dist


def gaze_dir_cam_from_dir_world(dir_world, extri_w2c):
    d = (extri_w2c[:3,:3].astype(np.float32) @ dir_world.astype(np.float32))
    return d / (np.linalg.norm(d) + 1e-9)


def theta_map_from_eye_and_worldpoints(wp, extri, eye_world, gdir_cam):
    R = extri[:3,:3].astype(np.float32); t = extri[:3,3].astype(np.float32)
    eye_cam = R @ eye_world.astype(np.float32) + t
    Pc = (R @ wp.reshape(-1,3).astype(np.float32).T).T + t[None,:]
    V  = Pc - eye_cam[None,:]
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    g  = gdir_cam.astype(np.float32); g /= np.linalg.norm(g) + 1e-9
    theta = np.arccos(np.clip(Vn @ g, -1.0, 1.0)).astype(np.float32).reshape(wp.shape[0], wp.shape[1])
    theta[~np.isfinite(theta)] = np.pi
    return theta


def sigma_from_cone_quantile(cone_half_rad, quantile):
    q = float(np.clip(quantile, 1e-4, 1-1e-4))
    z = float(np.sqrt(2.0) * torch.special.erfinv(torch.tensor(2.0*q-1.0)).item())
    return float(cone_half_rad / max(z, 1e-6))


def geometry_likelihood(theta, sigma):
    return np.exp(-0.5*(theta/(sigma+1e-12))**2).astype(np.float32)


def normalize_prob_map(x):
    x = x.astype(np.float32); s = float(np.sum(x))
    return x / s if (np.isfinite(s) and s > 0) else np.full_like(x, 1.0/x.size)


def fuse_poe(G, S, alpha, beta, eps=1e-12):
    logH = alpha*np.log(G+eps) + beta*np.log(S+eps)
    logH -= float(logsumexp(logH))
    return normalize_prob_map(np.exp(logH).astype(np.float32))


def overlay_heatmap_bgr(img, heat, out_path, alpha=0.6):
    u8  = (255.0*heat/(float(heat.max())+1e-12)).clip(0,255).astype(np.uint8)
    col = cv2.applyColorMap(u8, cv2.COLORMAP_MAGMA)
    out = (img.astype(np.float32)*(1-alpha) + col.astype(np.float32)*alpha).clip(0,255).astype(np.uint8)
    cv2.imwrite(out_path, out)


def _mask_to_u8(mask):
    if mask.ndim == 3: mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype == np.bool_: return (mask.astype(np.uint8)*255)
    if mask.dtype == np.uint8: return mask
    m = mask.astype(np.float32)
    if float(np.nanmax(m)) <= 1.0+1e-6: m = m*255.0
    return np.clip(m, 0, 255).astype(np.uint8)


def blend_image_with_mask(img_bgr, mask_u8, outside_dark=0.20):
    a = (mask_u8.astype(np.float32)/255.0)[...,None]; b = img_bgr.astype(np.float32)
    return np.clip(b*a + b*float(outside_dark)*(1-a), 0, 255).astype(np.uint8)


# =============================================================================
# SINGLE-OBJECT EXTRACTION
# =============================================================================
def extract_single_object_heatmap(H, rel_thresh=0.5, smooth_sigma=1.2, min_area=30, use_morph_close=True):
    H = H.astype(np.float32)
    Hs = cv2.GaussianBlur(H, (0,0), smooth_sigma, smooth_sigma) if smooth_sigma and smooth_sigma > 0 else H
    py, px = np.unravel_index(int(np.argmax(Hs)), Hs.shape)
    peak_val = float(Hs[py, px])
    if not np.isfinite(peak_val) or peak_val <= 0:
        return np.full_like(H, 1.0/H.size), (py, px), np.ones_like(H, dtype=bool)
    mask = (Hs >= float(rel_thresh)*peak_val).astype(np.uint8)
    if use_morph_close:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        H_obj = H.copy(); H_obj[mask==0] = 0.0
        return H_obj/(float(H_obj.sum())+1e-12), (py, px), mask.astype(bool)
    best = -1; best_a = 0
    for cid in range(1, n):
        if labels[py, px] == cid: best = cid; break
        if stats[cid, cv2.CC_STAT_AREA] > best_a: best_a = stats[cid, cv2.CC_STAT_AREA]; best = cid
    region = (labels == best)
    if int(region.sum()) < max(min_area, 1): region = mask.astype(bool)
    H_obj = H.copy(); H_obj[~region] = 0.0; H_obj /= (float(H_obj.sum())+1e-12)
    py2, px2 = np.unravel_index(int(np.argmax(H_obj * region)), H_obj.shape)
    return H_obj.astype(np.float32), (int(py2), int(px2)), region.astype(bool)


# =============================================================================
# METRICS
# =============================================================================
def _soft_gt_sigma_px(Hd, Wd):
    s = float(GT_SOFT_SIGMA_VALUE)*(float(min(Hd,Wd)) if GT_SOFT_SIGMA_MODE != "pixels" else 1.0)
    return float(np.clip(s, GT_SOFT_SIGMA_MIN_PX, GT_SOFT_SIGMA_MAX_PX))


def _make_soft_gt(gt_u8, Hd, Wd):
    gt = (gt_u8 > 0).astype(np.float32)
    if gt.sum() <= 0: return gt
    s = _soft_gt_sigma_px(Hd, Wd)
    if s > 0: gt = cv2.GaussianBlur(gt, (0,0), s, s)
    mx = float(gt.max()); return (gt/mx if mx > 0 else gt).astype(np.float32)


def _weighted_auc(scores, pos_w, eps=1e-12):
    s = scores.astype(np.float64).ravel(); wp = pos_w.astype(np.float64).ravel(); wn = 1-wp
    Wp, Wn = float(wp.sum()), float(wn.sum())
    if Wp <= eps or Wn <= eps: return None
    order = np.argsort(s); wps = wp[order]; wns = wn[order]; cneg = np.cumsum(wns)
    auc = 0.0; i = 0; N = s.size
    while i < N:
        j = i
        while j+1 < N and s[order[j+1]] == s[order[i]]: j += 1
        auc += float(wps[i:j+1].sum()) * 0.5*(cneg[i-1] if i > 0 else 0.0)+cneg[j]
        i = j+1
    # fix: correct formula
    i = 0; auc = 0.0
    while i < N:
        j = i
        while j+1 < N and s[order[j+1]] == s[order[i]]: j += 1
        cb = cneg[i-1] if i > 0 else 0.0
        auc += float(wps[i:j+1].sum()) * 0.5*(cb + cneg[j])
        i = j+1
    return float(auc / (Wp*Wn))


def _weighted_ap(scores, pos_w, eps=1e-12):
    s = scores.astype(np.float64).ravel(); wp = pos_w.astype(np.float64).ravel()
    P = float(wp.sum())
    if P <= eps: return None
    wn = 1-wp; order = np.argsort(-s); wp = wp[order]; wn = wn[order]
    tp = np.cumsum(wp); fp = np.cumsum(wn); prec = tp/np.maximum(tp+fp, eps); rec = tp/P
    ap = 0.0; pr = 0.0
    for p, r in zip(prec, rec): ap += float(p)*float(r-pr); pr = float(r)
    return float(ap)


def _mask_centroid(mask_u8):
    ys, xs = np.where(mask_u8 > 0)
    return (float(xs.mean()), float(ys.mean())) if xs.size else None


def _angle_deg_between(a, b):
    a = a.astype(np.float64)/(np.linalg.norm(a)+1e-12)
    b = b.astype(np.float64)/(np.linalg.norm(b)+1e-12)
    return float(np.rad2deg(np.arccos(float(np.clip(np.dot(a,b),-1.0,1.0)))))


# =============================================================================
# GROUND TRUTH
# =============================================================================
def _poly_to_mask(H, W, poly_xy):
    mask = np.zeros((H, W), dtype=np.uint8)
    if poly_xy and len(poly_xy) >= 3:
        cv2.fillPoly(mask, [np.array(poly_xy, dtype=np.int32).reshape(-1,1,2)], 255)
    return mask


def _annotate_polygon_on_image(ref_bgr, win_name):
    points: List[Tuple[int,int]] = []; saved = None; disp = ref_bgr.copy()
    def redraw():
        nonlocal disp; d = ref_bgr.copy()
        cv2.rectangle(d,(5,5),(625,155),(0,0,0),-1); cv2.rectangle(d,(5,5),(625,155),(255,255,255),2)
        for i,ln in enumerate(["ANNOTATION","Left click: add  Right click: save","u:undo r:reset s:save q:quit",f"Points:{len(points)}"]):
            cv2.putText(d,ln,(15,35+i*26),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2,cv2.LINE_AA)
        for i,(x,y) in enumerate(points):
            cv2.circle(d,(x,y),5,(0,0,255),-1)
            if i > 0: cv2.line(d,points[i-1],points[i],(0,255,0),2)
        if len(points)>=3: cv2.line(d,points[-1],points[0],(0,255,0),1)
        disp = d
    def on_mouse(ev,x,y,fl,_):
        nonlocal saved
        if ev==cv2.EVENT_LBUTTONDOWN: points.append((int(x),int(y))); redraw()
        elif ev==cv2.EVENT_RBUTTONDOWN and len(points)>=3: saved=[[int(px),int(py)] for px,py in points]
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL); cv2.setMouseCallback(win_name, on_mouse); redraw()
    while True:
        cv2.imshow(win_name, disp); k = cv2.waitKey(20)&0xFF
        if saved is not None: break
        if k==ord('u') and points: points.pop(); redraw()
        elif k==ord('r'): points.clear(); redraw()
        elif k==ord('s') and len(points)>=3: saved=[[int(px),int(py)] for px,py in points]; break
        elif k in (ord('q'),27): break
    cv2.destroyWindow(win_name); return saved


def _make_contact_sheet(primary_paths, max_imgs=6, thumb_w=480):
    thumbs = []
    for p in primary_paths[:max_imgs]:
        bgr = np.array(Image.open(p).convert("RGB"))[:,:,::-1]
        h,w = bgr.shape[:2]; t = cv2.resize(bgr,(thumb_w,int(round(h*thumb_w/w))),interpolation=cv2.INTER_AREA)
        bar = np.zeros((34,thumb_w,3),dtype=np.uint8)
        cv2.putText(bar,os.path.basename(p),(10,24),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2,cv2.LINE_AA)
        thumbs.append(np.vstack([bar,t]))
    if not thumbs: return np.zeros((200,600,3),dtype=np.uint8)
    def pad(im,H): return im if im.shape[0]>=H else np.vstack([im,np.zeros((H-im.shape[0],im.shape[1],3),dtype=np.uint8)])
    cols=3; rows=int(np.ceil(len(thumbs)/cols)); out=[]
    for r in range(rows):
        ch=thumbs[r*cols:(r+1)*cols]; Hm=max(im.shape[0] for im in ch)
        ch=[pad(im,Hm) for im in ch]
        while len(ch)<cols: ch.append(np.zeros((Hm,thumb_w,3),dtype=np.uint8))
        out.append(np.hstack(ch))
    return np.vstack(out)


def ensure_ground_truth_for_scenario(scenario_name, scenario_dir, scenario_out_dir,
                                     ref_paths, deepgaze_view_idxs, primary_paths, gt_root):
    gt_old = os.path.join(gt_root, scenario_name, GT_FILENAME)
    if os.path.isfile(gt_old):
        with open(gt_old) as f: return json.load(f)
    gt_new = os.path.join(scenario_out_dir, GT_FILENAME)
    if os.path.isfile(gt_new):
        with open(gt_new) as f: return json.load(f)
    if not ANNOTATE_IF_MISSING: return {}
    gt = {}
    sheet = _make_contact_sheet(primary_paths)
    cv2.namedWindow("SCENARIO CONTEXT", cv2.WINDOW_NORMAL); cv2.imshow("SCENARIO CONTEXT", sheet); cv2.waitKey(1)
    for v in deepgaze_view_idxs:
        if v >= len(ref_paths): continue
        ref_bgr = np.array(Image.open(ref_paths[v]).convert("RGB"))[:,:,::-1]
        poly = _annotate_polygon_on_image(ref_bgr, f"Annotate GT view_{v} — {scenario_name}")
        if poly and len(poly) >= 3:
            gt[f"view_{v}"] = {"polygon_fullres": poly, "ref_path": ref_paths[v]}
    cv2.destroyWindow("SCENARIO CONTEXT")
    with open(gt_new,"w") as f: json.dump(gt, f, indent=2)
    try:
        os.makedirs(os.path.dirname(gt_old), exist_ok=True)
        with open(gt_old,"w") as f: json.dump(gt, f, indent=2)
    except Exception: pass
    return gt


# =============================================================================
# INPUT DISCOVERY + RESUME
# =============================================================================
def list_images_in_dir(d):
    paths = []
    for e in ["*.png","*.jpg","*.jpeg","*.PNG","*.JPG","*.JPEG"]:
        paths += glob(os.path.join(d, e))
    return sorted(paths)


def discover_scenario_roots(base):
    ref_lower = {n.lower() for n in REF_IMAGE_NAMES}; roots = []
    for dp, dns, fns in os.walk(base):
        dns[:] = sorted(d for d in dns if not d.startswith("."))
        if ref_lower.issubset({f.lower() for f in fns}):
            roots.append(dp); dns.clear()
    return sorted(roots)


def list_scenario_dirs(input_root, output_root):
    dirs = []
    for p in sorted(glob(os.path.join(input_root, "*"))):
        if not os.path.isdir(p) or os.path.basename(p).startswith("."): continue
        if os.path.abspath(p) == os.path.abspath(output_root): continue
        if list_images_in_dir(p): dirs.append(p)
    return dirs


def _already_done(d):   return os.path.isfile(os.path.join(d, DONE_MARKER))
def _already_skipped(d): return os.path.isfile(os.path.join(d, SKIP_MARKER))


def _write_marker(d, which, payload):
    try:
        with open(os.path.join(d, which), "w") as f: json.dump(payload, f, indent=2)
    except Exception: pass


# =============================================================================
# CORE PIPELINE
# =============================================================================
def _run_shared_features(primary_path, ref_paths, out_dir_base, vggt_model, l2cs_pipeline):
    """VGGT + face detection. Returns shared dict or None on failure."""
    os.makedirs(out_dir_base, exist_ok=True)
    image_paths = [primary_path] + ref_paths

    try:
        orig_images, orig_sizes = [], []
        for p in image_paths:
            arr = np.array(Image.open(p).convert("RGB"))
            orig_images.append(arr); orig_sizes.append((arr.shape[0], arr.shape[1]))
        images = load_and_preprocess_images(image_paths).to(device_str)
        _S, _C, H_net, W_net = images.shape
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=(device_str=="cuda"), dtype=dtype):
                predictions = vggt_model(images)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        world_points = predictions["world_points"].detach().cpu().numpy()[0]
        extrinsic_np = extrinsic.detach().cpu().numpy()[0]
        intrinsic_np = intrinsic.detach().cpu().numpy()[0]
        del images, predictions, extrinsic, intrinsic
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception as e:
        _write_marker(out_dir_base, SKIP_MARKER, {"primary": primary_path, "status": "vggt_failed", "error": str(e)})
        return None

    gaze_bgr = orig_images[GAZE_VIEW_IDX][:,:,::-1].copy()
    try:
        res = l2cs_pipeline.step(gaze_bgr)
    except Exception as e:
        _write_marker(out_dir_base, SKIP_MARKER, {"primary": primary_path, "status": "detection_failed", "error": str(e)})
        return None

    if res.bboxes is None or len(res.bboxes) == 0:
        _write_marker(out_dir_base, SKIP_MARKER, {"primary": primary_path, "status": "no_face"})
        return None

    bbox = res.bboxes[0]
    h0, w0 = gaze_bgr.shape[:2]
    x1, y1, x2, y2 = _clip_bbox_xyxy(np.array(bbox, dtype=np.float32), h0, w0)
    if SAVE_INTERMEDIATES:
        cv2.imwrite(os.path.join(out_dir_base, "face_crop_raw.png"), gaze_bgr[y1:y2, x1:x2].copy())

    face_t_cpu, face_vis = preprocess_face(gaze_bgr, np.array(bbox, dtype=np.float32), torch.device("cpu"))
    face_t = face_t_cpu.unsqueeze(0)   # (1,3,224,224) on CPU, moved to GPU by _infer_fold
    if SAVE_INTERMEDIATES:
        cv2.imwrite(os.path.join(out_dir_base, "face_crop_224.png"), face_vis)

    origin3d, _ = estimate_eye_origin_world(
        bbox=bbox, world_points_view=world_points[GAZE_VIEW_IDX],
        orig_size=orig_sizes[GAZE_VIEW_IDX],
        H_net=H_net, W_net=W_net,
        intrinsic=intrinsic_np[GAZE_VIEW_IDX], extrinsic=extrinsic_np[GAZE_VIEW_IDX],
    )

    return dict(orig_images=orig_images, orig_sizes=orig_sizes,
                world_points=world_points, extrinsic_np=extrinsic_np,
                intrinsic_np=intrinsic_np, H_net=H_net, W_net=W_net,
                origin3d=origin3d, face_t=face_t)


def _run_condition(
    shared, yaw_probs, pitch_probs, ief_cfg: RefinementConfig,
    condition_name, primary_path, out_dir,
    dg_model, centerbias_template, gt_for_scenario, metrics_rows,
):
    """DeepGaze + fusion + eval for one (variant, fold_id or 'ensemble') condition."""
    os.makedirs(out_dir, exist_ok=True)

    wp = shared["world_points"]; oi = shared["orig_images"]
    os_ = shared["orig_sizes"]; ex = shared["extrinsic_np"]; ix = shared["intrinsic_np"]
    H_net = shared["H_net"]; W_net = shared["W_net"]; origin3d = shared["origin3d"]

    bin_angles_deg = (np.arange(ief_cfg.num_bins, dtype=np.float32)
                      * ief_cfg.bin_width_deg - ief_cfg.angle_offset_deg)

    yaw_deg   = expected_angle_deg(yaw_probs, bin_angles_deg)
    pitch_deg = expected_angle_deg(pitch_probs, bin_angles_deg)

    cone_deg = cone_half_angle_from_probs(yaw_probs, pitch_probs, bin_angles_deg)
    cone_rad = float(np.deg2rad(cone_deg))
    sigma    = sigma_from_cone_quantile(cone_rad, CONE_QUANTILE)

    dir_cam  = gaze_vector_from_angles(float(np.deg2rad(yaw_deg)), float(np.deg2rad(pitch_deg)))
    dir_cam /= np.linalg.norm(dir_cam) + 1e-9
    dir_world = (ex[GAZE_VIEW_IDX][:3,:3].T @ dir_cam).astype(np.float32)
    dir_world /= np.linalg.norm(dir_world) + 1e-9

    pts_all = wp.reshape(-1,3)
    inside, _ = compute_cone_mask_for_points(pts_all, origin3d, dir_world, cone_deg)
    cone_pts  = pts_all[inside] if inside.sum() > 0 else np.zeros((0,3), dtype=np.float32)

    _proj, pmasks = project_blob_to_images(
        blob_points=cone_pts, orig_images=oi, orig_sizes=os_,
        extrinsic_np=ex, intrinsic_np=ix, H_net=H_net, W_net=W_net, mask_method="kde",
    )

    with open(os.path.join(out_dir, "run_summary.txt"), "w") as f:
        f.write(f"primary={primary_path}\ncondition={condition_name}\n"
                f"yaw_deg={yaw_deg:.3f}\npitch_deg={pitch_deg:.3f}\n"
                f"cone_half_angle_deg={cone_deg:.3f}\nnum_bins={ief_cfg.num_bins}\n"
                f"bin_width_deg={ief_cfg.bin_width_deg}\n")

    for v in DEEPGAZE_VIEW_IDXS:
        mask_u8 = _mask_to_u8(pmasks[v])
        orig_bgr = oi[v][:,:,::-1].copy(); H0, W0 = orig_bgr.shape[:2]
        if mask_u8.shape[:2] != (H0, W0):
            mask_u8 = cv2.resize(mask_u8, (W0, H0), interpolation=cv2.INTER_NEAREST)

        if SAVE_INTERMEDIATES:
            cv2.imwrite(os.path.join(out_dir, f"gaze_cone_mask_view_{v}.png"), mask_u8)
        cone_bgr = blend_image_with_mask(orig_bgr, mask_u8, outside_dark=MASK_OUTSIDE_DARK)
        if SAVE_INTERMEDIATES:
            cv2.imwrite(os.path.join(out_dir, f"cone_cond_fullres_view_{v}.jpg"), cone_bgr)

        image = cone_bgr
        if max(H0, W0) > DEEPGAZE_MAX_DIM:
            sc = DEEPGAZE_MAX_DIM/float(max(H0,W0))
            image = cv2.resize(cone_bgr, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        Hd, Wd = image.shape[:2]
        if SAVE_INTERMEDIATES:
            cv2.imwrite(os.path.join(out_dir, f"cone_cond_view_{v}.jpg"), image)

        cb = cv2.resize(centerbias_template, (Wd, Hd), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        cb -= logsumexp(cb)

        img_t = torch.from_numpy(image.transpose(2,0,1)).unsqueeze(0).to(device_str)
        cb_t  = torch.from_numpy(cb).unsqueeze(0).to(device_str)
        if device_str == "cuda": img_t = img_t.half(); cb_t = cb_t.half()
        else: img_t = img_t.float(); cb_t = cb_t.float()

        with torch.inference_mode():
            ld = dg_model(img_t, cb_t)

        ldnp  = ld[0,0].float().cpu().numpy().astype(np.float32)
        S_sal = normalize_prob_map(np.exp(ldnp - float(ldnp.max())).astype(np.float32))

        dir_cam_v = gaze_dir_cam_from_dir_world(dir_world, ex[v])
        theta_net = theta_map_from_eye_and_worldpoints(wp[v], ex[v], origin3d, dir_cam_v)
        theta     = cv2.resize(theta_net, (Wd, Hd), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        G = normalize_prob_map(geometry_likelihood(theta, sigma))
        H_fused = fuse_poe(G, S_sal, alpha=float(ALPHA), beta=BETA, eps=EPS)

        if SAVE_INTERMEDIATES:
            np.save(os.path.join(out_dir, f"fused_H_view_{v}.npy"), H_fused)
            np.save(os.path.join(out_dir, f"geom_G_view_{v}.npy"), G)
            np.save(os.path.join(out_dir, f"deepgaze_sal_view_{v}.npy"), S_sal)
            overlay_heatmap_bgr(image, S_sal,   os.path.join(out_dir, f"_dg_sal_view_{v}.png"))
            overlay_heatmap_bgr(image, G,       os.path.join(out_dir, f"_geom_view_{v}.png"))
            overlay_heatmap_bgr(image, H_fused, os.path.join(out_dir, f"_fused_view_{v}.png"))

        H_used = H_fused; peak_xy: Optional[Tuple[int,int]] = None
        if SINGLEOBJ_ENABLE:
            H_s, (py, px), _ = extract_single_object_heatmap(
                H_fused, rel_thresh=SINGLEOBJ_REL_THRESH, smooth_sigma=SINGLEOBJ_SMOOTH_SIGMA,
                min_area=SINGLEOBJ_MIN_AREA, use_morph_close=SINGLEOBJ_MORPH_CLOSE,
            )
            H_used = H_s; peak_xy = (px, py)
            if SAVE_INTERMEDIATES:
                np.save(os.path.join(out_dir, f"fused_singleobj_view_{v}.npy"), H_s)
                overlay_heatmap_bgr(image, H_s, os.path.join(out_dir, f"_fused_so_view_{v}.png"))

        if DO_EVAL and gt_for_scenario and f"view_{v}" in gt_for_scenario:
            poly_full = gt_for_scenario[f"view_{v}"]["polygon_fullres"]
            gt_full   = _poly_to_mask(H0, W0, poly_full)
            gt_dg     = cv2.resize(gt_full, (Wd, Hd), interpolation=cv2.INTER_NEAREST)
            scores    = H_used.astype(np.float32)

            if USE_SOFT_GT_FOR_METRICS:
                gt_soft = _make_soft_gt(gt_dg, Hd, Wd)
                auc = _weighted_auc(scores, gt_soft); ap = _weighted_ap(scores, gt_soft)
                if SAVE_INTERMEDIATES: cv2.imwrite(os.path.join(out_dir, f"gt_soft_view_{v}.png"),
                            (255.0*gt_soft/(float(gt_soft.max())+1e-12)).astype(np.uint8))
            else:
                lbl = (gt_dg>0).astype(np.float32)
                auc = _weighted_auc(scores, lbl); ap = _weighted_ap(scores, lbl)

            if peak_xy is None:
                py2, px2 = np.unravel_index(int(np.argmax(scores)), scores.shape)
                peak_xy  = (int(px2), int(py2))

            c       = _mask_centroid(gt_dg)
            dist_px = None if c is None else float(np.hypot(peak_xy[0]-c[0], peak_xy[1]-c[1]))

            # 3-D angular errors via VGGT world map
            sx = W_net/float(os_[v][1]); sy = H_net/float(os_[v][0])
            poly_net = [[int(round(x*sx)), int(round(y*sy))] for x,y in poly_full]
            gt_net   = _poly_to_mask(H_net, W_net, poly_net)
            pts      = wp[v][gt_net > 0].reshape(-1,3)
            fin      = np.isfinite(pts).all(axis=1)

            ang_err_ief  = None
            ang_err_peak = None
            if fin.any():
                cen3d  = np.median(pts[fin], axis=0).astype(np.float32)
                dir_gt = (cen3d - origin3d).astype(np.float32)
                if np.isfinite(dir_gt).all() and np.linalg.norm(dir_gt) > 1e-9:
                    ang_err_ief = _angle_deg_between(dir_world, dir_gt)
                    # Peak pixel → 3-D world point → angle to GT centroid
                    pxn = int(np.clip(round(peak_xy[0]*W_net/Wd), 0, W_net-1))
                    pyn = int(np.clip(round(peak_xy[1]*H_net/Hd), 0, H_net-1))
                    pw3 = wp[v][pyn, pxn]
                    if np.isfinite(pw3).all():
                        dp = (pw3 - origin3d).astype(np.float32)
                        if np.linalg.norm(dp) > 1e-9:
                            ang_err_peak = _angle_deg_between(dp, dir_gt)

            dist_px_ief = None
            if c is not None:
                iy, ix = np.unravel_index(int(np.argmin(theta)), theta.shape)
                dist_px_ief = float(np.hypot(int(ix)-c[0], int(iy)-c[1]))

            metrics_rows.append({
                "primary_path":              primary_path,
                "cone_config":               condition_name,
                "view_idx":                  v,
                "yaw_deg":                   float(yaw_deg),
                "pitch_deg":                 float(pitch_deg),
                "cone_half_angle_deg":       float(cone_deg),
                "sigma_theta_deg":           float(np.rad2deg(sigma)),
                "alpha":                     float(ALPHA),
                "beta":                      float(BETA),
                "auc":                       auc,
                "ap":                        ap,
                "peak_x":                    int(peak_xy[0]),
                "peak_y":                    int(peak_xy[1]),
                "dist_px_to_gt_centroid":    dist_px,
                "dist_px_to_gt_centroid_ief": dist_px_ief,
                "angular_error_ief_deg":     ang_err_ief,
                "angular_error_peak_deg":    ang_err_peak,
                "gt_soft_sigma_px": float(_soft_gt_sigma_px(Hd,Wd)) if USE_SOFT_GT_FOR_METRICS else None,
                "ief_variant":               condition_name.rsplit("_",1)[0] if "_" in condition_name else condition_name,
                "fold_id":                   condition_name.rsplit("_",1)[-1] if "_" in condition_name else "ensemble",
            })

        del img_t, cb_t, ld
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    _write_marker(out_dir, DONE_MARKER, {"primary": primary_path, "condition": condition_name, "status": "done"})


def run_one_primary(
    primary_path, ref_paths, out_dir_base, vggt_model, l2cs_pipeline,
    ief_models, dg_model, centerbias_template, gt_for_scenario,
    metrics_by_condition,   # dict: condition_name → list of row dicts
):
    os.makedirs(out_dir_base, exist_ok=True)

    # Early exit: all conditions already done
    all_done = all(
        _already_done(os.path.join(out_dir_base, cname))
        for cname in metrics_by_condition
    )
    if RESUME_AUTO and all_done:
        return

    shared = _run_shared_features(primary_path, ref_paths, out_dir_base, vggt_model, l2cs_pipeline)
    if shared is None:
        return
    face_t = shared["face_t"]

    for variant_name, (models, cfgs) in ief_models.items():
        # Collect per-fold probs; skip folds whose backbone was missing (model=None)
        per_fold_probs: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
        ref_cfg = None
        for fold_idx, (m, cfg) in enumerate(zip(models, cfgs)):
            if m is None:
                per_fold_probs.append(None)
                continue
            yp, pp = _infer_fold(m, cfg, face_t, torch_dev)
            per_fold_probs.append((yp, pp))
            if ref_cfg is None:
                ref_cfg = cfg

        if ref_cfg is None:
            print(f"  [{variant_name}] no models loaded, skipping", flush=True)
            continue

        # Ensemble probs (average over available folds only)
        valid = [p for p in per_fold_probs if p is not None]
        avg_yaw   = np.mean([p[0] for p in valid], axis=0).astype(np.float32)
        avg_pitch = np.mean([p[1] for p in valid], axis=0).astype(np.float32)

        # Run DeepGaze for each available fold + ensemble
        conditions_to_run = []
        if not ENSEMBLE_ONLY:
            for i, fp in enumerate(per_fold_probs):
                if fp is not None:
                    conditions_to_run.append((f"{variant_name}_fold{i}", fp[0], fp[1]))
        conditions_to_run.append((f"{variant_name}_ensemble", avg_yaw, avg_pitch))

        for cname, yaw_p, pitch_p in conditions_to_run:
            cond_out = os.path.join(out_dir_base, cname)
            if RESUME_AUTO and _already_done(cond_out):
                print(f"        [skip] {cname}", flush=True)
                continue
            print(f"        [run ] {cname}", flush=True)
            try:
                _run_condition(
                    shared=shared, yaw_probs=yaw_p, pitch_probs=pitch_p,
                    ief_cfg=ref_cfg, condition_name=cname,
                    primary_path=primary_path, out_dir=cond_out,
                    dg_model=dg_model, centerbias_template=centerbias_template,
                    gt_for_scenario=gt_for_scenario,
                    metrics_rows=metrics_by_condition[cname],
                )
            except Exception as e:
                _write_marker(cond_out, SKIP_MARKER,
                              {"primary": primary_path, "condition": cname, "error": str(e)})
                print(f"        [ERR ] {cname}: {e}", flush=True)

    del shared; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


# =============================================================================
# EXCEL SUMMARY
# =============================================================================
def write_person_distance_excel(person_label, distance, data_by_cond, out_dir):
    """One Excel per person×distance. One sheet per condition (variant_fold* / variant_ensemble)."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{person_label} {distance}.xlsx")
    wb = openpyxl.Workbook(); wb.remove(wb.active)
    hfill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    afill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")

    for cond_name, subscenario_map in sorted(data_by_cond.items()):
        ws = wb.create_sheet(title=cond_name[:31])
        all_rows = [r for rows in subscenario_map.values() for r in rows]
        if not all_rows: continue
        fields = list(all_rows[0].keys()); excel_row = 1

        for ssname, rows in sorted(subscenario_map.items()):
            if not rows: continue
            for ci, fn in enumerate(fields, 1):
                cell = ws.cell(row=excel_row, column=ci, value=fn)
                cell.font = Font(bold=True); cell.fill = hfill
            excel_row += 1
            for row_dict in rows:
                for ci, fn in enumerate(fields, 1):
                    raw = row_dict.get(fn, "")
                    try: val: Any = float(raw) if raw not in ("","None","nan",None) else raw
                    except (ValueError, TypeError): val = raw
                    ws.cell(row=excel_row, column=ci, value=val)
                excel_row += 1
            # average row
            avg: Dict[str, Any] = {}
            for col in SUMMARY_AVG_COLS:
                if col not in fields: continue
                nums = []
                for r in rows:
                    try:
                        v = float(r.get(col, ""))
                        if np.isfinite(v): nums.append(v)
                    except (ValueError, TypeError): pass
                avg[col] = float(np.mean(nums)) if nums else ""
            for ci, fn in enumerate(fields, 1):
                cell = ws.cell(row=excel_row, column=ci, value=avg.get(fn, ""))
                cell.font = Font(bold=True); cell.fill = afill
            excel_row += 2

        for col_cells in ws.columns:
            ml = max((len(str(c.value)) for c in col_cells if c.value is not None), default=8)
            ws.column_dimensions[col_cells[0].column_letter].width = min(ml+2, 40)

    wb.save(out_path); return out_path


# =============================================================================
# ANGULAR ERROR SUMMARY  (mean ± std per variant × distance)
# =============================================================================
def _parse_distance(scenario_folder):
    m = re.search(r"_(\d+m)", scenario_folder)
    return m.group(1) if m else None


def print_angular_error_summary(metrics_files_by_cond: Dict[str, List[str]]) -> str:
    """
    Collect angular_error_peak_deg from all metrics CSVs, group by
    (variant, distance), and return a printed table string.
    """
    # {variant_base: {distance: [values]}}
    stats: Dict[str, Dict[str, List[float]]] = {}

    for cond_name, csv_paths in sorted(metrics_files_by_cond.items()):
        # cond_name = "ief_gaze360_fold0" / "ief_mpiigaze_ensemble" etc.
        variant_base = cond_name.rsplit("_", 1)[0]   # "ief_gaze360" or "ief_mpiigaze"
        fold_id      = cond_name.rsplit("_", 1)[-1]  # "fold0" … "fold14" / "ensemble"

        for csv_path in csv_paths:
            # Extract distance from path (…/person/scenario/outputs_ief/…)
            parts = csv_path.replace("\\", "/").split("/")
            dist  = None
            for part in parts:
                d = _parse_distance(part)
                if d: dist = d; break
            if dist is None:
                continue

            try:
                with open(csv_path, newline="") as f:
                    rows = list(csv.DictReader(f))
            except Exception:
                continue

            for row in rows:
                val = row.get("angular_error_peak_deg", "")
                try:
                    v = float(val)
                    if np.isfinite(v):
                        stats.setdefault(variant_base, {}).setdefault(dist, []).append(v)
                except (ValueError, TypeError):
                    pass

    lines = ["\n" + "="*60,
             "  ANGULAR ERROR SUMMARY  (angular_error_peak_deg, degrees)",
             "  Source: peak of fused heatmap → 3-D world point (VGGT)",
             "          → angle to GT centroid in 3-D",
             "="*60]

    if not stats:
        lines.append("  (no data found yet — run not complete)")
    else:
        for variant_base in sorted(stats):
            lines.append(f"\n  Variant: {variant_base}")
            for dist in sorted(stats[variant_base], key=lambda x: int(x.replace("m",""))):
                vals = stats[variant_base][dist]
                n    = len(vals)
                mn   = float(np.mean(vals))
                sd   = float(np.std(vals, ddof=1)) if n > 1 else 0.0
                lines.append(f"    {dist}: mean={mn:6.2f}°  std={sd:5.2f}°  (n={n})")

    lines.append("="*60 + "\n")
    return "\n".join(lines)


def save_angular_error_table(metrics_files_by_cond, out_dir):
    """Save a per-fold breakdown Excel showing mean ± std per distance."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "angular_error_summary.xlsx")
    wb = openpyxl.Workbook(); wb.remove(wb.active)
    hfill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    afill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")

    # {variant_base: {distance: {cond_name: [values]}}}
    data: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    for cond_name, csv_paths in sorted(metrics_files_by_cond.items()):
        variant_base = cond_name.rsplit("_", 1)[0]
        for csv_path in csv_paths:
            parts = csv_path.replace("\\", "/").split("/")
            dist  = None
            for p in parts:
                d = _parse_distance(p)
                if d: dist = d; break
            if dist is None: continue
            try:
                with open(csv_path, newline="") as f:
                    rows = list(csv.DictReader(f))
            except Exception: continue
            for row in rows:
                try:
                    v = float(row.get("angular_error_peak_deg", ""))
                    if np.isfinite(v):
                        (data.setdefault(variant_base, {})
                             .setdefault(dist, {})
                             .setdefault(cond_name, [])
                             .append(v))
                except (ValueError, TypeError): pass

    for variant_base, dist_data in sorted(data.items()):
        ws = wb.create_sheet(title=variant_base[:31])
        cond_names_sorted = sorted(set(cn for dd in dist_data.values() for cn in dd))

        # Header
        headers = ["distance", "metric"] + cond_names_sorted + ["MEAN_across_folds", "STD_across_folds"]
        for ci, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=ci, value=h)
            cell.font = Font(bold=True); cell.fill = hfill

        r = 2
        for dist in sorted(dist_data, key=lambda x: int(x.replace("m",""))):
            cond_vals = dist_data[dist]
            # One row per fold / ensemble
            fold_means = []
            for ci_off, cname in enumerate(cond_names_sorted):
                vals = cond_vals.get(cname, [])
                if vals: fold_means.append(float(np.mean(vals)))

            for ci_off, cname in enumerate(cond_names_sorted):
                vals = cond_vals.get(cname, [])
                ws.cell(row=r, column=1, value=dist)
                ws.cell(row=r, column=2, value=cname)
                ws.cell(row=r, column=3+ci_off, value=round(float(np.mean(vals)),3) if vals else "")
            # summary row
            ws.cell(row=r+1, column=1, value=dist)
            ws.cell(row=r+1, column=2, value="MEAN±STD")
            if fold_means:
                mn = float(np.mean(fold_means)); sd = float(np.std(fold_means, ddof=1)) if len(fold_means)>1 else 0.0
                summary_str = f"{mn:.2f} ± {sd:.2f}"
            else:
                summary_str = ""
            ws.cell(row=r+1, column=3, value=summary_str)
            for ci_off, cname in enumerate(cond_names_sorted):
                vals = cond_vals.get(cname, [])
                ws.cell(row=r, column=3+ci_off, value=round(float(np.mean(vals)),3) if vals else "")
                ws.cell(row=r+1, column=3+ci_off, value="")

            cell = ws.cell(row=r+1, column=3, value=summary_str)
            cell.font = Font(bold=True); cell.fill = afill
            r += 3

        for col_cells in ws.columns:
            ml = max((len(str(c.value)) for c in col_cells if c.value is not None), default=8)
            ws.column_dimensions[col_cells[0].column_letter].width = min(ml+2, 40)

    wb.save(out_path); return out_path


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    print(f"device: {device_str}, dtype: {dtype}", flush=True)
    print(f"BASE_ROOT: {BASE_ROOT}", flush=True)

    if not os.path.isfile(CENTERBIAS_NPY):
        raise RuntimeError(f"centerbias file not found: {CENTERBIAS_NPY}\n"
                           "Place it there or set CENTERBIAS_NPY in paths_local.py")

    data_roots = discover_scenario_roots(BASE_ROOT)
    if not data_roots:
        raise RuntimeError(f"No REF_IMAGE_NAMES found under {BASE_ROOT}")
    if ONLY_SCENARIO_NAME:
        data_roots = [r for r in data_roots if os.path.basename(r) == ONLY_SCENARIO_NAME]
    print(f"Found {len(data_roots)} scenario root(s)", flush=True)

    # ---- Load models ----
    print("\n[LOAD] VGGT …", flush=True)
    # Keep VGGT weights in float32: autocast (see _run_shared_features) handles fp16. VGGT's heads disable
    # autocast internally, so fp16 weights crash with "expected scalar type Float but found Half".
    vggt_model = VGGT.from_pretrained(VGGT_SOURCE).to(device_str)
    vggt_model.eval()

    print("[LOAD] L2CS (face detector) …", flush=True)
    l2cs_pipeline = Pipeline(weights=L2CS_WEIGHTS, arch="ResNet50", device=torch_dev)

    print("[LOAD] DeepGaze IIE …", flush=True)
    dg_model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(device_str)
    dg_model.eval()
    if device_str == "cuda": dg_model.half()

    centerbias_template = np.load(CENTERBIAS_NPY)

    print("[LOAD] IEF models …", flush=True)
    ief_models: Dict[str, Tuple[List, List]] = {}
    for vname, vcfg in IEF_VARIANTS.items():
        models, cfgs = load_ief_variant(vname, vcfg, torch_dev)
        ief_models[vname] = (models, cfgs)

    # Build list of all condition names — only include folds that actually loaded
    all_cond_names: List[str] = []
    for vname, (models, _cfgs) in ief_models.items():
        if not ENSEMBLE_ONLY:
            for i, m in enumerate(models):
                if m is not None:
                    all_cond_names.append(f"{vname}_fold{i}")
        all_cond_names.append(f"{vname}_ensemble")

    # ---- Outer loop ----
    all_metrics_by_cond: Dict[str, List[str]] = {cn: [] for cn in all_cond_names}  # condition → [csv paths]

    for data_root in data_roots:
        output_root = os.path.join(data_root, "outputs_ief")
        gt_root     = os.path.join(data_root, "outputs_batch")
        os.makedirs(output_root, exist_ok=True)

        ref_paths = [os.path.join(data_root, n) for n in REF_IMAGE_NAMES]
        if any(not os.path.isfile(p) for p in ref_paths):
            print(f"[SKIP] missing refs in {data_root}"); continue

        scenario_dirs = list_scenario_dirs(data_root, output_root)
        if not scenario_dirs: continue
        if ONLY_SUBSCENARIO_NAME:
            scenario_dirs = [d for d in scenario_dirs if os.path.basename(d) == ONLY_SUBSCENARIO_NAME]

        print(f"\n{'='*70}\n[DATA ROOT] {data_root}", flush=True)

        for scen_dir in scenario_dirs:
            scenario_name    = os.path.basename(scen_dir)
            scenario_out_dir = os.path.join(output_root, scenario_name)
            os.makedirs(scenario_out_dir, exist_ok=True)

            primary_list = list_images_in_dir(scen_dir)
            if not primary_list: continue

            print(f"\n[SCENARIO] {scenario_name}  ({len(primary_list)} primaries)", flush=True)

            gt_for_scenario = {}
            if DO_EVAL:
                gt_for_scenario = ensure_ground_truth_for_scenario(
                    scenario_name=scenario_name, scenario_dir=scen_dir,
                    scenario_out_dir=scenario_out_dir, ref_paths=ref_paths,
                    deepgaze_view_idxs=DEEPGAZE_VIEW_IDXS,
                    primary_paths=primary_list, gt_root=gt_root,
                )

            # metrics accumulator per condition for this sub-scenario
            metrics_by_cond: Dict[str, List[Dict]] = {cn: [] for cn in all_cond_names}

            for i, primary_path in enumerate(primary_list):
                base    = os.path.splitext(os.path.basename(primary_path))[0]
                out_dir = os.path.join(scenario_out_dir, base)
                print(f"  [{i+1}/{len(primary_list)}] {os.path.basename(primary_path)}", flush=True)
                run_one_primary(
                    primary_path=primary_path, ref_paths=ref_paths,
                    out_dir_base=out_dir, vggt_model=vggt_model,
                    l2cs_pipeline=l2cs_pipeline, ief_models=ief_models,
                    dg_model=dg_model, centerbias_template=centerbias_template,
                    gt_for_scenario=gt_for_scenario,
                    metrics_by_condition=metrics_by_cond,
                )

            # Save one metrics.csv per condition per subscenario
            if DO_EVAL:
                for cname, rows in metrics_by_cond.items():
                    if not rows: continue
                    # Store under: outputs_ief/<condition>/<subscenario>/metrics.csv
                    cond_scen_dir = os.path.join(output_root, cname, scenario_name)
                    os.makedirs(cond_scen_dir, exist_ok=True)
                    csv_path = os.path.join(cond_scen_dir, "metrics.csv")
                    with open(csv_path, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                        w.writeheader(); w.writerows(rows)
                    all_metrics_by_cond[cname].append(csv_path)
                    print(f"  [SAVE] {csv_path}  ({len(rows)} rows)", flush=True)

            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- Summaries ----
    if DO_EVAL and SUMMARY_OUT_DIR:
        os.makedirs(SUMMARY_OUT_DIR, exist_ok=True)

        # Re-scan disk in case we resumed and some CSVs were already on disk
        for cn in all_cond_names:
            existing = sorted(glob(
                os.path.join(BASE_ROOT, "*", "*", "outputs_ief", cn, "*", "metrics.csv")
            ))
            # Merge without duplicates
            known = set(all_metrics_by_cond[cn])
            for p in existing:
                if p not in known:
                    all_metrics_by_cond[cn].append(p); known.add(p)

        # Print + save summary table
        summary_str = print_angular_error_summary(all_metrics_by_cond)
        print(summary_str)
        txt_path = os.path.join(SUMMARY_OUT_DIR, "angular_error_summary.txt")
        with open(txt_path, "w") as f: f.write(summary_str)
        print(f"[SAVE] {txt_path}", flush=True)

        ae_xlsx = save_angular_error_table(all_metrics_by_cond, SUMMARY_OUT_DIR)
        print(f"[SAVE] {ae_xlsx}", flush=True)

        # Per-person per-distance Excel (same layout as alpha=2)
        print("\n[SUMMARY] Per-person per-distance Excel …", flush=True)
        all_csv_files = sorted(glob(
            os.path.join(BASE_ROOT, "*", "*", "outputs_ief", "*", "*", "metrics.csv")
        ))
        # {(person_label, distance): {condition_name: {subscenario: [rows]}}}
        excel_data: Dict[Tuple[str,str], Dict[str, Dict[str, List[Dict]]]] = {}
        for mf in all_csv_files:
            parts = os.path.relpath(mf, BASE_ROOT).replace("\\","/").split("/")
            # person / scenario / outputs_ief / cond_name / subscenario / metrics.csv
            if len(parts) != 6: continue
            pfolder, scen_folder, _, cname, ssname, _ = parts
            plabel = PERSON_LABEL_MAP.get(pfolder, pfolder.lower())
            dist   = _parse_distance(scen_folder)
            if not dist: continue
            try:
                with open(mf, newline="") as f: rows = list(csv.DictReader(f))
            except Exception: continue
            if not rows: continue
            key = (plabel, dist)
            excel_data.setdefault(key, {})
            excel_data[key].setdefault(cname, {})
            excel_data[key][cname].setdefault(ssname, [])
            excel_data[key][cname][ssname].extend(rows)

        for (plabel, dist), cond_data in sorted(excel_data.items()):
            out_path = write_person_distance_excel(
                person_label=plabel, distance=dist,
                data_by_cond=cond_data, out_dir=SUMMARY_OUT_DIR,
            )
            print(f"  [EXCEL] {out_path}", flush=True)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
