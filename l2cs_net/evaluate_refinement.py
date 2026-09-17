"""
Evaluation harness for the IEF gaze refinement head.

Produces per-step angular error, 2D joint coverage (raw + clipped), NLL,
ECE, a reliability diagram, and the headline per-step convergence figure.

Coverage is 2D joint: pitch and yaw distributions are combined as an
independent joint over the 90x90 bin grid.  The HPD credible region is
then the smallest set of (pitch_bin, yaw_bin) cells whose total probability
mass reaches the target level.  This is more correct than averaging two
independent 1D intervals, which overstates coverage by ~5 pp at the 95%
level (0.95 * 0.95 = 0.90, not 0.95).

Coverage is reported two ways:
  raw     - GT angle used as-is
  clipped - GT angle clipped to +-42 deg before bin lookup (removes the
            Gaze360/MPIIGaze domain-range artefact; all MPIIGaze GT angles
            are within +-180 deg so raw == clipped here, but kept for clarity)

Usage:
  python evaluate_refinement.py ^
    --weights     "path/to/L2CSNet_gaze360.pkl" ^
    --checkpoint  "output/refinement/best_fold0.pt" ^
    --image_dir   "datasets/MPIIFaceGaze/Image" ^
    --label_dir   "datasets/MPIIFaceGaze/Label" ^
    --output      "output/refinement" ^
    --fold 0
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from l2cs import L2CS, select_device, Mpiigaze
from l2cs.wrapper import L2CSWrapper
from l2cs.refinement_head import RefinementConfig, IEFGazeModel


# ── Gaze math ─────────────────────────────────────────────────────────────────

def soft_argmax_np(probs, bin_width, angle_offset):
    idx  = np.arange(probs.shape[1])
    bins = idx * bin_width - angle_offset
    return (probs * bins).sum(axis=1)


def angular_error_deg(pred_pitch_deg, pred_yaw_deg, gt_pitch_deg, gt_yaw_deg):
    """Mean 3D arc-distance error in degrees. All inputs: (N,) numpy."""
    def to3d(p, y):
        p, y = np.deg2rad(p), np.deg2rad(y)
        return np.stack([
            -np.cos(p) * np.sin(y),
            -np.sin(p),
            -np.cos(p) * np.cos(y),
        ], axis=1)
    cos_sim = np.clip((to3d(pred_pitch_deg, pred_yaw_deg) *
                       to3d(gt_pitch_deg,   gt_yaw_deg)).sum(axis=1), -1.0, 0.9999999)
    return np.rad2deg(np.arccos(cos_sim))


def angle_to_bin(angle_deg, bin_width, angle_offset, num_bins, clip=False):
    if clip:
        angle_deg = np.clip(angle_deg, -angle_offset, angle_offset - bin_width)
    idx = np.floor((angle_deg + angle_offset) / bin_width).astype(int)
    return np.clip(idx, 0, num_bins - 1)


# ── 2D joint coverage ─────────────────────────────────────────────────────────

def joint_conf_needed(pitch_probs, yaw_probs, gt_pitch_bin, gt_yaw_bin):
    """Compute the HPD mass needed to just cover the true (pitch, yaw) cell.

    Under independence:  P_joint(i,j) = P_pitch(i) * P_yaw(j)

    conf_needed[n] = sum of all P_joint cells with probability >= P_joint(true cell)
                   = the smallest credible level whose HPD set contains the truth.

    pitch_probs:  (N, B)  float32
    yaw_probs:    (N, B)  float32
    gt_pitch_bin: (N,)    int
    gt_yaw_bin:   (N,)    int
    Returns:      (N,)    float32 in [0, 1]
    """
    N, B = pitch_probs.shape
    # Joint distribution flattened: (N, B*B)
    joint = (pitch_probs[:, :, None] * yaw_probs[:, None, :]).reshape(N, B * B)  # (N, B*B)

    # Probability of the true cell for each sample
    true_cell = gt_pitch_bin * B + gt_yaw_bin          # (N,)
    true_p    = joint[np.arange(N), true_cell]         # (N,)

    # HPD mass needed = sum of all cells with P >= P(true cell)
    # (includes the true cell itself and any ties)
    conf = (joint * (joint >= true_p[:, None])).sum(axis=1)  # (N,)
    return conf.astype(np.float32)


def joint_hpd_coverage(pitch_probs, yaw_probs, gt_pitch_bin, gt_yaw_bin, level=0.95):
    """Fraction of samples where the joint HPD set at `level` contains the true cell."""
    conf = joint_conf_needed(pitch_probs, yaw_probs, gt_pitch_bin, gt_yaw_bin)
    return (conf <= level).mean()


# ── Reliability diagram + ECE (joint) ─────────────────────────────────────────

def reliability_diagram_joint(conf_needed, num_buckets=10):
    """
    conf_needed: (N,) from joint_conf_needed — confidence level required to cover truth.

    A well-calibrated model: fraction of samples covered at level alpha ~= alpha.
    i.e. coverage(alpha) = mean(conf_needed <= alpha) should equal alpha.

    We bucket samples by conf_needed, then for each bucket compute:
      - x = bucket centre (claimed confidence level)
      - y = fraction of samples in that bucket that are actually covered at x

    Returns conf_bins, accuracy, counts, ece.
    """
    edges     = np.linspace(0, 1, num_buckets + 1)
    conf_bins = 0.5 * (edges[:-1] + edges[1:])
    accuracy  = np.zeros(num_buckets)
    counts    = np.zeros(num_buckets, dtype=int)

    for k in range(num_buckets):
        lo, hi = edges[k], edges[k + 1]
        mask = (conf_needed >= lo) & (conf_needed < hi)
        counts[k] = mask.sum()
        if counts[k] > 0:
            # At the bucket centre confidence level, what fraction is covered?
            accuracy[k] = (conf_needed[mask] <= conf_bins[k]).mean()

    ece = np.sum(np.abs(accuracy - conf_bins) * counts) / max(len(conf_needed), 1)
    return conf_bins, accuracy, counts, ece


# ── Collect predictions ───────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, loader, cfg, device):
    model.eval()
    n_steps = cfg.num_steps + 1

    all_pitch = [[] for _ in range(n_steps)]
    all_yaw   = [[] for _ in range(n_steps)]
    all_gtp, all_gty = [], []

    for images, _labels, cont_labels, _name in loader:
        images = images.to(device)
        steps  = model(images)
        all_gtp.append(cont_labels[:, 0].numpy())
        all_gty.append(cont_labels[:, 1].numpy())
        for t, (pl, yl) in enumerate(steps):
            all_pitch[t].append(F.softmax(pl, dim=1).cpu().numpy())
            all_yaw[t].append(F.softmax(yl, dim=1).cpu().numpy())

    return {
        "pitch_probs": [np.concatenate(all_pitch[t]) for t in range(n_steps)],
        "yaw_probs":   [np.concatenate(all_yaw[t])   for t in range(n_steps)],
        "gt_pitch":    np.concatenate(all_gtp),
        "gt_yaw":      np.concatenate(all_gty),
    }


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(preds, cfg):
    BW, OFF = cfg.bin_width_deg, cfg.angle_offset_deg
    n_steps = cfg.num_steps + 1
    gt_p, gt_y = preds["gt_pitch"], preds["gt_yaw"]

    metrics = {k: [] for k in [
        "angular_error",
        "nll",               # joint NLL = NLL_pitch + NLL_yaw (under independence)
        "cov95_raw",  "cov90_raw",  "cov50_raw",
        "cov95_clip", "cov90_clip", "cov50_clip",
        "ece_raw", "ece_clip",
    ]}
    rel_data_raw  = []
    rel_data_clip = []

    for t in range(n_steps):
        pp = preds["pitch_probs"][t].astype(np.float32)   # (N, 90)
        yp = preds["yaw_probs"][t].astype(np.float32)

        # Angular error (3D arc distance)
        pred_p = soft_argmax_np(pp, BW, OFF)
        pred_y = soft_argmax_np(yp, BW, OFF)
        metrics["angular_error"].append(angular_error_deg(pred_p, pred_y, gt_p, gt_y).mean())

        # Joint NLL (sum of per-axis NLLs — exact under independence)
        eps = 1e-9
        gt_pb_raw = angle_to_bin(gt_p, BW, OFF, pp.shape[1], clip=False)
        gt_yb_raw = angle_to_bin(gt_y, BW, OFF, yp.shape[1], clip=False)
        nll = (-np.log(pp[np.arange(len(pp)), gt_pb_raw] + eps)
               - np.log(yp[np.arange(len(yp)), gt_yb_raw] + eps)).mean()
        metrics["nll"].append(nll)

        # 2D joint coverage — raw GT bins
        gt_pb_clip = angle_to_bin(gt_p, BW, OFF, pp.shape[1], clip=True)
        gt_yb_clip = angle_to_bin(gt_y, BW, OFF, yp.shape[1], clip=True)

        conf_raw  = joint_conf_needed(pp, yp, gt_pb_raw,  gt_yb_raw)
        conf_clip = joint_conf_needed(pp, yp, gt_pb_clip, gt_yb_clip)

        for level, key in [(0.95, "95"), (0.90, "90"), (0.50, "50")]:
            metrics[f"cov{key}_raw"].append( (conf_raw  <= level).mean())
            metrics[f"cov{key}_clip"].append((conf_clip <= level).mean())

        # ECE (joint)
        cb_r, acc_r, cnt_r, ece_r = reliability_diagram_joint(conf_raw)
        cb_c, acc_c, cnt_c, ece_c = reliability_diagram_joint(conf_clip)
        metrics["ece_raw"].append(ece_r)
        metrics["ece_clip"].append(ece_c)
        rel_data_raw.append((cb_r, acc_r, cnt_r))
        rel_data_clip.append((cb_c, acc_c, cnt_c))

    return metrics, rel_data_raw, rel_data_clip


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_perstep(metrics, cfg, out_path):
    steps = list(range(cfg.num_steps + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(steps, metrics["angular_error"], "o-", color="steelblue", label="IEF refined")
    ax1.axhline(metrics["angular_error"][0], color="grey", linestyle="--",
                label=f"Backbone baseline ({metrics['angular_error'][0]:.2f} deg)")
    ax1.set_xlabel("Refinement step")
    ax1.set_ylabel("Mean angular error (deg)")
    ax1.set_title("Angular error vs step")
    ax1.legend()
    ax1.set_xticks(steps)

    ax2.plot(steps, [v * 100 for v in metrics["cov95_raw"]],
             "s--", color="tomato",   label="95% joint coverage (raw)")
    ax2.plot(steps, [v * 100 for v in metrics["cov95_clip"]],
             "o-",  color="seagreen", label="95% joint coverage (clipped)")
    ax2.axhline(95, color="black", linestyle=":", linewidth=1, label="Ideal (95%)")
    ax2.set_xlabel("Refinement step")
    ax2.set_ylabel("2D joint coverage (%)")
    ax2.set_title("95% joint gaze-cone coverage vs step\n"
                  "(raw = full domain gap; clipped = within MPIIGaze range)")
    ax2.legend()
    ax2.set_xticks(steps)
    ax2.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved per-step figure  -> {out_path}")


def plot_reliability(rel_data, cfg, out_path, title_suffix=""):
    n_steps = cfg.num_steps + 1
    fig, axes = plt.subplots(1, n_steps, figsize=(4 * n_steps, 4), sharey=True)
    if n_steps == 1:
        axes = [axes]

    for t, (cb, acc, _cnt) in enumerate(rel_data):
        ax = axes[t]
        ax.bar(cb, acc, width=0.08, alpha=0.6, color="steelblue")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
        ax.set_title("Backbone (t=0)" if t == 0 else f"Step {t}")
        ax.set_xlabel("Claimed confidence level")
        if t == 0:
            ax.set_ylabel("Actual coverage fraction")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle(f"2D joint reliability diagram{title_suffix}", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved reliability diagram -> {out_path}")


def write_summary(metrics, cfg, out_path):
    lines = [
        "IEF Gaze Refinement — Evaluation Summary (2D joint coverage)",
        f"num_steps={cfg.num_steps}  num_bins={cfg.num_bins}  "
        f"bin_width={cfg.bin_width_deg}deg  max_delta={cfg.max_delta_deg}deg",
        "",
        "2D joint coverage: P_joint(i,j) = P_pitch(i) * P_yaw(j).",
        "HPD set = smallest set of (pitch,yaw) cells summing to >= level.",
        "True joint 95% coverage should equal 95%; per-axis average would",
        "overstate by ~5pp (0.95*0.95=0.90 for independent axes).",
        "",
        f"{'Step':<6} {'Ang.Err':>9} {'Joint NLL':>10} "
        f"{'Cov95_raw':>11} {'Cov95_clip':>12} "
        f"{'Cov90_clip':>12} {'Cov50_clip':>12} "
        f"{'ECE_raw':>9} {'ECE_clip':>9}",
        "-" * 100,
    ]
    for t in range(cfg.num_steps + 1):
        tag = "  <- baseline" if t == 0 else ""
        lines.append(
            f"{t:<6} "
            f"{metrics['angular_error'][t]:>8.3f}  "
            f"{metrics['nll'][t]:>9.3f}  "
            f"{metrics['cov95_raw'][t]*100:>10.1f}%  "
            f"{metrics['cov95_clip'][t]*100:>11.1f}%  "
            f"{metrics['cov90_clip'][t]*100:>11.1f}%  "
            f"{metrics['cov50_clip'][t]*100:>11.1f}%  "
            f"{metrics['ece_raw'][t]:>8.4f}  "
            f"{metrics['ece_clip'][t]:>8.4f}"
            f"{tag}"
        )
    lines += [
        "",
        "raw:     GT angle used as-is (includes Gaze360/MPIIGaze range artefact).",
        "clipped: GT clipped to +-angle_offset — isolates calibration from range mismatch.",
    ]
    text = "\n".join(lines)
    print(text)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nSaved summary -> {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate IEF gaze refinement head")
    p.add_argument("--weights",    required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image_dir",  required=True)
    p.add_argument("--label_dir",  required=True)
    p.add_argument("--output",     default="output/refinement")
    p.add_argument("--fold",       default=0,  type=int)
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--gpu",        default="0", type=str)
    return p.parse_args()


def main():
    args   = parse_args()
    device = select_device(args.gpu)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg: RefinementConfig = ckpt["cfg"]
    print(f"Checkpoint: epoch={ckpt['epoch']}  val_err={ckpt['val_angular_error']:.3f}deg")
    print(f"Config: {cfg}")

    base = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins=90)
    base.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=False))
    wrapper = L2CSWrapper(base)
    model   = IEFGazeModel(wrapper, cfg).to(device)
    model.head.load_state_dict(ckpt["head_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    label_paths = sorted([
        os.path.join(args.label_dir, f)
        for f in os.listdir(args.label_dir) if f.endswith(".label")
    ])
    val_set    = Mpiigaze(label_paths, args.image_dir, transform, False, 42, args.fold)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    print(f"Val set: {len(val_set)} samples (fold={args.fold})")

    print("Running inference...")
    preds = collect_predictions(model, val_loader, cfg, device)

    print("Computing metrics...")
    metrics, rel_raw, rel_clip = evaluate(preds, cfg)

    os.makedirs(args.output, exist_ok=True)
    write_summary(metrics, cfg,
                  os.path.join(args.output, f"eval_fold{args.fold}.txt"))
    plot_perstep(metrics, cfg,
                 os.path.join(args.output, f"perstep_fold{args.fold}.png"))
    plot_reliability(rel_clip, cfg,
                     os.path.join(args.output, f"reliability_fold{args.fold}.png"),
                     title_suffix=" (clipped GT)")


if __name__ == "__main__":
    main()
