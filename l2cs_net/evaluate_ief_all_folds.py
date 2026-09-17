"""
Evaluate IEF MLP head across all 15 folds and save per-fold results.

Expects checkpoints at: output/refinement/best_foldN.pt

Usage:
  python evaluate_ief_all_folds.py ^
    --weights    "path/to/L2CSNet_gaze360.pkl" ^
    --ckpt_dir   "output/refinement" ^
    --image_dir  "datasets/MPIIFaceGaze/Image" ^
    --label_dir  "datasets/MPIIFaceGaze/Label" ^
    --output     "output/eval_ief"
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from l2cs import L2CS, select_device, Mpiigaze
from l2cs.wrapper import L2CSWrapper
from l2cs.refinement_head import IEFGazeModel
from evaluate_refinement import (
    soft_argmax_np, angular_error_deg, angle_to_bin,
    joint_conf_needed, reliability_diagram_joint,
)

BIN_WIDTH = 4.0
ANGLE_OFF = 180.0
NUM_BINS  = 90


@torch.no_grad()
def collect(model, loader, cfg, device):
    model.eval()
    n_steps = cfg.num_steps + 1
    pp_all  = [[] for _ in range(n_steps)]
    yp_all  = [[] for _ in range(n_steps)]
    gtp, gty = [], []

    for images, _labels, cont_labels, _name in loader:
        images = images.to(device)
        steps  = model(images)
        gtp.append(cont_labels[:, 0].numpy())
        gty.append(cont_labels[:, 1].numpy())
        for t, (pl, yl) in enumerate(steps):
            pp_all[t].append(F.softmax(pl, dim=1).cpu().numpy())
            yp_all[t].append(F.softmax(yl, dim=1).cpu().numpy())

    return ([np.concatenate(pp_all[t]).astype(np.float32) for t in range(n_steps)],
            [np.concatenate(yp_all[t]).astype(np.float32) for t in range(n_steps)],
            np.concatenate(gtp), np.concatenate(gty))


def eval_step(pp, yp, gt_p, gt_y):
    pred_p  = soft_argmax_np(pp, BIN_WIDTH, ANGLE_OFF)
    pred_y  = soft_argmax_np(yp, BIN_WIDTH, ANGLE_OFF)
    ang_err = angular_error_deg(pred_p, pred_y, gt_p, gt_y).mean()

    gt_pb = angle_to_bin(gt_p, BIN_WIDTH, ANGLE_OFF, NUM_BINS, clip=False)
    gt_yb = angle_to_bin(gt_y, BIN_WIDTH, ANGLE_OFF, NUM_BINS, clip=False)

    eps = 1e-9
    nll = (-np.log(pp[np.arange(len(pp)), gt_pb] + eps)
           - np.log(yp[np.arange(len(yp)), gt_yb] + eps)).mean()

    conf = joint_conf_needed(pp, yp, gt_pb, gt_yb)
    _, _, _, ece = reliability_diagram_joint(conf)

    return dict(ang_err=float(ang_err), nll=float(nll),
                cov95=float((conf <= 0.95).mean()),
                cov90=float((conf <= 0.90).mean()),
                cov50=float((conf <= 0.50).mean()),
                ece=float(ece))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",   required=True)
    p.add_argument("--ckpt_dir",  default="output/refinement")
    p.add_argument("--image_dir", required=True)
    p.add_argument("--label_dir", required=True)
    p.add_argument("--output",    default="output/eval_ief")
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--gpu",        default="0", type=str)
    return p.parse_args()


def main():
    args   = parse_args()
    device = select_device(args.gpu)
    os.makedirs(args.output, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    label_paths = sorted([
        os.path.join(args.label_dir, f)
        for f in os.listdir(args.label_dir) if f.endswith(".label")
    ])

    # Header
    print(f"{'Fold':<6} {'Step':<6} {'Ang.Err':>9} {'NLL':>8} "
          f"{'Cov95':>8} {'Cov90':>8} {'Cov50':>8} {'ECE':>8}")
    print("-" * 70)

    all_final = []   # final-step metrics per fold
    all_data  = {}   # full per-fold per-step data for JSON

    missing = []
    for fold in range(15):
        ckpt_path = os.path.join(args.ckpt_dir, f"best_fold{fold}.pt")
        if not os.path.exists(ckpt_path):
            print(f"fold {fold}: checkpoint not found — skipping")
            missing.append(fold)
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg  = ckpt["cfg"]

        base    = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins=90)
        state   = torch.load(args.weights, map_location="cpu", weights_only=False)
        base.load_state_dict(state)
        wrapper = L2CSWrapper(base)
        model   = IEFGazeModel(wrapper, cfg).to(device)
        model.head.load_state_dict(ckpt["head_state_dict"])
        model.eval()

        val_set    = Mpiigaze(label_paths, args.image_dir, transform, False, 42, fold)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)

        pp_steps, yp_steps, gt_p, gt_y = collect(model, val_loader, cfg, device)

        fold_results = []
        for t in range(cfg.num_steps + 1):
            m = eval_step(pp_steps[t], yp_steps[t], gt_p, gt_y)
            fold_results.append(m)
            tag = " <-baseline" if t == 0 else ""
            print(f"{fold:<6} {t:<6} {m['ang_err']:>9.3f} {m['nll']:>8.3f} "
                  f"{m['cov95']*100:>7.1f}% {m['cov90']*100:>7.1f}% "
                  f"{m['cov50']*100:>7.1f}% {m['ece']:>8.4f}{tag}")

        all_data[fold]  = fold_results
        all_final.append(fold_results[-1])   # final refinement step
        print()

    if not all_final:
        print("No folds evaluated.")
        return

    # Summary over evaluated folds (final step only)
    keys = ["ang_err", "nll", "cov95", "cov90", "cov50", "ece"]
    means = {k: np.mean([m[k] for m in all_final]) for k in keys}
    stds  = {k: np.std( [m[k] for m in all_final]) for k in keys}

    print("-" * 70)
    print(f"Final step — mean across {len(all_final)} folds:")
    print(f"  Angular error : {means['ang_err']:.3f} +- {stds['ang_err']:.3f} deg")
    print(f"  Joint NLL     : {means['nll']:.3f} +- {stds['nll']:.3f}")
    print(f"  95% coverage  : {means['cov95']*100:.1f}% +- {stds['cov95']*100:.1f}%")
    print(f"  90% coverage  : {means['cov90']*100:.1f}% +- {stds['cov90']*100:.1f}%")
    print(f"  50% coverage  : {means['cov50']*100:.1f}% +- {stds['cov50']*100:.1f}%")
    print(f"  ECE           : {means['ece']:.4f} +- {stds['ece']:.4f}")

    if missing:
        print(f"\nNote: folds {missing} were skipped (checkpoints not yet available).")

    # Save JSON (per-fold, per-step — everything)
    json_path = os.path.join(args.output, "ief_all_folds.json")
    with open(json_path, "w") as f:
        json.dump({str(k): v for k, v in all_data.items()}, f, indent=2)

    # Save plain text summary
    txt_path = os.path.join(args.output, "ief_all_folds.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("IEF MLP head — all folds (final refinement step)\n")
        f.write(f"Bin config: {NUM_BINS} bins, {BIN_WIDTH} deg/bin, +-{ANGLE_OFF} deg\n\n")
        f.write(f"{'Fold':<6} {'Ang.Err':>9} {'NLL':>8} {'Cov95':>8} "
                f"{'Cov90':>8} {'Cov50':>8} {'ECE':>8}\n")
        for fold, m in zip(sorted(all_data.keys()), all_final):
            f.write(f"{fold:<6} {m['ang_err']:>9.3f} {m['nll']:>8.3f} "
                    f"{m['cov95']*100:>7.1f}% {m['cov90']*100:>7.1f}% "
                    f"{m['cov50']*100:>7.1f}% {m['ece']:>8.4f}\n")
        f.write(f"\nMean:  ang_err={means['ang_err']:.3f}  nll={means['nll']:.3f}  "
                f"cov95={means['cov95']*100:.1f}%  cov50={means['cov50']*100:.1f}%  "
                f"ece={means['ece']:.4f}\n")
        f.write(f"Std:   ang_err={stds['ang_err']:.3f}   nll={stds['nll']:.3f}   "
                f"cov95={stds['cov95']*100:.1f}%  cov50={stds['cov50']*100:.1f}%  "
                f"ece={stds['ece']:.4f}\n")

    print(f"\nSaved -> {txt_path}")
    print(f"Saved -> {json_path}")


if __name__ == "__main__":
    main()
