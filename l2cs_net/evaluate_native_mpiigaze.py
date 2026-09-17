"""
Evaluate the native MPIIGaze-trained L2CS checkpoints (28 bins, 3 deg/bin, +-42 deg)
using the same 2D joint coverage metrics as the IEF evaluation.

Runs all 15 folds and reports mean +- std.

Usage:
  python evaluate_native_mpiigaze.py ^
    --model_dir "models/MPIIGaze-20260529T070835Z-3-001/MPIIGaze" ^
    --image_dir "datasets/MPIIFaceGaze/Image" ^
    --label_dir "datasets/MPIIFaceGaze/Label" ^
    --output    "output/eval_native"
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from l2cs import L2CS, select_device, Mpiigaze
from evaluate_refinement import (
    soft_argmax_np, angular_error_deg,
    angle_to_bin, joint_conf_needed,
    reliability_diagram_joint,
)

# Native MPIIGaze bin config
BIN_WIDTH = 3.0
ANGLE_OFF = 42.0
NUM_BINS  = 28


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    pp_all, yp_all, gtp_all, gty_all = [], [], [], []
    for images, _labels, cont_labels, _name in loader:
        images = images.to(device)
        pl, yl = model(images)
        pp_all.append(F.softmax(pl, dim=1).cpu().numpy())
        yp_all.append(F.softmax(yl, dim=1).cpu().numpy())
        gtp_all.append(cont_labels[:, 0].numpy())
        gty_all.append(cont_labels[:, 1].numpy())
    return (np.concatenate(pp_all).astype(np.float32),
            np.concatenate(yp_all).astype(np.float32),
            np.concatenate(gtp_all),
            np.concatenate(gty_all))


def eval_fold(pp, yp, gt_p, gt_y):
    pred_p = soft_argmax_np(pp, BIN_WIDTH, ANGLE_OFF)
    pred_y = soft_argmax_np(yp, BIN_WIDTH, ANGLE_OFF)
    ang_err = angular_error_deg(pred_p, pred_y, gt_p, gt_y).mean()

    gt_pb = angle_to_bin(gt_p, BIN_WIDTH, ANGLE_OFF, NUM_BINS, clip=True)
    gt_yb = angle_to_bin(gt_y, BIN_WIDTH, ANGLE_OFF, NUM_BINS, clip=True)

    eps = 1e-9
    nll = (-np.log(pp[np.arange(len(pp)), gt_pb] + eps)
           - np.log(yp[np.arange(len(yp)), gt_yb] + eps)).mean()

    conf = joint_conf_needed(pp, yp, gt_pb, gt_yb)
    cov95 = (conf <= 0.95).mean()
    cov90 = (conf <= 0.90).mean()
    cov50 = (conf <= 0.50).mean()
    _, _, _, ece = reliability_diagram_joint(conf)

    return dict(ang_err=ang_err, nll=nll,
                cov95=cov95, cov90=cov90, cov50=cov50, ece=ece)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",  required=True)
    p.add_argument("--image_dir",  required=True)
    p.add_argument("--label_dir",  required=True)
    p.add_argument("--output",     default="output/eval_native")
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--gpu",        default="0", type=str)
    return p.parse_args()


def main():
    args   = parse_args()
    device = select_device(args.gpu)

    transform = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    label_paths = sorted([
        os.path.join(args.label_dir, f)
        for f in os.listdir(args.label_dir) if f.endswith(".label")
    ])

    os.makedirs(args.output, exist_ok=True)
    results = []

    print(f"{'Fold':<6} {'Ang.Err':>9} {'NLL':>8} {'Cov95':>8} {'Cov90':>8} {'Cov50':>8} {'ECE':>8}")
    print("-" * 60)

    for fold in range(15):
        ckpt_path = os.path.join(args.model_dir, f"fold{fold}.pkl")

        # Load — DataParallel-wrapped state dict
        base  = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins=28)
        model = nn.DataParallel(base, device_ids=[0])
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model.to(device).eval()

        val_set    = Mpiigaze(label_paths, args.image_dir, transform, False, 42, fold)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)

        pp, yp, gt_p, gt_y = collect(model, val_loader, device)
        m = eval_fold(pp, yp, gt_p, gt_y)
        results.append(m)

        print(f"{fold:<6} {m['ang_err']:>9.3f} {m['nll']:>8.3f} "
              f"{m['cov95']*100:>7.1f}% {m['cov90']*100:>7.1f}% "
              f"{m['cov50']*100:>7.1f}% {m['ece']:>8.4f}")

    # Summary
    keys = ["ang_err", "nll", "cov95", "cov90", "cov50", "ece"]
    print("-" * 60)
    means = {k: np.mean([r[k] for r in results]) for k in keys}
    stds  = {k: np.std( [r[k] for r in results]) for k in keys}
    print(f"{'Mean':<6} {means['ang_err']:>9.3f} {means['nll']:>8.3f} "
          f"{means['cov95']*100:>7.1f}% {means['cov90']*100:>7.1f}% "
          f"{means['cov50']*100:>7.1f}% {means['ece']:>8.4f}")
    print(f"{'Std':<6} {stds['ang_err']:>9.3f} {stds['nll']:>8.3f} "
          f"{stds['cov95']*100:>7.1f}% {stds['cov90']*100:>7.1f}% "
          f"{stds['cov50']*100:>7.1f}% {stds['ece']:>8.4f}")

    # Save
    out_path = os.path.join(args.output, "eval_all_folds.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Native MPIIGaze L2CS — all 15 folds\n")
        f.write(f"Bin config: {NUM_BINS} bins, {BIN_WIDTH} deg/bin, +-{ANGLE_OFF} deg\n")
        f.write(f"Joint coverage: {NUM_BINS}x{NUM_BINS} = {NUM_BINS**2} cells\n\n")
        f.write(f"{'Fold':<6} {'Ang.Err':>9} {'NLL':>8} {'Cov95':>8} {'Cov90':>8} {'Cov50':>8} {'ECE':>8}\n")
        for fold, m in enumerate(results):
            f.write(f"{fold:<6} {m['ang_err']:>9.3f} {m['nll']:>8.3f} "
                    f"{m['cov95']*100:>7.1f}% {m['cov90']*100:>7.1f}% "
                    f"{m['cov50']*100:>7.1f}% {m['ece']:>8.4f}\n")
        f.write(f"\nMean:  ang_err={means['ang_err']:.3f}  nll={means['nll']:.3f}  "
                f"cov95={means['cov95']*100:.1f}%  cov90={means['cov90']*100:.1f}%  "
                f"cov50={means['cov50']*100:.1f}%  ece={means['ece']:.4f}\n")
        f.write(f"Std:   ang_err={stds['ang_err']:.3f}   nll={stds['nll']:.3f}   "
                f"cov95={stds['cov95']*100:.1f}%  cov90={stds['cov90']*100:.1f}%  "
                f"cov50={stds['cov50']*100:.1f}%  ece={stds['ece']:.4f}\n")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
