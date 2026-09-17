"""
Evaluate the fine-tuned L2CS baseline with the same 2D joint coverage metrics
used for the IEF model, so results are directly comparable.

Usage:
  python evaluate_finetune_l2cs.py ^
    --checkpoint  "output/finetune_l2cs/best_fold0.pt" ^
    --image_dir   "datasets/MPIIFaceGaze/Image" ^
    --label_dir   "datasets/MPIIFaceGaze/Label" ^
    --output      "output/finetune_l2cs" ^
    --fold 0
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from l2cs import L2CS, select_device
from train_finetune_l2cs import MpiigazeGaze360Bins

# Reuse the joint coverage helpers from evaluate_refinement
from evaluate_refinement import (
    soft_argmax_np, angular_error_deg,
    angle_to_bin, joint_conf_needed, joint_hpd_coverage,
    reliability_diagram_joint,
)

BIN_WIDTH = 4.0
ANGLE_OFF = 180.0
NUM_BINS  = 90


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    pitch_probs_all, yaw_probs_all = [], []
    gt_p_all, gt_y_all = [], []

    for images, _bins, cont_labels in loader:
        images = images.to(device)
        pl, yl = model(images)
        pitch_probs_all.append(F.softmax(pl, dim=1).cpu().numpy())
        yaw_probs_all.append(F.softmax(yl, dim=1).cpu().numpy())
        gt_p_all.append(cont_labels[:, 0].numpy())
        gt_y_all.append(cont_labels[:, 1].numpy())

    return (np.concatenate(pitch_probs_all).astype(np.float32),
            np.concatenate(yaw_probs_all).astype(np.float32),
            np.concatenate(gt_p_all),
            np.concatenate(gt_y_all))


def evaluate_and_print(pp, yp, gt_p, gt_y, label="model"):
    pred_p = soft_argmax_np(pp, BIN_WIDTH, ANGLE_OFF)
    pred_y = soft_argmax_np(yp, BIN_WIDTH, ANGLE_OFF)
    ang_err = angular_error_deg(pred_p, pred_y, gt_p, gt_y).mean()

    gt_pb = angle_to_bin(gt_p, BIN_WIDTH, ANGLE_OFF, NUM_BINS, clip=False)
    gt_yb = angle_to_bin(gt_y, BIN_WIDTH, ANGLE_OFF, NUM_BINS, clip=False)

    eps = 1e-9
    nll = (-np.log(pp[np.arange(len(pp)), gt_pb] + eps)
           - np.log(yp[np.arange(len(yp)), gt_yb] + eps)).mean()

    conf = joint_conf_needed(pp, yp, gt_pb, gt_yb)
    cov95 = (conf <= 0.95).mean()
    cov90 = (conf <= 0.90).mean()
    cov50 = (conf <= 0.50).mean()
    _, _, _, ece = reliability_diagram_joint(conf)

    print(f"\n=== {label} ===")
    print(f"  Angular error : {ang_err:.3f} deg")
    print(f"  Joint NLL     : {nll:.3f}")
    print(f"  95% coverage  : {cov95*100:.1f}%  (ideal: 95%)")
    print(f"  90% coverage  : {cov90*100:.1f}%  (ideal: 90%)")
    print(f"  50% coverage  : {cov50*100:.1f}%  (ideal: 50%)")
    print(f"  ECE           : {ece:.4f}")

    return dict(ang_err=ang_err, nll=nll, cov95=cov95, cov90=cov90,
                cov50=cov50, ece=ece, conf=conf)


def plot_reliability(conf, out_path, title):
    cb, acc, _, _ = reliability_diagram_joint(conf)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(cb, acc, width=0.08, alpha=0.6, color="steelblue", label="Model")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
    ax.set_xlabel("Claimed confidence level")
    ax.set_ylabel("Actual coverage fraction")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image_dir",  required=True)
    p.add_argument("--label_dir",  required=True)
    p.add_argument("--output",     default="output/finetune_l2cs")
    p.add_argument("--fold",       default=0,  type=int)
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--gpu",        default="0", type=str)
    return p.parse_args()


def main():
    args   = parse_args()
    device = select_device(args.gpu)

    ckpt  = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins=90)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}  val_err={ckpt['val_angular_error']:.3f} deg")

    transform = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    label_paths = sorted([
        os.path.join(args.label_dir, f)
        for f in os.listdir(args.label_dir) if f.endswith(".label")
    ])
    val_set    = MpiigazeGaze360Bins(label_paths, args.image_dir, transform, False, args.fold)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    print("Running inference...")
    pp, yp, gt_p, gt_y = collect(model, val_loader, device)

    os.makedirs(args.output, exist_ok=True)
    metrics = evaluate_and_print(pp, yp, gt_p, gt_y,
                                  label=f"Fine-tuned L2CS (fold {args.fold})")

    plot_reliability(metrics["conf"],
                     os.path.join(args.output, f"reliability_fold{args.fold}.png"),
                     f"Fine-tuned L2CS — reliability diagram (fold {args.fold})")

    # Save summary
    out_txt = os.path.join(args.output, f"eval_fold{args.fold}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Fine-tuned L2CS baseline — fold {args.fold}\n")
        f.write(f"Angular error : {metrics['ang_err']:.3f} deg\n")
        f.write(f"Joint NLL     : {metrics['nll']:.3f}\n")
        f.write(f"95% coverage  : {metrics['cov95']*100:.1f}%\n")
        f.write(f"90% coverage  : {metrics['cov90']*100:.1f}%\n")
        f.write(f"50% coverage  : {metrics['cov50']*100:.1f}%\n")
        f.write(f"ECE           : {metrics['ece']:.4f}\n")
    print(f"Saved summary -> {out_txt}")


if __name__ == "__main__":
    main()
