"""
Fine-tune the Gaze360 L2CS backbone on MPIIGaze — fair comparison baseline.

This answers the question: how much does iterative refinement (IEF) help
compared to simply fine-tuning the backbone directly on the target domain?

Both approaches use the same training data (same fold split), same loss
(NLL with soft Gaussian targets + log-ratio calibration regularizer), and
are evaluated with the same harness.

The fine-tuned model is saved as a standard L2CS checkpoint and can be
loaded by evaluate_finetune.py for comparison.

Usage:
  python train_finetune.py ^
    --weights   "path/to/L2CSNet_gaze360.pkl" ^
    --image_dir "datasets/MPIIFaceGaze/Image" ^
    --label_dir "datasets/MPIIFaceGaze/Label" ^
    --output    "output/finetune" ^
    --fold 0
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from l2cs import L2CS, select_device, Mpiigaze


# ── Same math helpers as train_refinement ─────────────────────────────────────
# Backbone config: Gaze360 — 90 bins, 4 deg/bin, ±180 deg
NUM_BINS    = 90
BIN_WIDTH   = 4.0
ANGLE_OFF   = 180.0


def soft_argmax(logits):
    probs = F.softmax(logits, dim=1)
    idx   = torch.arange(NUM_BINS, dtype=logits.dtype, device=logits.device)
    return (probs * (idx * BIN_WIDTH - ANGLE_OFF)).sum(dim=1)


def dist_variance(logits):
    probs = F.softmax(logits, dim=1)
    idx   = torch.arange(NUM_BINS, dtype=logits.dtype, device=logits.device)
    bins  = idx * BIN_WIDTH - ANGLE_OFF
    mean  = (probs * bins).sum(dim=1)
    return (probs * (bins - mean.unsqueeze(1)) ** 2).sum(dim=1)


def gaussian_target(target_deg, sigma=BIN_WIDTH):
    idx       = torch.arange(NUM_BINS, dtype=target_deg.dtype, device=target_deg.device)
    bin_angles = idx * BIN_WIDTH - ANGLE_OFF
    diff      = target_deg.unsqueeze(1) - bin_angles.unsqueeze(0)
    return F.softmax(-0.5 * (diff / sigma) ** 2, dim=1)


def nll_loss(logits, soft_targets):
    return -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def calibration_reg(logits, gt_deg):
    mean   = soft_argmax(logits)
    var    = dist_variance(logits)
    sq_err = (mean - gt_deg) ** 2
    return (torch.log(var + 1e-6) - torch.log(sq_err + 1e-6)).pow(2).mean()


def batch_angular_error(pitch_logits, yaw_logits, gt_pitch_deg, gt_yaw_deg):
    p  = soft_argmax(pitch_logits) * math.pi / 180
    y  = soft_argmax(yaw_logits)   * math.pi / 180
    gp = gt_pitch_deg.float() * math.pi / 180
    gy = gt_yaw_deg.float()   * math.pi / 180

    def to3d(pitch, yaw):
        return torch.stack([
            -torch.cos(pitch) * torch.sin(yaw),
            -torch.sin(pitch),
            -torch.cos(pitch) * torch.cos(yaw),
        ], dim=1)

    cos_sim = (to3d(p, y) * to3d(gp, gy)).sum(dim=1).clamp(-1.0, 0.9999999)
    return (torch.acos(cos_sim) * 180 / math.pi).mean().item()


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, lambda_cal, train=True):
    model.train(train)
    total_loss = 0.0
    total_err  = 0.0
    n = 0

    if train:
        optimizer.zero_grad()

    for images, _labels, cont_labels, _name in loader:
        images   = images.to(device)
        gt_pitch = cont_labels[:, 0].float().to(device)
        gt_yaw   = cont_labels[:, 1].float().to(device)

        with torch.set_grad_enabled(train):
            pitch_logits, yaw_logits = model(images)

            p_tgt = gaussian_target(gt_pitch)
            y_tgt = gaussian_target(gt_yaw)
            loss  = nll_loss(pitch_logits, p_tgt) + nll_loss(yaw_logits, y_tgt)
            loss += lambda_cal * (calibration_reg(pitch_logits, gt_pitch) +
                                  calibration_reg(yaw_logits,   gt_yaw))

        if train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            total_loss += loss.item()
            total_err  += batch_angular_error(pitch_logits.detach(),
                                               yaw_logits.detach(),
                                               gt_pitch, gt_yaw)
        n += 1

    nb = max(n, 1)
    return total_loss / nb, total_err / nb


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Gaze360 L2CS on MPIIGaze")
    p.add_argument("--weights",    required=True)
    p.add_argument("--image_dir",  required=True)
    p.add_argument("--label_dir",  required=True)
    p.add_argument("--output",     default="output/finetune")
    p.add_argument("--fold",       default=0,    type=int)
    p.add_argument("--num_epochs", default=30,   type=int)
    p.add_argument("--batch_size", default=32,   type=int)
    p.add_argument("--lr",         default=1e-4, type=float)
    p.add_argument("--lambda_cal", default=0.01, type=float)
    p.add_argument("--gpu",        default="0",  type=str)
    return p.parse_args()


def main():
    args   = parse_args()
    device = select_device(args.gpu)

    # ── Model — full backbone, all weights unfrozen ───────────────────────────
    model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins=NUM_BINS)
    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Fine-tune model: {n_params:,} params (all trainable)")

    # ── Data ──────────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    label_paths = sorted([
        os.path.join(args.label_dir, f)
        for f in os.listdir(args.label_dir) if f.endswith(".label")
    ])
    train_set = Mpiigaze(label_paths, args.image_dir, transform, True,  42, args.fold)
    val_set   = Mpiigaze(label_paths, args.image_dir, transform, False, 42, args.fold)
    print(f"Train: {len(train_set)}  Val: {len(val_set)}  (fold={args.fold})")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ── Optimizer — all parameters ────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.output, exist_ok=True)
    best_val_err = float("inf")
    log_path     = os.path.join(args.output, f"fold{args.fold}_train.log")

    with open(log_path, "w", encoding="utf-8") as log:
        header = (f"finetune fold={args.fold}  lr={args.lr}  "
                  f"batch={args.batch_size}  lambda_cal={args.lambda_cal}\n")
        print(header)
        log.write(header + "\n")

        for epoch in range(1, args.num_epochs + 1):
            t0 = time.time()
            train_loss, train_err = run_epoch(model, train_loader, optimizer,
                                               device, args.lambda_cal, train=True)
            with torch.no_grad():
                val_loss, val_err = run_epoch(model, val_loader, optimizer,
                                               device, args.lambda_cal, train=False)

            elapsed = time.time() - t0
            row = (f"Epoch {epoch}/{args.num_epochs}  ({elapsed:.0f}s)  "
                   f"train_loss={train_loss:.4f}  train_err={train_err:.3f}  "
                   f"val_loss={val_loss:.4f}  val_err={val_err:.3f}")
            print(row)
            log.write(row + "\n")
            log.flush()

            if val_err < best_val_err:
                best_val_err = val_err
                ckpt_path = os.path.join(args.output, f"best_fold{args.fold}.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_angular_error": val_err,
                    "num_bins": NUM_BINS,
                    "bin_width_deg": BIN_WIDTH,
                    "angle_offset_deg": ANGLE_OFF,
                }, ckpt_path)
                print(f"  --> Saved best checkpoint  val_err={val_err:.3f}")
                log.write(f"  --> Saved  val_err={val_err:.3f}\n")

    print(f"\nDone. Best val angular error: {best_val_err:.3f}")


if __name__ == "__main__":
    main()
