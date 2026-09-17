"""
Train the IEF refinement head on top of a native MPIIGaze-trained L2CS backbone.

Each fold uses its own matching backbone checkpoint (fold N backbone was
trained on subjects 0-14 except N), so there is no data leakage.

Backbone: 28-bin, 3 deg/bin, +-42 deg  (native MPIIGaze L2CS, DataParallel)
IEF head: same NLL + calibration loss as train_refinement.py

Usage:
  python train_ief_mpiigaze.py ^
    --model_dir "models/MPIIGaze-20260529T070835Z-3-001/MPIIGaze" ^
    --image_dir "datasets/MPIIFaceGaze/Image" ^
    --label_dir "datasets/MPIIFaceGaze/Label" ^
    --output    "output/refinement_mpiigaze" ^
    --fold 0
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from l2cs import L2CS, select_device, Mpiigaze
from l2cs.wrapper import L2CSWrapper
from l2cs.refinement_head import RefinementConfig, IEFGazeModel

# MPIIGaze backbone bin config
NUM_BINS  = 28
BIN_WIDTH = 3.0
ANGLE_OFF = 42.0


# ── Same math helpers as train_refinement.py ──────────────────────────────────

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

def run_epoch(model, loader, optimizer, cfg, device, lambda_cal, grad_accum, train=True):
    if train:
        model.head.train()
        model.wrapper.eval()
    else:
        model.eval()

    total_loss  = 0.0
    step_errors = [0.0] * (cfg.num_steps + 1)
    n_batches   = 0

    if train:
        optimizer.zero_grad()

    for batch_idx, (images, _labels, cont_labels, _name) in enumerate(loader):
        images   = images.to(device)
        gt_pitch = cont_labels[:, 0].float().to(device)
        gt_yaw   = cont_labels[:, 1].float().to(device)

        with torch.set_grad_enabled(train):
            steps = model(images)

            loss = torch.tensor(0.0, device=device)
            for t in range(1, len(steps)):
                pitch_t, yaw_t     = steps[t]
                pitch_prev, yaw_prev = steps[t - 1]

                with torch.no_grad():
                    pp = soft_argmax(pitch_prev)
                    yp = soft_argmax(yaw_prev)
                    d  = cfg.max_delta_deg
                    pitch_tgt = pp + torch.clamp(gt_pitch - pp, -d, d)
                    yaw_tgt   = yp + torch.clamp(gt_yaw   - yp, -d, d)

                p_tgt = gaussian_target(pitch_tgt)
                y_tgt = gaussian_target(yaw_tgt)
                loss += nll_loss(pitch_t, p_tgt) + nll_loss(yaw_t, y_tgt)
                loss += lambda_cal * (calibration_reg(pitch_t, gt_pitch) +
                                      calibration_reg(yaw_t,   gt_yaw))

        if train:
            (loss / grad_accum).backward()
            if (batch_idx + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

        with torch.no_grad():
            total_loss += loss.item()
            for t, (pl, yl) in enumerate(steps):
                step_errors[t] += batch_angular_error(pl.detach(), yl.detach(),
                                                       gt_pitch, gt_yaw)
        n_batches += 1

    if train and (n_batches % grad_accum != 0):
        optimizer.step()
        optimizer.zero_grad()

    nb = max(n_batches, 1)
    return total_loss / nb, [e / nb for e in step_errors]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train IEF on MPIIGaze native backbone")
    p.add_argument("--model_dir",  required=True, help="Dir with fold0.pkl ... fold14.pkl")
    p.add_argument("--image_dir",  required=True)
    p.add_argument("--label_dir",  required=True)
    p.add_argument("--output",     default="checkpoints/ief_mpiigaze")
    p.add_argument("--fold",       default=0,    type=int)
    p.add_argument("--num_steps",  default=3,    type=int)
    p.add_argument("--max_delta",  default=5.0,  type=float, dest="max_delta_deg",
                   help="Max step in degrees — smaller than Gaze360 version because range is +-42 deg")
    p.add_argument("--hidden_dim", default=256,  type=int)
    p.add_argument("--num_epochs", default=30,   type=int)
    p.add_argument("--batch_size", default=32,   type=int)
    p.add_argument("--lr",         default=1e-4, type=float)
    p.add_argument("--lambda_cal", default=0.01, type=float)
    p.add_argument("--grad_accum", default=1,    type=int, dest="grad_accum_steps")
    p.add_argument("--gpu",        default="0",  type=str)
    return p.parse_args()


def main():
    args   = parse_args()
    device = select_device(args.gpu)

    # ── Load fold-specific MPIIGaze backbone (DataParallel, 28 bins) ──────────
    ckpt_path = os.path.join(args.model_dir, f"fold{args.fold}.pkl")
    base  = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins=NUM_BINS)
    dp    = nn.DataParallel(base)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dp.load_state_dict(state)
    # L2CSWrapper unwraps DataParallel automatically
    wrapper = L2CSWrapper(dp)
    print(f"Loaded backbone: fold{args.fold}.pkl  (num_bins={wrapper.num_bins})")

    cfg = RefinementConfig(
        num_bins=NUM_BINS,
        num_steps=args.num_steps,
        max_delta_deg=args.max_delta_deg,
        hidden_dim=args.hidden_dim,
        freeze_backbone=True,
        head_type="mlp",
        bin_width_deg=BIN_WIDTH,
        angle_offset_deg=ANGLE_OFF,
    )
    model = IEFGazeModel(wrapper, cfg).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_trainable:,}  |  Config: {cfg}")

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
    print(f"Train: {len(train_set)}  Val: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    optimizer = torch.optim.Adam(model.head.parameters(), lr=args.lr)

    os.makedirs(args.output, exist_ok=True)
    best_val_err = float("inf")
    log_path = os.path.join(args.output, f"fold{args.fold}_train.log")

    with open(log_path, "w", encoding="utf-8") as log:
        header = (f"fold={args.fold}  backbone=mpiigaze_native  steps={cfg.num_steps}  "
                  f"max_delta={cfg.max_delta_deg}  lambda_cal={args.lambda_cal}\n")
        print(header)
        log.write(header + "\n")

        for epoch in range(1, args.num_epochs + 1):
            t0 = time.time()
            train_loss, train_errs = run_epoch(
                model, train_loader, optimizer, cfg, device,
                args.lambda_cal, args.grad_accum_steps, train=True)
            with torch.no_grad():
                val_loss, val_errs = run_epoch(
                    model, val_loader, optimizer, cfg, device,
                    args.lambda_cal, args.grad_accum_steps, train=False)

            elapsed = time.time() - t0
            summary = (f"\nEpoch {epoch}/{args.num_epochs}  ({elapsed:.0f}s)  "
                       f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
            print(summary);  log.write(summary + "\n")

            for t in range(cfg.num_steps + 1):
                tag = "  <-- baseline" if t == 0 else ""
                row = (f"  step {t}: train_err={train_errs[t]:.3f}  "
                       f"val_err={val_errs[t]:.3f}{tag}")
                print(row);  log.write(row + "\n")
            log.flush()

            if val_errs[-1] < best_val_err:
                best_val_err = val_errs[-1]
                ckpt_out = os.path.join(args.output, f"best_fold{args.fold}.pt")
                torch.save({
                    "epoch": epoch,
                    "cfg": cfg,
                    "head_state_dict": model.head.state_dict(),
                    "val_angular_error": best_val_err,
                }, ckpt_out)
                print(f"  --> Saved  val_err={best_val_err:.3f}")
                log.write(f"  --> Saved  val_err={best_val_err:.3f}\n")

    print(f"\nDone. Best val error: {best_val_err:.3f}")


if __name__ == "__main__":
    main()
