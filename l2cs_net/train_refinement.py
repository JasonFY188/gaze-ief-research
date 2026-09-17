"""
Train the IEF refinement head on MPIIGaze with a frozen Gaze360 backbone.

Usage example (fold 0 held out for validation):
  python train_refinement.py ^
    --weights   "path/to/L2CSNet_gaze360.pkl" ^
    --image_dir "datasets/MPIIFaceGaze/Image" ^
    --label_dir "datasets/MPIIFaceGaze/Label" ^
    --output    "output/refinement" ^
    --fold 0

Key flags:
  --num_steps     refinement iterations after t=0 (default 3)
  --max_delta     max angle-mean shift per step in degrees (default 10)
  --lambda_cal    calibration regularizer weight (default 0.01)
  --grad_accum    gradient accumulation steps (use if you hit OOM; default 1)
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
from l2cs.wrapper import L2CSWrapper
from l2cs.refinement_head import RefinementConfig, IEFGazeModel


# ── Gaze math helpers ─────────────────────────────────────────────────────────

def soft_argmax(logits, bin_width, angle_offset):
    """(B, num_bins) → (B,) predicted angle in degrees."""
    probs = F.softmax(logits, dim=1)
    idx = torch.arange(logits.size(1), dtype=logits.dtype, device=logits.device)
    return (probs * (idx * bin_width - angle_offset)).sum(dim=1)


def dist_variance(logits, bin_width, angle_offset):
    """(B, num_bins) → (B,) predicted variance in degrees²."""
    probs = F.softmax(logits, dim=1)
    idx = torch.arange(logits.size(1), dtype=logits.dtype, device=logits.device)
    bins = idx * bin_width - angle_offset
    mean = (probs * bins).sum(dim=1)
    return (probs * (bins - mean.unsqueeze(1)) ** 2).sum(dim=1)


def gaussian_target(target_deg, num_bins, bin_width, angle_offset, sigma):
    """Soft Gaussian distribution over bins centred on target_deg.
    target_deg: (B,) — angles in degrees
    Returns:    (B, num_bins) — normalised soft probabilities
    """
    idx = torch.arange(num_bins, dtype=target_deg.dtype, device=target_deg.device)
    bin_angles = idx * bin_width - angle_offset           # (num_bins,)
    diff = target_deg.unsqueeze(1) - bin_angles.unsqueeze(0)   # (B, num_bins)
    return F.softmax(-0.5 * (diff / sigma) ** 2, dim=1)


def nll_loss(logits, soft_targets):
    """NLL / cross-entropy between predicted logits and a soft target distribution."""
    return -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def calibration_reg(logits, gt_deg, bin_width, angle_offset):
    """Log-ratio calibration penalty: (log var − log sq_err)².
    Scale-invariant — bounded regardless of how large errors are.
    Zero when predicted variance equals the actual squared error.
    """
    mean   = soft_argmax(logits, bin_width, angle_offset)
    var    = dist_variance(logits, bin_width, angle_offset)
    sq_err = (mean - gt_deg) ** 2
    return (torch.log(var + 1e-6) - torch.log(sq_err + 1e-6)).pow(2).mean()


def batch_angular_error(pitch_logits, yaw_logits, gt_pitch_deg, gt_yaw_deg,
                        bin_width, angle_offset):
    """Mean angular error in degrees (3D arc distance) over the batch."""
    p = soft_argmax(pitch_logits, bin_width, angle_offset) * math.pi / 180
    y = soft_argmax(yaw_logits,   bin_width, angle_offset) * math.pi / 180
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


# ── One epoch ────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, cfg, device,
              lambda_cal, grad_accum_steps, train=True):
    """Run one full pass over `loader`.

    Returns:
        avg_loss:   float
        step_errors: list[float] of mean angular error at each step (t=0..N)
        step_vars:   list[float] of mean predicted pitch variance at each step
    """
    if train:
        model.head.train()
        model.wrapper.eval()   # backbone BN always uses running statistics
    else:
        model.eval()

    BW  = cfg.bin_width_deg
    OFF = cfg.angle_offset_deg

    total_loss   = 0.0
    step_errors  = [0.0] * (cfg.num_steps + 1)
    step_vars    = [0.0] * (cfg.num_steps + 1)
    n_batches    = 0

    if train:
        optimizer.zero_grad()

    for batch_idx, (images, _labels, cont_labels, _name) in enumerate(loader):
        images   = images.to(device)
        gt_pitch = cont_labels[:, 0].float().to(device)   # degrees
        gt_yaw   = cont_labels[:, 1].float().to(device)   # degrees

        with torch.set_grad_enabled(train):
            steps = model(images)   # [(pitch_t, yaw_t), ...] length = num_steps+1

            loss = torch.tensor(0.0, device=device)
            for t in range(1, len(steps)):
                pitch_t, yaw_t     = steps[t]
                pitch_prev, yaw_prev = steps[t - 1]

                # Bounded-step curriculum: target is one max_delta step
                # closer to ground truth from the previous estimate.
                with torch.no_grad():
                    pp = soft_argmax(pitch_prev, BW, OFF)
                    yp = soft_argmax(yaw_prev,   BW, OFF)
                    d  = cfg.max_delta_deg
                    pitch_tgt = pp + torch.clamp(gt_pitch - pp, -d, d)
                    yaw_tgt   = yp + torch.clamp(gt_yaw   - yp, -d, d)

                # NLL with soft Gaussian targets (sigma = 1 bin width)
                p_tgt = gaussian_target(pitch_tgt, cfg.num_bins, BW, OFF, sigma=BW)
                y_tgt = gaussian_target(yaw_tgt,   cfg.num_bins, BW, OFF, sigma=BW)
                loss += nll_loss(pitch_t, p_tgt) + nll_loss(yaw_t, y_tgt)

                # Calibration regularizer
                loss += lambda_cal * (
                    calibration_reg(pitch_t, gt_pitch, BW, OFF) +
                    calibration_reg(yaw_t,   gt_yaw,   BW, OFF)
                )

        if train:
            (loss / grad_accum_steps).backward()
            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Logging (detached, no memory overhead)
        with torch.no_grad():
            total_loss += loss.item()
            for t, (pl, yl) in enumerate(steps):
                step_errors[t] += batch_angular_error(pl, yl, gt_pitch, gt_yaw, BW, OFF)
                step_vars[t]   += dist_variance(pl, BW, OFF).mean().item()
        n_batches += 1

    # Flush any remainder from gradient accumulation
    if train and (n_batches % grad_accum_steps != 0):
        optimizer.step()
        optimizer.zero_grad()

    nb = max(n_batches, 1)
    return (total_loss / nb,
            [e / nb for e in step_errors],
            [v / nb for v in step_vars])


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train IEF gaze refinement head")
    p.add_argument("--weights",     required=True,          help="Path to L2CSNet_gaze360.pkl")
    p.add_argument("--image_dir",   required=True,          help="MPIIFaceGaze Image directory")
    p.add_argument("--label_dir",   required=True,          help="MPIIFaceGaze Label directory")
    p.add_argument("--output",      default="checkpoints/ief_gaze360")
    p.add_argument("--fold",        default=0,    type=int, help="Hold-out fold for validation (0-14)")
    # Model
    p.add_argument("--num_steps",   default=3,    type=int)
    p.add_argument("--max_delta",   default=10.0, type=float, dest="max_delta_deg",
                   help="Max angle-mean shift per step (degrees)")
    p.add_argument("--hidden_dim",  default=256,  type=int)
    p.add_argument("--head_type",   default="mlp", choices=["mlp", "attn"],
                   help="Head architecture: mlp (Step 3) or attn (Step 6)")
    p.add_argument("--d_attn",      default=256,  type=int,
                   help="Attention projection dim (attn head only)")
    # Training
    p.add_argument("--num_epochs",  default=30,   type=int)
    p.add_argument("--batch_size",  default=32,   type=int)
    p.add_argument("--lr",          default=1e-4, type=float)
    p.add_argument("--lambda_cal",  default=0.01, type=float,
                   help="Weight for calibration regularizer")
    p.add_argument("--grad_accum",  default=1,    type=int, dest="grad_accum_steps",
                   help="Gradient accumulation steps (increase if OOM)")
    p.add_argument("--gpu",         default="0",  type=str)
    return p.parse_args()


def main():
    args = parse_args()
    device = select_device(args.gpu)

    # ── Model ────────────────────────────────────────────────────────────────
    base = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins=90)
    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    base.load_state_dict(state)
    wrapper = L2CSWrapper(base)

    cfg = RefinementConfig(
        num_bins=90,
        num_steps=args.num_steps,
        max_delta_deg=args.max_delta_deg,
        hidden_dim=args.hidden_dim,
        freeze_backbone=True,
        head_type=args.head_type,
        d_attn=args.d_attn,
    )
    model = IEFGazeModel(wrapper, cfg).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"Model ready. Trainable: {n_trainable:,} / {n_total:,} params")
    print(f"Config: {cfg}")

    # ── Data ─────────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    label_paths = sorted([
        os.path.join(args.label_dir, f)
        for f in os.listdir(args.label_dir) if f.endswith(".label")
    ])

    # angle=42 matches MPIIGaze's ±42° range
    train_set = Mpiigaze(label_paths, args.image_dir, transform, True,  42, args.fold)
    val_set   = Mpiigaze(label_paths, args.image_dir, transform, False, 42, args.fold)
    print(f"Train: {len(train_set)} samples  |  Val: {len(val_set)} samples  (fold={args.fold})")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.head.parameters(), lr=args.lr)

    os.makedirs(args.output, exist_ok=True)
    best_val_error = float("inf")
    log_path = os.path.join(args.output, f"fold{args.fold}_train.log")

    with open(log_path, "w", encoding="utf-8") as log:
        header = (f"fold={args.fold}  steps={cfg.num_steps}  "
                  f"max_delta={cfg.max_delta_deg}°  lambda_cal={args.lambda_cal}  "
                  f"lr={args.lr}  batch={args.batch_size}\n")
        print(header)
        log.write(header + "\n")

        for epoch in range(1, args.num_epochs + 1):
            t0 = time.time()

            train_loss, train_errs, train_vars = run_epoch(
                model, train_loader, optimizer, cfg, device,
                args.lambda_cal, args.grad_accum_steps, train=True)

            with torch.no_grad():
                val_loss, val_errs, val_vars = run_epoch(
                    model, val_loader, optimizer, cfg, device,
                    args.lambda_cal, args.grad_accum_steps, train=False)

            elapsed = time.time() - t0
            summary = (f"\nEpoch {epoch}/{args.num_epochs}  ({elapsed:.0f}s)  "
                       f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
            print(summary)
            log.write(summary + "\n")

            for t in range(cfg.num_steps + 1):
                tag = "  <-- baseline (backbone only)" if t == 0 else ""
                row = (f"  step {t}: "
                       f"train_err={train_errs[t]:.3f}°  val_err={val_errs[t]:.3f}°  "
                       f"val_var={val_vars[t]:.2f}deg2{tag}")
                print(row)
                log.write(row + "\n")
            log.flush()

            # Save if best on final step
            final_val = val_errs[-1]
            if final_val < best_val_error:
                best_val_error = final_val
                ckpt_path = os.path.join(args.output, f"best_fold{args.fold}.pt")
                torch.save({
                    "epoch": epoch,
                    "cfg": cfg,
                    "head_state_dict": model.head.state_dict(),
                    "val_angular_error": final_val,
                }, ckpt_path)
                print(f"  --> Saved best checkpoint  val_err={final_val:.3f}°")
                log.write(f"  --> Saved best checkpoint  val_err={final_val:.3f}°\n")

    print(f"\nTraining complete. Best val angular error: {best_val_error:.3f}°")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
