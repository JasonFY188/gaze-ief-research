"""
Fine-tune the Gaze360 L2CS backbone on MPIIGaze using the original L2CS
training objective: CrossEntropy on binned labels + MSE on continuous angles.

This is the fair comparison baseline: same model architecture (90-bin ResNet50),
same starting weights (Gaze360 pretrained), same MPIIGaze fold split — but
adapted using standard L2CS training rather than the IEF approach.

Usage:
  python train_finetune_l2cs.py ^
    --weights   "path/to/L2CSNet_gaze360.pkl" ^
    --image_dir "datasets/MPIIFaceGaze/Image" ^
    --label_dir "datasets/MPIIFaceGaze/Label" ^
    --output    "output/finetune_l2cs" ^
    --fold 0
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from l2cs import L2CS, select_device
from l2cs.utils import gazeto3d, angular


# ── Dataset — MPIIGaze with 90-bin Gaze360 labels ────────────────────────────

class MpiigazeGaze360Bins(Dataset):
    """MPIIGaze with binned labels matching the Gaze360 90-bin structure.

    The Gaze360 backbone uses 90 bins at 4 deg/bin spanning -180 to +176 deg.
    MPIIGaze GT angles are within +-42 deg so they map cleanly to bins 34-55.
    """

    BIN_WIDTH  = 4.0
    ANGLE_OFF  = 180.0
    NUM_BINS   = 90
    BINS       = np.array([b * 4.0 - 180.0 for b in range(90)])  # bin centres

    def __init__(self, label_paths, image_dir, transform, train=True, fold=0):
        self.image_dir = image_dir
        self.transform = transform
        self.lines     = []

        paths = label_paths.copy()
        if train:
            paths.pop(fold)
        else:
            paths = [paths[fold]]

        for path in (paths if isinstance(paths, list) else [paths]):
            with open(path) as f:
                lines = f.readlines()[1:]
            for line in lines:
                gaze2d = line.strip().split(" ")[7]
                label  = np.array(gaze2d.split(",")).astype("float")
                pitch_deg = label[0] * 180 / np.pi
                yaw_deg   = label[1] * 180 / np.pi
                if abs(pitch_deg) <= 42 and abs(yaw_deg) <= 42:
                    self.lines.append(line)

        split = "train" if train else "test"
        print(f"MpiigazeGaze360Bins [{split}, fold={fold}]: {len(self.lines)} samples")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        parts    = self.lines[idx].strip().split(" ")
        face     = parts[0]
        gaze2d   = parts[7]

        label     = np.array(gaze2d.split(",")).astype("float")
        pitch_deg = float(label[0] * 180 / np.pi)
        yaw_deg   = float(label[1] * 180 / np.pi)

        # Binned labels for CE loss (Gaze360 bin structure)
        pitch_bin = int(np.digitize([pitch_deg], self.BINS)[0] - 1)
        yaw_bin   = int(np.digitize([yaw_deg],   self.BINS)[0] - 1)
        pitch_bin = max(0, min(pitch_bin, self.NUM_BINS - 1))
        yaw_bin   = max(0, min(yaw_bin,   self.NUM_BINS - 1))

        img = Image.open(os.path.join(self.image_dir, face)).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor([pitch_bin, yaw_bin]), torch.FloatTensor([pitch_deg, yaw_deg])


# ── Angular error (same as original test.py) ──────────────────────────────────

def compute_angular_error_batch(pitch_pred_deg, yaw_pred_deg, pitch_gt_deg, yaw_gt_deg):
    """Mean angular error in degrees over a batch."""
    errs = []
    for p, y, gp, gy in zip(pitch_pred_deg, yaw_pred_deg, pitch_gt_deg, yaw_gt_deg):
        pred_3d = gazeto3d([p * np.pi / 180, y * np.pi / 180])
        gt_3d   = gazeto3d([gp * np.pi / 180, gy * np.pi / 180])
        errs.append(angular(pred_3d, gt_3d))
    return float(np.mean(errs))


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, reg_criterion,
              softmax, idx_tensor, alpha, device, train=True):
    model.train(train)
    total_err = 0.0
    total_loss = 0.0
    n = 0

    for images, bin_labels, cont_labels in loader:
        images     = images.to(device)
        label_pitch_bin = bin_labels[:, 0].to(device)
        label_yaw_bin   = bin_labels[:, 1].to(device)
        label_pitch_deg = cont_labels[:, 0].float().to(device)
        label_yaw_deg   = cont_labels[:, 1].float().to(device)

        with torch.set_grad_enabled(train):
            pitch_logits, yaw_logits = model(images)

            # Cross-entropy loss on binned labels
            loss_ce_pitch = criterion(pitch_logits, label_pitch_bin)
            loss_ce_yaw   = criterion(yaw_logits,   label_yaw_bin)

            # MSE loss on continuous angles
            pitch_probs = softmax(pitch_logits)
            yaw_probs   = softmax(yaw_logits)
            pitch_pred  = (torch.sum(pitch_probs * idx_tensor, dim=1) * 4.0 - 180.0)
            yaw_pred    = (torch.sum(yaw_probs   * idx_tensor, dim=1) * 4.0 - 180.0)

            loss_mse_pitch = reg_criterion(pitch_pred, label_pitch_deg)
            loss_mse_yaw   = reg_criterion(yaw_pred,   label_yaw_deg)

            loss = (loss_ce_pitch + alpha * loss_mse_pitch +
                    loss_ce_yaw   + alpha * loss_mse_yaw)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            total_loss += loss.item()
            total_err  += compute_angular_error_batch(
                pitch_pred.detach().cpu().numpy(),
                yaw_pred.detach().cpu().numpy(),
                label_pitch_deg.cpu().numpy(),
                label_yaw_deg.cpu().numpy(),
            )
        n += 1

    nb = max(n, 1)
    return total_loss / nb, total_err / nb


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Gaze360 L2CS on MPIIGaze (original L2CS loss)")
    p.add_argument("--weights",    required=True)
    p.add_argument("--image_dir",  required=True)
    p.add_argument("--label_dir",  required=True)
    p.add_argument("--output",     default="output/finetune_l2cs")
    p.add_argument("--fold",       default=0,    type=int)
    p.add_argument("--num_epochs", default=30,   type=int)
    p.add_argument("--batch_size", default=32,   type=int)
    p.add_argument("--lr",         default=1e-4, type=float)
    p.add_argument("--alpha",      default=1.0,  type=float,
                   help="MSE loss weight (same default as original L2CS training)")
    p.add_argument("--gpu",        default="0",  type=str)
    return p.parse_args()


def main():
    args   = parse_args()
    device = select_device(args.gpu)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins=90)
    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    print(f"Loaded Gaze360 weights. Total params: {sum(p.numel() for p in model.parameters()):,}")

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
    train_set = MpiigazeGaze360Bins(label_paths, args.image_dir, transform, True,  args.fold)
    val_set   = MpiigazeGaze360Bins(label_paths, args.image_dir, transform, False, args.fold)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ── Loss + optimizer (same as original L2CS training) ─────────────────────
    criterion     = nn.CrossEntropyLoss().to(device)
    reg_criterion = nn.MSELoss().to(device)
    softmax       = nn.Softmax(dim=1)
    idx_tensor    = torch.FloatTensor(list(range(90))).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.output, exist_ok=True)
    best_val_err = float("inf")
    log_path     = os.path.join(args.output, f"fold{args.fold}_train.log")

    with open(log_path, "w", encoding="utf-8") as log:
        header = f"finetune_l2cs  fold={args.fold}  lr={args.lr}  batch={args.batch_size}  alpha={args.alpha}\n"
        print(header)
        log.write(header + "\n")

        for epoch in range(1, args.num_epochs + 1):
            t0 = time.time()
            train_loss, train_err = run_epoch(
                model, train_loader, optimizer, criterion, reg_criterion,
                softmax, idx_tensor, args.alpha, device, train=True)
            with torch.no_grad():
                val_loss, val_err = run_epoch(
                    model, val_loader, optimizer, criterion, reg_criterion,
                    softmax, idx_tensor, args.alpha, device, train=False)

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
                    "num_bins": 90,
                    "bin_width_deg": 4.0,
                    "angle_offset_deg": 180.0,
                }, ckpt_path)
                print(f"  --> Saved  val_err={val_err:.3f}")
                log.write(f"  --> Saved  val_err={val_err:.3f}\n")

    print(f"\nDone. Best val angular error: {best_val_err:.3f}")


if __name__ == "__main__":
    main()
