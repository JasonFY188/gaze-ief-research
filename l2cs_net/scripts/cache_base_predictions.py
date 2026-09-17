"""
One-shot script: run the frozen Gaze360 L2CS backbone over all MPIIGaze face
images and save per-sample (pitch_logits, yaw_logits) tensors to a single cache
file.  Run this once before training the refinement head.

The Gaze360 backbone outputs 90-bin logits (4 deg/bin, ±180°).  Those raw
logits become the t=0 estimate fed into the IEF refinement head at training time.

Output format — a single .pt file containing a dict:
  {
    "p00\\face\\1.jpg": {
        "pitch": FloatTensor(90,),   # raw logits (pre-softmax)
        "yaw":   FloatTensor(90,),
    },
    ...
  }
Keys are the face-path strings exactly as they appear in the label files, so
RefinementDataset can do O(1) lookups.

Usage:
  python scripts/cache_base_predictions.py ^
    --weights   "path/to/L2CSNet_gaze360.pkl" ^
    --image_dir "datasets/MPIIFaceGaze/Image" ^
    --label_dir "datasets/MPIIFaceGaze/Label" ^
    --output    "datasets/MPIIFaceGaze/base_cache.pt" ^
    [--batch_size 64] [--limit N]
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from l2cs.model import L2CS
from l2cs.wrapper import L2CSWrapper


TRANSFORM = transforms.Compose([
    transforms.Resize(448),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def parse_args():
    p = argparse.ArgumentParser(description="Cache Gaze360 base predictions for MPIIGaze")
    p.add_argument("--weights",    required=True,  help="Path to L2CSNet_gaze360.pkl")
    p.add_argument("--image_dir",  required=True,  help="MPIIFaceGaze Image directory")
    p.add_argument("--label_dir",  required=True,  help="MPIIFaceGaze Label directory")
    p.add_argument("--output",     required=True,  help="Output .pt cache file path")
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--limit",      default=None, type=int,
                   help="Process only the first N images (for quick smoke-tests)")
    return p.parse_args()


def collect_face_paths(label_dir):
    """Return sorted list of unique face-image paths from all label files."""
    paths = set()
    for fname in sorted(os.listdir(label_dir)):
        if not fname.endswith(".label"):
            continue
        with open(os.path.join(label_dir, fname)) as f:
            lines = f.readlines()[1:]   # skip header
        for line in lines:
            face = line.strip().split(" ")[0]
            paths.add(face)
    return sorted(paths)


@torch.no_grad()
def run_batch(wrapper, img_tensors, device):
    batch = torch.stack(img_tensors).to(device)
    feats = wrapper.forward_with_features(batch)
    # Return logits on CPU so we don't hold GPU memory in the cache dict
    return feats.pitch_logits.cpu(), feats.yaw_logits.cpu()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load frozen Gaze360 backbone (90 bins, ResNet50)
    base = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins=90)
    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    base.load_state_dict(state)
    base.to(device)
    base.eval()
    wrapper = L2CSWrapper(base)
    wrapper.eval()
    print(f"Loaded weights from {args.weights}  (num_bins={wrapper.num_bins})")

    # Collect unique face paths
    all_paths = collect_face_paths(args.label_dir)
    if args.limit:
        all_paths = all_paths[: args.limit]
    print(f"Found {len(all_paths)} unique face images to process")

    if os.path.exists(args.output):
        print(f"Cache already exists at {args.output} — loading to check coverage")
        existing = torch.load(args.output, weights_only=False)
        missing = [p for p in all_paths if p not in existing]
        if not missing:
            print("Cache is complete. Nothing to do.")
            return
        print(f"{len(missing)} images not yet cached — processing those only")
        all_paths = missing
        cache = existing
    else:
        cache = {}

    # Process in batches
    batch_imgs, batch_keys = [], []
    n_processed = 0

    def flush():
        nonlocal n_processed
        if not batch_imgs:
            return
        pitch_batch, yaw_batch = run_batch(wrapper, batch_imgs, device)
        for key, p, y in zip(batch_keys, pitch_batch, yaw_batch):
            cache[key] = {"pitch": p, "yaw": y}
        n_processed += len(batch_imgs)
        batch_imgs.clear()
        batch_keys.clear()

    for face_path in tqdm(all_paths, desc="Caching predictions"):
        img_full = os.path.join(args.image_dir, face_path)
        try:
            img = Image.open(img_full).convert("RGB")
        except FileNotFoundError:
            print(f"  WARNING: image not found, skipping: {img_full}")
            continue
        batch_imgs.append(TRANSFORM(img))
        batch_keys.append(face_path)
        if len(batch_imgs) == args.batch_size:
            flush()

    flush()  # remainder

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(cache, args.output)
    print(f"\nSaved {len(cache)} entries → {args.output}")


if __name__ == "__main__":
    main()
