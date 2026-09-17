import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


TRANSFORM = transforms.Compose([
    transforms.Resize(448),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class RefinementDataset(Dataset):
    """MPIIGaze dataset augmented with cached Gaze360 base-model logits.

    Mirrors the Mpiigaze split logic (leave-one-out over 15 folds) but adds
    the frozen backbone's per-sample bin logits so the refinement head has a
    t=0 estimate without re-running the backbone inside the training loop.

    Each sample returns:
        image              FloatTensor (3, 448, 448)
        gt_pitch_deg       float  — ground-truth pitch in degrees
        gt_yaw_deg         float  — ground-truth yaw in degrees
        base_pitch_logits  FloatTensor (90,) — raw logits from Gaze360 backbone
        base_yaw_logits    FloatTensor (90,) — raw logits from Gaze360 backbone

    Args:
        label_paths: list of all 15 label file paths (p00.label … p14.label)
        image_dir:   root directory for face images
        cache_path:  path to the .pt cache produced by scripts/cache_base_predictions.py
        train:       if True, use all folds except `fold`; if False, use only `fold`
        angle:       max gaze angle in degrees to include (default 42 for MPIIGaze)
        fold:        which subject fold to hold out (0–14)
        transform:   optional override; defaults to Resize(448)+Normalize
    """

    def __init__(
        self,
        label_paths,
        image_dir,
        cache_path,
        train=True,
        angle=42,
        fold=0,
        transform=None,
    ):
        self.image_dir = image_dir
        self.transform = transform or TRANSFORM
        self.lines = []

        # Replicate Mpiigaze split logic exactly
        paths = label_paths.copy()
        if train:
            paths.pop(fold)          # all subjects except the test fold
        else:
            paths = [paths[fold]]    # only the test fold

        if isinstance(paths, list):
            for path in paths:
                with open(path) as f:
                    lines = f.readlines()[1:]   # skip header
                for line in lines:
                    gaze2d = line.strip().split(" ")[7]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs(label[0] * 180 / np.pi) <= angle and \
                       abs(label[1] * 180 / np.pi) <= angle:
                        self.lines.append(line)
        else:
            with open(paths) as f:
                lines = f.readlines()[1:]
            for line in lines:
                gaze2d = line.strip().split(" ")[7]
                label = np.array(gaze2d.split(",")).astype("float")
                if abs(label[0] * 180 / np.pi) <= 42 and \
                   abs(label[1] * 180 / np.pi) <= 42:
                    self.lines.append(line)

        print(f"RefinementDataset: {len(self.lines)} samples "
              f"({'train' if train else 'test'}, fold={fold})")

        # Load cache — keys are face-path strings as in the label files
        self.cache = torch.load(cache_path, weights_only=False)
        n_missing = sum(
            1 for line in self.lines
            if line.strip().split(" ")[0] not in self.cache
        )
        if n_missing:
            raise RuntimeError(
                f"{n_missing} samples have no cache entry. "
                f"Re-run scripts/cache_base_predictions.py."
            )

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        parts = self.lines[idx].strip().split(" ")
        face_path = parts[0]
        gaze2d    = parts[7]

        label = np.array(gaze2d.split(",")).astype("float")
        gt_pitch_deg = float(label[0] * 180 / np.pi)
        gt_yaw_deg   = float(label[1] * 180 / np.pi)

        img = Image.open(os.path.join(self.image_dir, face_path)).convert("RGB")
        if self.transform:
            img = self.transform(img)

        cached = self.cache[face_path]
        base_pitch_logits = cached["pitch"]   # FloatTensor (90,)
        base_yaw_logits   = cached["yaw"]     # FloatTensor (90,)

        return img, gt_pitch_deg, gt_yaw_deg, base_pitch_logits, base_yaw_logits
