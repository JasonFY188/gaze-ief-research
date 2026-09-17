# L2CS-Net + IEF Refinement — Project Guide

## What this project is

This repo extends L2CS-Net (ResNet-50 backbone, gaze estimation) with an **IEF (Iterative Error Feedback) refinement head** — a small learned module that sits on top of a frozen L2CS backbone and iteratively corrects the gaze prediction over 3 steps.

The IEF head takes the backbone's spatial feature map + current gaze estimate as input and predicts a bounded correction to the bin logits. After 3 steps the final logits give both a more accurate mean angle and a calibrated uncertainty cone.

Conceptual reference: Carreira et al., "Human Pose Estimation with Iterative Error Feedback", CVPR 2016 (arXiv:1507.06550).

**Important constraint:** improvement is only valid when shown via **calibration** (coverage probability), not just a smaller angular error. Never claim success from angular error alone. Every result must include 2D joint 95% coverage.

## Two model variants

| Variant | Backbone | Bins | Bin width | Range | Checkpoint folder |
|---|---|---|---|---|---|
| `ief_gaze360` | L2CSNet_gaze360.pkl (Gaze360 dataset) | 90 | 4 deg/bin | ±180 deg | `checkpoints/ief_gaze360/` |
| `ief_mpiigaze` | fold{N}.pkl (MPIIFaceGaze-trained) | 28 | 3 deg/bin | ±42 deg | `checkpoints/ief_mpiigaze/` |

The IEF head is always trained on MPIIFaceGaze (leave-one-out, 15 folds). The backbone is always frozen during head training.

## Directory layout

```
L2CS-Net/
├── l2cs/
│   ├── model.py              # L2CS model definition (ResNet-50 based)
│   ├── wrapper.py            # L2CSWrapper — exposes spatial feature map
│   ├── refinement_head.py    # IEFGazeModel, RefinementHeadMLP, RefinementHeadAttn, RefinementConfig
│   ├── ief_pipeline.py       # IEFPipeline — drop-in for Pipeline (see below)
│   ├── pipeline.py           # Original Pipeline (plain L2CS-Net)
│   ├── refinement_dataset.py # MPIIGaze dataset + cached backbone logits
│   └── datasets.py           # Gaze360, Mpiigaze dataset classes
│
├── checkpoints/
│   ├── ief_gaze360/          # Checkpoints from train_refinement.py
│   │   └── best_fold{N}.pt   # saved by training, one per fold
│   └── ief_mpiigaze/         # Checkpoints from train_ief_mpiigaze.py
│       └── best_fold{N}.pt
│
├── train_refinement.py       # Train IEF head — Gaze360 backbone
├── train_ief_mpiigaze.py     # Train IEF head — MPIIGaze backbone
├── evaluate_refinement.py    # Full evaluation: angular error + coverage + plots
├── evaluate_ief_all_folds.py # Run evaluate_refinement.py across all 15 folds
├── evaluate_ief_mpiigaze_all_folds.py
│
├── Gaze360-20260603T102045Z-3-001/Gaze360/
│   └── L2CSNet_gaze360.pkl   # Frozen Gaze360 backbone weights
│
├── models/
│   └── MPIIGaze-20260529T070835Z-3-001/MPIIGaze/
│       └── fold{0..14}.pkl   # Per-fold MPIIGaze backbone weights
│
└── datasets/
    └── MPIIFaceGaze/
        ├── Image/            # Face images
        └── Label/            # p00.label … p14.label
```

## Key classes

### `RefinementConfig` (`l2cs/refinement_head.py`)
Dataclass holding all hyperparameters. Saved inside every checkpoint so you never need to specify it manually when loading.

Key fields:
- `num_bins` — must match the frozen backbone (90 for Gaze360, 28 for MPIIGaze)
- `num_steps` — refinement iterations after t=0 (default 3)
- `max_delta_deg` — max angle-mean shift per step (bounded step curriculum)
- `bin_width_deg`, `angle_offset_deg` — bin geometry for soft-argmax
- `head_type` — `"mlp"` (Step 3 baseline) or `"attn"` (Step 6 spatial attention upgrade)
- `freeze_backbone` — always True during head training

### `IEFGazeModel` (`l2cs/refinement_head.py`)
The full model. Wraps a `L2CSWrapper` (frozen backbone) and the refinement head.

`forward(x)` returns a **list of (pitch_logits, yaw_logits) tuples**, length `num_steps + 1`:
- `steps[0]` = raw backbone output (t=0, no refinement, use as baseline)
- `steps[-1]` = final refined prediction

### `L2CSWrapper` (`l2cs/wrapper.py`)
Wraps L2CS to also expose the spatial feature map from layer4 (before avgpool) via `forward_with_features(x)` → `GazeFeatures(pitch_logits, yaw_logits, feature_map)`.

**Naming quirk:** `fc_yaw_gaze` in model.py produces what callers call pitch, and vice versa. The wrapper follows the caller convention. Do not swap these when loading.

### `IEFPipeline` (`l2cs/ief_pipeline.py`)
Drop-in replacement for the original `Pipeline`. Same `.step(frame)` and `.predict_gaze(frame)` interface.

## How to use IEFPipeline (drop-in for L2CS-Net)

```python
# Old — plain L2CS-Net
from l2cs import Pipeline
pipe = Pipeline(weights="L2CSNet_gaze360.pkl", arch="ResNet50", device="cuda")

# New — IEF on Gaze360 backbone
from l2cs import IEFPipeline
pipe = IEFPipeline(
    backbone_weights="Gaze360-20260603T102045Z-3-001/Gaze360/L2CSNet_gaze360.pkl",
    head_weights="checkpoints/ief_gaze360/best_fold0.pt",
    device="cuda",
)

# New — IEF on MPIIGaze backbone (fold 0)
pipe = IEFPipeline(
    backbone_weights="models/MPIIGaze-20260529T070835Z-3-001/MPIIGaze/fold0.pkl",
    head_weights="checkpoints/ief_mpiigaze/best_fold0.pt",
    device="cuda",
)

# Usage is identical to Pipeline:
result = pipe.step(frame)          # frame: BGR numpy array
pitch, yaw = pipe.predict_gaze(frame)  # returns radians, shape (N,)
```

The `RefinementConfig` is loaded from the checkpoint automatically — no need to specify bin counts.

## Training

### Variant 1 — IEF on Gaze360 backbone (saves to `checkpoints/ief_gaze360/`)
```
python train_refinement.py \
  --weights   "Gaze360-20260603T102045Z-3-001/Gaze360/L2CSNet_gaze360.pkl" \
  --image_dir "datasets/MPIIFaceGaze/Image" \
  --label_dir "datasets/MPIIFaceGaze/Label" \
  --fold 0
```
Runs 15 folds. Default output: `checkpoints/ief_gaze360/`. Saves `best_fold{N}.pt` when validation angular error improves.

Key flags: `--num_steps` (default 3), `--max_delta` (default 10.0 deg), `--head_type mlp|attn`, `--lambda_cal` (calibration regularizer weight, default 0.01), `--grad_accum` (increase if OOM on 8 GB RTX 4060).

### Variant 2 — IEF on MPIIGaze backbone (saves to `checkpoints/ief_mpiigaze/`)
```
python train_ief_mpiigaze.py \
  --model_dir "models/MPIIGaze-20260529T070835Z-3-001/MPIIGaze" \
  --image_dir "datasets/MPIIFaceGaze/Image" \
  --label_dir "datasets/MPIIFaceGaze/Label" \
  --fold 0
```
Loads `fold{N}.pkl` for each fold (DataParallel-wrapped, unwrapped automatically). Default `--max_delta` is 5.0 deg (smaller because ±42 deg range vs ±180 deg).

## Evaluating

```
python evaluate_refinement.py \
  --weights    "Gaze360-.../L2CSNet_gaze360.pkl" \
  --checkpoint "checkpoints/ief_gaze360/best_fold0.pt" \
  --image_dir  "datasets/MPIIFaceGaze/Image" \
  --label_dir  "datasets/MPIIFaceGaze/Label" \
  --fold 0 \
  --output     "checkpoints/ief_gaze360"
```

Produces:
- `eval_fold{N}.txt` — angular error, NLL, 95/90/50% 2D joint coverage, ECE (raw and clipped)
- `perstep_fold{N}.png` — headline figure: error + coverage vs refinement step
- `reliability_fold{N}.png` — reliability diagram per step

**Coverage is 2D joint** (not average of two independent 1D intervals). `conf_needed[n]` = minimum HPD credible level to contain the true (pitch, yaw) cell. Good calibration: `coverage(α) ≈ α`.

## Checkpoint format

Both training scripts save:
```python
{
    "epoch": int,
    "cfg": RefinementConfig,          # full config — bin count, head type, etc.
    "head_state_dict": OrderedDict,   # only the refinement head weights (not backbone)
    "val_angular_error": float,
}
```
The backbone is never saved in the checkpoint — provide the original `.pkl` separately when loading.

## Bounded-step curriculum (do not remove)

The training loop does NOT train each step to jump straight to ground truth. Instead each step's target is clamped to be at most `max_delta_deg` from the **previous step's estimate**:
```python
pitch_tgt = prev_pitch + clamp(gt_pitch - prev_pitch, -max_delta, +max_delta)
```
Removing this causes ~10-point error degradation and drift over steps (from the IEF paper). The curriculum is wired into `run_epoch()` in both training scripts.

## Hardware notes

- Backbone is always frozen during head training (only head gradients flow)
- BN layers in the frozen backbone stay in eval mode even when `model.head.train()` — this is intentional

## What NOT to do

- Do not report a result as better based on smaller variance / spread alone — always pair with coverage
- Do not unfreeze the backbone unless explicitly asked (there is a `freeze_backbone` flag in RefinementConfig)
- Do not average two independent 1D coverage values — use the 2D joint HPD coverage from `evaluate_refinement.py`
