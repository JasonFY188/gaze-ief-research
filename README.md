# Gaze-IEF Research Pipeline

Gaze estimation pipeline combining **L2CS-Net / IEF** (Iterative Error Feedback) gaze direction with **DeepGaze IIE** saliency and **VGGT** 3D reconstruction to predict where a person is looking in a scene.

## How it works

```
Face image → L2CS-Net (or IEF) → gaze direction + uncertainty cone
                                          ↓
Reference image × gaze cone mask → DeepGaze IIE saliency map
                                          ↓
        Product of Experts fusion (geometry × saliency)
                                          ↓
        Heatmap peak → VGGT 3D world point → angular error vs GT
```

## Key files

| File | Description |
|------|-------------|
| `final_form.py` | Plain L2CS-Net pipeline with multiple cone configs |
| `final_form_ief.py` | IEF per-fold + ensemble pipeline |
| `paths.py` | Default file locations (override per machine in `paths_local.py`) |
| `check_env.py` | Checks libraries, GPU and file locations on a new machine |
| `blob_projection_utils.py` | Projects 3D gaze blobs into reference views |
| `l2cs_net/l2cs/` | L2CS-Net library with IEF extensions (see `l2cs_net/CLAUDE.md`) |
| `l2cs_net/train_refinement.py` | Train IEF head on Gaze360 backbone |
| `l2cs_net/train_ief_mpiigaze.py` | Train IEF head on MPIIGaze backbone |
| `pyproject.toml` / `uv.lock` | Exact library versions — the environment is rebuilt from these |

## Setup (Windows or Linux)

The environment is managed with [uv](https://docs.astral.sh/uv/). It creates an isolated `.venv/`
inside this folder with its own Python 3.11, so it cannot clash with other Python installs,
conda envs or projects on the machine. `uv.lock` pins every library (including the GitHub-only
ones: VGGT, DeepGaze, face-detection) to the exact versions that were tested.

### 1. Install uv (once per machine)
```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and install
```bash
git clone https://github.com/JasonFY188/gaze-ief-research
cd gaze-ief-research
uv sync
```
This installs PyTorch (CUDA 12.8 build), VGGT, DeepGaze, face-detection, and `l2cs_net/` in
editable mode — changes you make in `l2cs_net/l2cs/` take effect immediately, no reinstall.

> **GPU driver:** the CUDA 12.8 build needs NVIDIA driver ≥ 570. It supports RTX 50xx
> (required for these) as well as 30xx/40xx cards.

### 3. Check the environment
```bash
uv run python check_env.py
```
All libraries and CUDA should show `[OK]`. Files marked `[--]` are just not in place yet (step 4).

### 4. Put model weights and data in place
By default everything lives inside the repo (these folders are git-ignored):

```
gaze-ief-research/
├── weights/
│   ├── L2CSNet_gaze360.pkl          # L2CS-Net releases: https://github.com/Ahmednull/L2CS-Net
│   ├── centerbias_mit1003.npy       # DeepGaze centerbias (MIT1003)
│   └── MPIIGaze/fold0.pkl … fold14.pkl   # train with l2cs_net/train.py on MPIIGaze
├── l2cs_net/checkpoints/
│   ├── ief_gaze360/best_fold*.pt    # l2cs_net/train_refinement.py
│   └── ief_mpiigaze/best_fold*.pt   # l2cs_net/train_ief_mpiigaze.py
├── data/experiment/                 # dataset root (structure below)
└── outputs/                         # Excel/text summaries are written here
```
VGGT (`facebook/VGGT-1B`) and the DeepGaze IIE weights download automatically on first run.

**Files somewhere else?** Copy `paths_local.example.py` to `paths_local.py` and set only the paths
that differ. `paths_local.py` is git-ignored, so every machine keeps its own and `git pull` never
overwrites it.

### 5. Run
```bash
uv run python final_form.py        # plain L2CS pipeline
uv run python final_form_ief.py    # IEF pipeline (per-fold + ensemble)

# training / evaluation scripts work the same way, e.g.
uv run python l2cs_net/train_refinement.py --help
```
`uv run` always uses this project's `.venv`, no activation needed. If you prefer activating it
(e.g. to select it as the interpreter in VS Code): `.venv\Scripts\activate` on Windows,
`source .venv/bin/activate` on Linux.

## Adding things later

| Task | Command |
|------|---------|
| Add a library | `uv add <package>` (updates `pyproject.toml` + `uv.lock`; commit both) |
| Remove a library | `uv remove <package>` |
| Get a teammate's changes | `git pull` then `uv sync` |
| Rebuild a broken environment | delete `.venv/` and run `uv sync` |
| Run the l2cs tests | `uv run --group dev pytest l2cs_net/tests/test_wrapper.py` |

Avoid `pip install` into `.venv` directly — uv won't record it, so other machines won't get it.

## Troubleshooting

- **`UnicodeEncodeError: 'cp932' codec can't encode …`** (Japanese Windows): some scripts print
  characters like `—`. Enable Python's UTF-8 mode once, then open a new terminal:
  `setx PYTHONUTF8 1`
- **`CUDA not available` in `check_env.py`**: update the NVIDIA driver (≥ 570).
- **Out of GPU memory**: VGGT (~5 GB of float32 weights, ~6 GB peak for 3 views) and DeepGaze IIE
  are loaded together, so cards with 8 GB or less may run out of memory.

## Dataset structure expected
```
dataset_root/
└── PersonName/
    └── scenario_N_Xm/
        ├── testing_Nscenario_left_Xm/
        │   ├── view1.jpg
        │   ├── view2.jpg
        │   └── *.jpg   (primary images)
        └── ...
```

## Key metrics
- `angular_error_peak_deg` — angle between heatmap peak's 3D world position (via VGGT) and GT centroid
- `angular_error_ief_deg` — raw IEF gaze direction error (no fusion)
- `cone_half_angle_deg` — width of gaze cone mask fed to DeepGaze
