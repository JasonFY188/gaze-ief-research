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
| `paths.py` | **Edit this** to set paths for your machine |
| `blob_projection_utils.py` | Projects 3D gaze blobs into reference views |
| `l2cs_net/l2cs/` | L2CS-Net library with IEF extensions |
| `l2cs_net/train_refinement.py` | Train IEF head on Gaze360 backbone |
| `l2cs_net/train_ief_mpiigaze.py` | Train IEF head on MPIIGaze backbone |

## Setup

### 1. Conda environment
```bash
conda create -n vggt python=3.10
conda activate vggt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt   # or use environment.yml if provided
```

### 2. Install L2CS-Net with IEF
```bash
pip install -e l2cs_net/
```

### 3. Download model weights
Place these in the locations set in `paths.py`:

| Weight | Source |
|--------|--------|
| `L2CSNet_gaze360.pkl` | [L2CS-Net releases](https://github.com/Ahmednull/L2CS-Net) |
| `fold0.pkl … fold14.pkl` (MPIIGaze backbones) | Train with `l2cs_net/train.py` on MPIIGaze |
| `checkpoints/ief_gaze360/best_fold*.pt` | Train with `l2cs_net/train_refinement.py` |
| `checkpoints/ief_mpiigaze/best_fold*.pt` | Train with `l2cs_net/train_ief_mpiigaze.py` |
| VGGT (`facebook/VGGT-1B`) | Auto-downloaded from HuggingFace |
| DeepGaze IIE | Auto-downloaded from torch hub |
| `centerbias_mit1003.npy` | [MIT1003 centerbias](https://people.csail.mit.edu/tjudd/WherePeopleLook/) |

### 4. Edit paths
```python
# paths.py — set these for your machine
BASE_ROOT             = "/path/to/your/dataset"
IEF_REPO              = "/path/to/l2cs_net"
L2CS_WEIGHTS          = "/path/to/L2CSNet_gaze360.pkl"
GAZE360_BACKBONE      = "/path/to/L2CSNet_gaze360.pkl"
MPIIGAZE_BACKBONE_DIR = "/path/to/MPIIGaze/"
```

### 5. Run
```bash
# Plain L2CS pipeline
python final_form.py

# IEF pipeline (per-fold + ensemble)
python final_form_ief.py
```

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
