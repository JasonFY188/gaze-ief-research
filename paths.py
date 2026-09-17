"""
Path configuration for final_form.py and final_form_ief.py.

Defaults point at folders inside this repo (see README "Where to put files"),
so a fresh clone works without editing anything.

To use different locations on a particular machine, copy
`paths_local.example.py` to `paths_local.py` and override only what you need.
`paths_local.py` is git-ignored, so each machine keeps its own copy.
"""
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _repo(*parts):
    return os.path.join(REPO_ROOT, *parts)


# ----- DATASET ----------------------------------------------------------------
BASE_ROOT = _repo("data", "experiment")

# ----- L2CS library (IEF repo) ------------------------------------------------
# Only used to locate IEF head checkpoints: <IEF_REPO>/checkpoints/ief_*/best_fold*.pt
# (the l2cs code itself is installed from l2cs_net/ into the environment).
IEF_REPO = _repo("l2cs_net")

# ----- Plain L2CS weights — used ONLY for face detection ----------------------
L2CS_WEIGHTS = _repo("weights", "L2CSNet_gaze360.pkl")

# ----- IEF backbone weights ---------------------------------------------------
GAZE360_BACKBONE = _repo("weights", "L2CSNet_gaze360.pkl")
MPIIGAZE_BACKBONE_DIR = _repo("weights", "MPIIGaze")  # contains fold0.pkl … fold14.pkl

# ----- DeepGaze centerbias ----------------------------------------------------
CENTERBIAS_NPY = _repo("weights", "centerbias_mit1003.npy")

# ----- VGGT HuggingFace model ID or local path --------------------------------
VGGT_SOURCE = "facebook/VGGT-1B"

# ----- DeepGaze IIE -- torch hub cache ----------------------------------------
# Leave None to use the default ~/.cache/torch/hub location.
DEEPGAZE_CACHE = None

# ----- Output summary directories ---------------------------------------------
SUMMARY_OUT_DIR = _repo("outputs", "summary_ief")      # final_form_ief.py
FINAL_FORM_SUMMARY_DIR = _repo("outputs", "summary")   # final_form.py

# ----- Machine-specific overrides ---------------------------------------------
try:
    from paths_local import *  # noqa: F401,F403
except ImportError:
    pass
