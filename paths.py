"""
Machine-specific path configuration for final_form_ief.py.

Edit ONLY this file when moving to a new machine.
All other scripts import from here.
"""
import os

# ----- PACKAGE ROOT (optional) ------------------------------------------------
# If you unpack the USB bundle, set this to wherever you put the package root.
# Leave as None to use the individual paths below (default for the lab machine).
PACKAGE_ROOT = os.environ.get("IEF_PACKAGE_ROOT", None)

def _p(*parts):
    """Resolve a path, optionally rooted under PACKAGE_ROOT."""
    if PACKAGE_ROOT and not os.path.isabs(parts[0]):
        return os.path.join(PACKAGE_ROOT, *parts)
    return os.path.join(*parts)

# Every path can be set with an environment variable (see .env.example); the defaults are placeholders.

# ----- DATASET ----------------------------------------------------------------
BASE_ROOT = _p(os.environ.get("GAZE_DATASET_ROOT", "data/scenes"))

# ----- L2CS library (IEF repo) ------------------------------------------------
IEF_REPO = _p(os.environ.get("IEF_REPO", os.path.join(os.path.dirname(__file__), "l2cs_net")))

# ----- Plain L2CS weights — used ONLY for face detection ----------------------
L2CS_WEIGHTS = _p(os.environ.get("L2CS_WEIGHTS", "models/L2CSNet_gaze360.pkl"))

# ----- IEF backbone weights ---------------------------------------------------
GAZE360_BACKBONE = _p(os.environ.get("GAZE360_BACKBONE", "models/Gaze360/L2CSNet_gaze360.pkl"))
MPIIGAZE_BACKBONE_DIR = _p(os.environ.get("MPIIGAZE_BACKBONE_DIR", "models/MPIIGaze"))

# ----- centerbias (relative to gaze-estimation/ dir is fine) -----------------
CENTERBIAS_NPY = _p(os.path.dirname(__file__), "centerbias_mit1003.npy")

# ----- VGGT HuggingFace model ID or local safetensors path -------------------
# Keep as-is to download from HF, or set to a local path:
#   VGGT_SOURCE = "/path/to/package/models/vggt_1b.safetensors"
VGGT_SOURCE = "facebook/VGGT-1B"

# ----- DeepGaze IIE -- torch hub cache ---------------------------------------
# Set to a local .pth path to skip downloading:
#   DEEPGAZE_CACHE = "/path/to/package/models/deepgaze2e.pth"
# Leave None to use the default ~/.cache/torch/hub location.
DEEPGAZE_CACHE = None

# ----- Output summary directory -----------------------------------------------
SUMMARY_OUT_DIR = os.environ.get("GAZE_OUTPUT_DIR", "outputs_ief")
