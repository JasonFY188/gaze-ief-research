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

# ----- DATASET ----------------------------------------------------------------
BASE_ROOT = _p("/home/keisokulab/Downloads/Jikken_2 (3rd copy)")

# ----- L2CS library (IEF repo) ------------------------------------------------
IEF_REPO = _p("/home/keisokulab/Downloads/Dealing with uncertainty from l2cs net/L2CS-Net")

# ----- Plain L2CS weights — used ONLY for face detection ----------------------
L2CS_WEIGHTS = _p("/home/keisokulab/learning_workshop/L2CSNet_gaze360.pkl")

# ----- IEF backbone weights ---------------------------------------------------
GAZE360_BACKBONE = _p(
    "/home/keisokulab/Downloads/Dealing with uncertainty from l2cs net"
    "/Gaze360-20260603T102045Z-3-001/Gaze360/L2CSNet_gaze360.pkl"
)
MPIIGAZE_BACKBONE_DIR = _p(
    "/home/keisokulab/Downloads/Dealing with uncertainty from l2cs net"
    "/MPIIGaze-20260529T070835Z-3-001"
)

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
SUMMARY_OUT_DIR = os.path.expanduser("~/Documents/output of ief")
