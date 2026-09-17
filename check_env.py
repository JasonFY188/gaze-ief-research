"""
Sanity check for a new machine: run `uv run python check_env.py`.

Verifies that every library imports, the GPU is usable, and the files that
paths.py points to exist. Nothing is downloaded or modified.
"""
import importlib
import os
import sys

ok = True


def report(label, good, detail=""):
    global ok
    ok &= good
    print(f"  [{'OK' if good else '!!'}] {label}" + (f"  - {detail}" if detail else ""))


print(f"Python {sys.version.split()[0]}  ({sys.executable})")

print("\nLibraries")
for mod in ["torch", "torchvision", "numpy", "cv2", "PIL", "scipy", "matplotlib",
            "pandas", "openpyxl", "tqdm", "face_detection", "deepgaze_pytorch", "vggt", "l2cs"]:
    try:
        m = importlib.import_module(mod)
        report(mod, True, getattr(m, "__version__", ""))
    except Exception as e:  # noqa: BLE001
        report(mod, False, f"{type(e).__name__}: {e}")

try:
    import l2cs
    report("l2cs comes from this repo", "l2cs_net" in os.path.abspath(l2cs.__file__), l2cs.__file__)
except Exception:  # noqa: BLE001
    pass

print("\nGPU")
try:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        (torch.ones(2, device="cuda") * 2).sum().item()  # fails if the CUDA build doesn't support this GPU
        report("CUDA", True, f"{name}, torch CUDA {torch.version.cuda}")
    else:
        report("CUDA", False, "not available - pipeline will run on CPU (very slow)")
except Exception as e:  # noqa: BLE001
    report("CUDA", False, f"{type(e).__name__}: {e}")

print("\nFiles from paths.py (missing files only matter for the script that needs them)")
import paths  # noqa: E402

for name in ["BASE_ROOT", "L2CS_WEIGHTS", "GAZE360_BACKBONE", "MPIIGAZE_BACKBONE_DIR", "CENTERBIAS_NPY"]:
    p = getattr(paths, name)
    exists = os.path.exists(p)
    print(f"  [{'OK' if exists else '--'}] {name} = {p}")
ckpt_dir = os.path.join(paths.IEF_REPO, "checkpoints")
n_ckpt = sum(len(fs) for _, _, fs in os.walk(ckpt_dir)) if os.path.isdir(ckpt_dir) else 0
print(f"  [{'OK' if n_ckpt else '--'}] IEF checkpoints in {ckpt_dir}: {n_ckpt} file(s)")

print("\n" + ("Environment looks good." if ok else "Some checks failed (see [!!] above)."))
sys.exit(0 if ok else 1)
