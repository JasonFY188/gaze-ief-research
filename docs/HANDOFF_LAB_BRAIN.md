# Handoff: lab PC (RTX 5090) = brain server for the robot

To use this note: on the lab PC, open this repo, run `git checkout active-camera-piper` and `git pull`, then tell Claude:
"Read docs/HANDOFF_LAB_BRAIN.md and set this PC up as the brain server for the robot laptop."

## The idea

The robot laptop (RTX 4060 Laptop GPU, 8 GB, 16 GB RAM) runs the PiPER arm and the RealSense, but it is too small for
VGGT-1B: loading the model ran out of Windows memory ("The paging file is too small", random exit code 139 crashes).
So the heavy models run **here**, and the laptop sends photos over the network:

```
LAPTOP (robot)                                         THIS PC (RTX 5090)
robot/gaze_demo.py run --brain <this PC's IP>          robot/brain_server.py  (port 8765)
  RealSense frame  --- /faces  (small JPEG) -------->    RetinaFace (L2CS detector)
  6 room photos + person photo --- /plan ----------->    VGGT + L2CS + gaze footprint + FastSearch
                   <-- robot pan/tilt to turn to ----
  photo after turning --- /target ------------------>    VGGT again + gaze cone x DeepGaze IIE (final_form.py)
                   <-- heat map overlay + peak pixel -
```

Main pipeline only (`final_form.py`). **No IEF.**

## Files (all in `robot/`)

| File | Runs on | What it does |
|---|---|---|
| `gaze_brain.py` | this PC | The method split into calls: `faces()`, `plan()` (where to turn), `target()` (where the person looks). Same constants as `active_camera_steps.py`. Loads VGGT straight to the GPU tensor by tensor (`load_vggt`), to avoid the RAM spike of `from_pretrained` |
| `brain_server.py` | this PC | HTTP server (Python stdlib, port 8765) around `gaze_brain.Brain`. `--selftest` runs without the network |
| `brain_client.py` | laptop | `RemoteBrain(ip)`: same calls, over the network |
| `gaze_demo.py` | laptop | The sequence: find the person, face them, SPACE + 5 s countdown, photo, plan, turn, photo, target |
| `arm_server.py`, `arm_client.py`, `jog.py`, `realsense_cam.py` | laptop | PiPER arm over CAN (WSL) and the RealSense (Windows) |

## What to do on this PC

1. **Environment.** Use the existing env from `docs/ACTIVE_CAMERA.md` (VGGT, L2CS, DeepGaze IIE, opencv). The brain needs
   nothing new. Check that `models/L2CSNet_gaze360.pkl` and `centerbias_mit1003.npy` are in place.
2. **Self-test (no robot, no network):**
   ```
   python robot/brain_server.py --selftest
   ```
   It runs `plan()` + `target()` on `data/captures/20261116_1857_Room_60_degree` and the Person1 photo.
   Expected: **pan about -124, tilt about -10, captured about 94%** (same as `active_camera_steps.py` on this circle).
   Overlay: `outputs_robot/selftest/target_overlay.jpg`. **This code has NOT run with the real models yet**:
   the laptop ran out of memory before it got there. If it fails or the numbers differ, fix `gaze_brain.py` first.
3. **Start the server:**
   ```
   python robot/brain_server.py
   ```
   Windows asks whether Python may use the network: allow **Private networks**. If there is no prompt and the laptop
   cannot connect, add a rule (admin PowerShell):
   `New-NetFirewallRule -DisplayName "gaze brain 8765" -Direction Inbound -Protocol TCP -LocalPort 8765 -Action Allow`
4. **Give the laptop the address:** `ipconfig` → IPv4 Address of the adapter on the same network as the laptop.
5. **Check from the laptop:** open `http://<IP>:8765/ping` in the laptop's browser. It should show the 5090.
   University Wi-Fi often blocks PC-to-PC traffic (port 8443 was already blocked for Turn Capture). If it is blocked,
   use a LAN cable between the two PCs, a phone hotspot, or a small router.

## What the laptop does (for reference)

```
# WSL (Ubuntu-22.04): arm server, after `usbipd attach --wsl --busid 1-6` and piper_can_up.sh
python3 /mnt/c/Users/ASUS/Desktop/piper_gaze_testing/gaze-ief-research/robot/arm_server.py
# Windows, gazecam env
python robot/jog.py                                   # once: camera level + straight ahead, Shift+H = home
python robot/gaze_demo.py run --brain <lab PC IP>     # first run also makes the robot's 6-photo room map
```

## Status and known limits (be honest about these)

- Laptop side tested: arm server ↔ Windows over TCP; brain client ↔ server with a fake brain (images, intrinsics,
  orientation, errors). The first robot movement test worked (`piper_wiggle.py`).
- **Not tested yet:** `gaze_brain.py` with the real models; the live demo; the home pose (not set yet).
- The arm is treated as a pan/tilt head: pan = J1, tilt = J5, around a saved home pose. No hand-eye calibration and no
  `center_fn` yet, so the camera centre moving on the arm is ignored (expect a few degrees of aim error).
- The robot needs its **own** room map from where it stands: the iPhone circles are of another spot. `gaze_demo.py scan`
  takes 6 photos at pan -150, -90, -30, 30, 90, 150 (J1 is limited to ±150°; the RealSense hfov 70.1° covers the gaps).
- RealSense colour 1280×720: hfov 70.1°, vfov 43.2° (factory calibration, `robot/realsense_cam.py`).
- `plan()` prints `room_pan_residual_deg`: VGGT's angle of each room photo minus the robot's angle. Large values mean the
  VGGT frame and the robot frame disagree (wrong pan sign, bad map): check this first if the robot turns the wrong way.
- VGGT units are not metres; not needed yet for pan/tilt only.

## Rules from my research setup

- Explain in simple language: conclusion → reason → example → next step.
- Be honest about results and look for weaknesses.
- Never commit weights, datasets, person photos or certificates (`.gitignore` blocks `data/`, `models/`, `output*/`).
- Slides and speaker notes are in Japanese; code and docs can be in English.
