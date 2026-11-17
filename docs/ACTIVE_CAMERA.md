# Active camera: turn the camera so it sees where the person is looking

Branch `active-camera-piper`. Goal: put this on a robot (AgileX PiPER arm + camera) later.

**What it does.** From one photo of a person (L2CS gaze) and a few photos of the room (VGGT 3D map), it finds the
surfaces the person's gaze cone lands on (the gaze "footprint", 裾野). It then finds the pan/tilt that makes the
camera's field of view cover as much of that footprint as possible. The result is "turn to pan −139°,
tilt −12° → the camera sees 87% of the footprint", computed in about 30 ms.

```
person photo --L2CS--> gaze direction + 95% cone ─┐
room photos  --VGGT--> 3D points, camera poses ───┼─> gaze footprint (weights on 3D points)
camera FOV (MEASURED, not VGGT's guess) ──────────┴─> FastSearch: best pan/tilt -> turn the camera / arm
```

## Files

| File | What it is |
|---|---|
| `active_camera.py` | The library: gaze footprint, visibility (angular z-buffer), `FastSearch` (pan/tilt search), zoom choice |
| `active_camera_steps.py` | Real room, one figure per step (inputs → VGGT → L2CS → footprint → search → check with real photos) |
| `active_camera_demo.py` | Synthetic room test, no models needed. Run this first on a new machine |
| `active_camera_room.py` | Older single-run version of the steps script |
| `fov_from_rotation.py` | **Measures a camera's real FOV** from photos taken while turning on one spot |
| `fov_check_vggt.py` | Checks whether VGGT's 3D map has the right angles (it does; only its focal guess is off) |
| `view_3d.py`, `view_open3d.py` | 3D view of a result (browser / Open3D window) |
| `final_form.py` | Main gaze pipeline (L2CS cone × DeepGaze IIE × VGGT), now with the ellipse cone |
| `turn_capture/` | iPhone web page that takes photos at exact gyroscope angles (20/30/60°) for the room map |
| `docs/IDEAS_TODO.md` | Parked ideas |

## Set up a new computer

```bash
git clone https://github.com/JasonFY188/gaze-ief-research.git
cd gaze-ief-research
git checkout active-camera-piper

python -m venv .venv
# Windows: .venv\Scripts\activate      Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # pick the line for your GPU on pytorch.org
pip install -r requirements-active-camera.txt
pip install --no-deps "vggt @ git+https://github.com/facebookresearch/vggt@a288dd0f14786c93483e45524328726ab7b1b4ce"
pip install -e l2cs_net
python active_camera_demo.py        # smoke test -> outputs_active/demo.png
```

**Not in git; copy these from the lab PC by USB or cloud** (licences and size; `.gitignore` blocks them):

| File | Where it goes | Where it comes from |
|---|---|---|
| `L2CSNet_gaze360.pkl` | `models/L2CSNet_gaze360.pkl` | lab PC `sotsuron/models/`, originally the L2CS-Net repo's pretrained models (https://github.com/Ahmednull/L2CS-Net) |
| `centerbias_mit1003.npy` | repo root | lab PC `sotsuron/`, originally from DeepGaze (https://github.com/matthias-k/DeepGaze) |
| person photos | `data/scenes/<Person>/<scenario>/...` | your own photos |
| room circles | `data/captures/<session>/` (`*.jpg` + `poses.jsonl`) | Turn Capture |

VGGT-1B (~5 GB) and DeepGaze IIE weights download automatically on first use.
A GPU with about 10 GB or more is comfortable for VGGT with about 20 images.

## Use it

```bash
# 1. measure the camera's real field of view (needs a circle with 20-30° steps so photos overlap)
python fov_from_rotation.py data/captures/<session_20deg>

# 2. the whole chain, one figure per step, with the measured short-side FOV
python active_camera_steps.py --fov-short 55.9 \
    --person data/scenes/Person1/scenario_1_2m/testing_1scenario_2m/person_01.jpg \
    --rooms data/captures/<session_20deg> data/captures/<session_30deg>

# 3. look at the result in 3D
python view_3d.py outputs_active/steps/<session_20deg>        # writes view3d.html
python view_open3d.py outputs_active/steps/<session_20deg>    # Open3D window
```

In your own code:

```python
import active_camera as A
fs = A.FastSearch(P, w, cam_center, ref_forward, up, area=area)        # occlusion is computed once here
best = fs.best(hfov_deg, vfov_deg, tilt_range=(-45, 45), aim_sigma_deg=3.0)
best["pan"], best["tilt"], best["mass"]     # direction and share of the footprint it sees
```

## Getting the field of view right (important)

- **Never use VGGT's focal-length guess as the camera's FOV.** For the iPhone 15 Pro browser photos, VGGT
  says about 70° (short side), but the real value is about 56°. That was measured with `fov_from_rotation.py`
  (55.3° and 56.5° on two circles), and the EXIF of the same phone (24 mm equivalent) gives 56.8°. VGGT's 3D
  points themselves span about 53°, so the map is fine; only the camera model must be replaced.
  With 70°, the real photos looked as if they saw 95% of the footprint; with the true 56° they see 58%.
- **Phone:** `fov_from_rotation.py` (no checkerboard, no EXIF). Method: for pure rotation, overlapping photos are
  related by H = K R K⁻¹, so we search the focal length that makes K⁻¹HK a rotation
  (Hartley, *Self-calibration of stationary cameras*, IJCV 1997).
- **RealSense:** read the factory calibration for the exact stream and resolution you use. The FOV changes with
  resolution and aspect ratio, and the colour and depth cameras differ. Use the colour stream, because that
  image goes to DeepGaze.

```python
import math, pyrealsense2 as rs
pipe, cfg = rs.pipeline(), rs.config()
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
prof = pipe.start(cfg)
i = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
print(i.width, i.height, i.fx, i.fy, i.ppx, i.ppy, i.model, i.coeffs)
print("hfov", math.degrees(2 * math.atan(i.width / 2 / i.fx)), "vfov", math.degrees(2 * math.atan(i.height / 2 / i.fy)))
pipe.stop()
```

## Speed (lab PC, RTX 5090)

| Step | Time |
|---|---|
| VGGT load (once) | 8.9 s |
| VGGT, 18 room photos + 1 person photo | 0.95 s |
| L2CS on a 24 MP photo | 0.74 s |
| FastSearch, occlusion + about 1,700 directions | 17–31 ms |
| Old search (z-buffer per direction) | 175–366 ms |

**Why FastSearch is fast:** when the camera only rotates about its optical centre, what is hidden behind what
does not change. So occlusion is computed once, and each direction only needs an "inside the frame?" test.
It picks a view within 0.2–0.8 points of the old search when both are scored with the full z-buffer.

## Plan for the PiPER (to do)

1. **Camera mount** on the PiPER flange (design from photos of the flange and the camera).
2. **PiPER SDK** (AgileX, https://github.com/agilexrobotics/piper_sdk; read on 2026-11-17):
   - Install with `pip install piper_sdk python-can`. Linux only for the CAN tools, Ubuntu 18.04/20.04/22.04.
   - Bring up CAN with `piper-can activate can0 1000000`.
   - Connect with `C_PiperInterface_V3("can0")` and `.ConnectPort()`; read joints with `GetArmJointMsgs()`.
   - V3 is for firmware 1.8-8 and later. The end-pose and motion commands are in `assets/V3/INTERFACE_V3.MD` in that repo.
3. **Hand-eye calibration:** camera pose relative to the flange (`T_flange_cam`), for example with
   OpenCV `cv2.calibrateHandEye` and a checkerboard.
4. **Robot base in the room map (`T_map_base`).** Find where the camera is in the VGGT map, for example
   with one photo matched to the map photos plus PnP, then remove the camera-to-base part.
5. **Metric scale:** VGGT units are NOT metres. Get the scale before using arm distances, for example from
   RealSense depth or one measured distance.
6. **Arm offset:** the camera does not turn about its optical centre on an arm. Give `FastSearch.best` a
   `center_fn(pan, tilt)` that returns the camera centre for that pose (inverse kinematics →
   `T_map_base @ FK(q) @ T_flange_cam`); the best candidates are then re-scored exactly.
7. **Loop:** person photo → L2CS → footprint → FastSearch → arm IK → move → take the photo → `final_form.py`.

## Known limits

- Tested on one room and one person photo so far.
- The absolute "% seen" depends on how occlusion is approximated (fast and old methods differ by 3–5 points),
  although the chosen direction agrees.
- VGGT's `load_and_preprocess_images` "crop" mode cuts portrait photos, while `final_form.py` maps the face box
  as if nothing was cut. The active-camera scripts resize to 392×518 without cropping instead.
- Turn Capture: the iPhone must trust the certificate made by `turn_capture/make_certs.sh <PC IP>`.
  Some university networks block port 8443; port 8000 worked.
