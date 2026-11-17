# Handoff to the robot PC (PiPER + RealSense)

To use this note: on the robot PC, run `git checkout active-camera-piper`, then tell Claude:
"Read docs/HANDOFF_ROBOT_PC.md and docs/ACTIVE_CAMERA.md and help me connect this to the robot."

## What I want to do

I want to use this system to **change the camera's point of view so it can see what a person is looking at**.

- A camera (the RealSense on this PC) sees a person.
- L2CS estimates their gaze direction with an uncertainty cone (95%).
- With a 3D map of the room (VGGT), the code finds where that cone lands on real surfaces: the
  **gaze footprint**.
- `FastSearch` (in `active_camera.py`) finds the camera direction whose field of view covers most of the
  footprint.
- The robot (PiPER arm with the camera on it) moves the camera to that view and takes the picture.
- That picture goes into `final_form.py` (gaze cone × DeepGaze IIE saliency) to find the looked-at target.

## Already done (lab PC, tested)

- The footprint and pan/tilt search work on a real room: one person photo and iPhone Turn Capture circles.
  The chosen direction is consistent across circles (pan −124° to −145°, tilt −10° to −15°). It sees 77–94%
  of the footprint, against about 10% for a random photo.
- Speed: about 30 ms per decision after the map is built (VGGT about 1 s, L2CS about 0.7 s).
- The camera FOV must be **measured**: VGGT's focal guess was 70°, the real value 56° for the iPhone.
  For the RealSense, use the SDK intrinsics of the colour stream at the resolution in use
  (snippet in ACTIVE_CAMERA.md).

## To do on the robot PC (this PC knows the robot setup; decide here)

1. Install the environment (ACTIVE_CAMERA.md, "Set up a new computer") and run `python active_camera_demo.py`.
2. Read the RealSense colour intrinsics → hfov/vfov for `FastSearch.best(hfov, vfov, ...)`.
3. Mount the RealSense on the PiPER, then do the hand-eye calibration (camera ↔ flange).
4. Express the camera pose in the room map:
   - the robot base in the map;
   - metric scale (VGGT units are not metres; RealSense depth can give the scale).
5. Turn the chosen view into an arm pose (IK), and give `FastSearch.best` a `center_fn(pan, tilt)`.
   On an arm the camera centre moves when it turns.
6. Loop:

   ```
   RealSense frame with the person -> L2CS -> footprint -> FastSearch -> move arm -> capture -> final_form.py
   ```

7. Check: does the chosen view find the true looked-at object more often than a random or fixed view?

## Rules from my research setup

- Explain in simple language: conclusion → reason → example → next step.
- Be honest about results and look for weaknesses.
- Never commit weights, datasets, person photos or certificates (`.gitignore` already blocks them).
- Slides and speaker notes are in Japanese; code and docs can be in English.
