r"""
fov_check_vggt.py - is VGGT's room map angularly right, or only its focal-length guess wrong?

For each room photo we compare three field-of-view numbers (short side of the portrait photo):
  1. VGGT's camera head (its focal length guess)
  2. the angle VGGT's own 3D points span, seen from that camera (left image edge to right image edge)
  3. the measured value from fov_from_rotation.py (outputs_fov/<circle>.json)
If 2 ~ 3 the map's angles are right and only the focal guess is off (use the measured FOV for the camera).
If 2 ~ 1 the whole map is stretched sideways and must be rebuilt with the measured focal length.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from active_camera_steps import NET_H, NET_W, load  # noqa: E402

circle = sys.argv[1]
paths = sorted(glob.glob(os.path.join(circle, "*.jpg")))
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402

model = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda").eval()
images, _ = load(paths)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    pred = model(images.cuda())
extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
extr, intr = extr[0].float().cpu().numpy(), intr[0].float().cpu().numpy()
wp = pred["world_points"][0].float().cpu().numpy()
conf = pred["world_points_conf"][0].float().cpu().numpy()
depth = pred["depth"][0, ..., 0].float().cpu().numpy()
meas = json.load(open(os.path.join(HERE, "outputs_fov", os.path.basename(circle.rstrip("/\\")) + ".json")))

rows = []
for v in range(len(paths)):
    R, t = extr[v][:3, :3], extr[v][:3, 3]
    head = np.degrees(2 * np.arctan(NET_W / 2 / intr[v][0, 0]))
    spans = []
    for name, X in (("point head", wp[v]), ("depth head", None)):
        if X is None:      # depth map unprojected with VGGT's own K -> span = head by construction, skip
            continue
        Pc = (X.reshape(-1, 3) @ R.T + t).reshape(NET_H, NET_W, 3)
        c = conf[v]
        band = slice(NET_H // 3, 2 * NET_H // 3)            # middle third of the rows, near the horizon
        def col_angle(cols):
            P = Pc[band, cols].reshape(-1, 3); cc = c[band, cols].reshape(-1)
            P = P[cc >= np.median(cc)]
            return np.degrees(np.arctan2(P[:, 0], P[:, 2]))
        left, right = np.median(col_angle(slice(0, 6))), np.median(col_angle(slice(NET_W - 6, NET_W)))
        spans.append(right - left + 2 * 3 * (right - left) / NET_W)   # edge columns are 3 px in from the border
    rows.append((os.path.basename(paths[v]), head, spans[0]))
    print(f"{os.path.basename(paths[v]):26s} VGGT focal guess {head:5.1f}   VGGT points span {spans[0]:5.1f}")
print(f"\nmedian: VGGT focal guess {np.median([r[1] for r in rows]):.1f} deg, VGGT points span "
      f"{np.median([r[2] for r in rows]):.1f} deg, measured (rotation) {meas['short_fov_deg']:.1f} deg, "
      f"EXIF of the iPhone 15 Pro main camera (24 mm equiv., 4:3) 56.8 deg")
