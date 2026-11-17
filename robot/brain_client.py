"""Laptop side of robot/brain_server.py: same calls as gaze_brain.Brain, but the work runs on the GPU PC.

    from robot.brain_client import RemoteBrain
    brain = RemoteBrain("192.168.1.23")      # IP of the GPU PC
    brain.faces(bgr); brain.plan(...); brain.target(...)
"""
import base64
import json
import urllib.request

import cv2
import numpy as np


def enc(img, q=90, max_side=None):
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1]).decode()


def imread(p):
    return p if isinstance(p, np.ndarray) else cv2.imread(p, cv2.IMREAD_COLOR)


class RemoteBrain:
    def __init__(self, host, port=8765, timeout=300):
        self.url, self.timeout = f"http://{host}:{port}", timeout
        with urllib.request.urlopen(self.url + "/ping", timeout=10) as r:
            info = json.loads(r.read())
        print(f"[brain] connected to {host}:{port} ({info['gpu']})")

    def _post(self, path, obj):
        req = urllib.request.Request(self.url + path, json.dumps(obj).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.loads(r.read())
        except urllib.error.HTTPError as e:
            out = json.loads(e.read())
        if not out.pop("ok", False):
            raise RuntimeError(f"brain server: {out.get('error')}")
        return out

    def faces(self, bgr):
        # send a small copy for speed, scale the boxes back
        s = min(1.0, 640 / max(bgr.shape[:2]))
        small = cv2.resize(bgr, None, fx=s, fy=s) if s < 1 else bgr
        fs = self._post("/faces", {"image": enc(small, 80)})["faces"]
        return [(x1 / s, y1 / s, x2 / s, y2 / s, sc) for x1, y1, x2, y2, sc in fs]

    def plan(self, room, person, K_room, K_person, hfov, vfov, room_pans=None, pan_limits=None):
        return self._post("/plan", {
            "room": [enc(imread(p)) for p in room], "person": enc(imread(person)),
            "K_room": np.asarray(K_room).tolist(), "K_person": np.asarray(K_person).tolist(),
            "hfov": hfov, "vfov": vfov, "room_pans": None if room_pans is None else list(room_pans),
            "pan_limits": None if pan_limits is None else list(pan_limits)})

    def plan_map(self, room_dir, K_room, person, K_person, hfov, vfov, pan0=0.0, tilt0=0.0, pan_limits=None,
                 anchor=None, K_anchor=None, arm=None, viz=False):
        """room_dir / anchor: paths RELATIVE TO THE REPO; the brain PC reads them from its own copy of data/.
        viz=True also returns the 3D points, footprint and search grid (for robot/report.py)."""
        out = self._post("/plan_map", {
            "viz": viz,
            "room_dir": room_dir.replace("\\", "/"), "anchor": anchor.replace("\\", "/") if anchor else None,
            "K_room": np.asarray(K_room).tolist(), "K_anchor": None if K_anchor is None else np.asarray(K_anchor).tolist(),
            "person": enc(imread(person)), "K_person": np.asarray(K_person).tolist(), "hfov": hfov, "vfov": vfov,
            "pan0": pan0, "tilt0": tilt0, "pan_limits": None if pan_limits is None else list(pan_limits), "arm": arm})
        if out.get("viz"):
            out["viz"] = {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in out["viz"].items()}
        return out

    def target(self, photo, K, out_dir=None, viz=False):
        out = self._post("/target", {"image": enc(imread(photo)), "K": np.asarray(K).tolist(), "viz": viz})
        if out["result"].get("viz"):
            out["result"]["viz"] = {k: np.asarray(v) for k, v in out["result"]["viz"].items()}
        over = cv2.imdecode(np.frombuffer(base64.b64decode(out["overlay"]), np.uint8), cv2.IMREAD_COLOR)
        if out_dir:
            import os
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, "target_overlay.jpg"), over)
        return out["result"], over
