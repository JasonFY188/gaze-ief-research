"""Brain server: runs the heavy models (VGGT, L2CS, DeepGaze IIE) on a GPU PC (e.g. the lab RTX 5090)
and answers the robot laptop over the network.

On the GPU PC (repo root, the env from docs/ACTIVE_CAMERA.md):
    python robot/brain_server.py                 # listens on port 8765
    python robot/brain_server.py --selftest      # no network: run plan + target on the saved 60-degree circle

The first time, Windows asks whether Python may use the network: allow it (Private network).
Find this PC's IP with `ipconfig` (IPv4 Address) and give it to the laptop: gaze_demo.py run --brain <IP>

Endpoints (JSON; images are base64 JPEG):
    GET  /ping                                    -> {"ok": true, "gpu": "..."}
    POST /faces   {image}                         -> {"faces": [[x1, y1, x2, y2, score], ...]}
    POST /plan    {room: [image...], person, K_room, K_person, hfov, vfov, room_pans, pan_limits} -> plan dict
    POST /target  {image, K}                      -> {"result": {...}, "overlay": image}
"""
import argparse
import base64
import json
import os
import sys
import threading
import time
# NOT ThreadingHTTPServer: cuDNN's autotuning cache is per thread (cudnn.benchmark is on), so a new thread per
# request re-tuned every model on every call (measured: RetinaFace 16 s per call in a new thread, 0.02 s on the
# same thread). One thread = the warm-up below is reused by every request.
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def dec(b64):
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)


def enc(img, q=90):
    return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1]).decode()


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


class Handler(BaseHTTPRequestHandler):
    brain = None
    lock = threading.Lock()          # one GPU job at a time

    def _send(self, code, obj):
        body = json.dumps(jsonable(obj)).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/ping":
            import torch
            return self._send(200, {"ok": True, "gpu": torch.cuda.get_device_name(0)})
        self._send(404, {"ok": False, "error": "unknown path"})

    def do_POST(self):
        t0 = time.time()
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            b = self.brain
            with self.lock:
                if self.path == "/faces":
                    out = {"faces": [list(f) for f in b.faces(dec(req["image"]))]}
                elif self.path == "/plan":
                    out = b.plan([dec(x) for x in req["room"]], dec(req["person"]),
                                 np.array(req["K_room"]), np.array(req["K_person"]), req["hfov"], req["vfov"],
                                 room_pans=req.get("room_pans"), pan_limits=req.get("pan_limits"))
                elif self.path == "/target":
                    res, over = b.target(dec(req["image"]), np.array(req["K"]))
                    out = {"result": res, "overlay": enc(over)}
                else:
                    return self._send(404, {"ok": False, "error": "unknown path"})
            out["ok"] = True
            if self.path != "/faces":
                print(f"{self.path} from {self.client_address[0]}: {time.time() - t0:.1f} s", flush=True)
            self._send(200, out)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})

    def log_message(self, *a):        # keep the console quiet (faces is called several times a second)
        pass


def selftest(brain):
    import glob
    import math
    room = os.path.join(HERE, "data", "captures", "20261116_1857_Room_60_degree")
    person = os.path.join(HERE, "data", "scenes", "Person1", "scenario_1_2m", "testing_1scenario_2m", "person_01.jpg")
    paths = sorted(glob.glob(os.path.join(room, "*.jpg")))
    from PIL import Image, ImageOps
    w, h = ImageOps.exif_transpose(Image.open(paths[0])).size
    fx = (min(w, h) / 2) / math.tan(math.radians(55.9 / 2))
    K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1.0]])
    hfov, vfov = math.degrees(2 * math.atan(w / 2 / fx)), math.degrees(2 * math.atan(h / 2 / fx))
    plan = brain.plan(paths, person, K, K, hfov, vfov)
    print("plan:", json.dumps(jsonable(plan), indent=1))
    # The best direction is a broad plateau: on the lab map -124/-10 sees 94% and -141/-16 sees 91%;
    # on the brain's map -124/-10 sees 89% and -142/-16 sees 90%. So check the plateau, not one exact angle.
    ok = -150 <= plan["pan_vggt"] <= -115 and plan["captured"] >= 0.85
    print("lab results: pan -124 to -145, tilt -10 to -16, captured 90-94% (broad peak)")
    print("SELFTEST PLAN:", "PASS" if ok else "FAIL - check gaze_brain.py")
    res, _ = brain.target(paths[4], K, out_dir=os.path.join(HERE, "outputs_robot", "selftest"))
    print("target:", res, "-> outputs_robot/selftest/target_overlay.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--warmup", default="1280x720",
                    help="robot camera image size to warm the GPU up for (first call per size is 30-60 s); 'none' to skip")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    from robot.gaze_brain import Brain
    Handler.brain = Brain()
    if args.selftest:
        return selftest(Handler.brain)
    if args.warmup != "none":
        ww, hh = map(int, args.warmup.lower().split("x"))
        Handler.brain.warmup(ww, hh)
        s = min(1.0, 640 / max(ww, hh))           # brain_client.faces() sends frames shrunk to 640 px
        Handler.brain.faces(np.full((int(round(hh * s)), int(round(ww * s)), 3), 127, np.uint8))
    srv = HTTPServer(("0.0.0.0", args.port), Handler)   # single thread, same as the warm-up (see import note)
    print(f"brain server ready on port {args.port}. Give the laptop this PC's IPv4 address (ipconfig).", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
