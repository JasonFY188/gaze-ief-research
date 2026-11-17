"""RealSense colour camera on Windows (native USB3, no usbipd) + its real FOV from the factory calibration.

    python robot/realsense_cam.py          # prints intrinsics + hfov/vfov, writes robot/realsense_intrinsics.json
"""
import json
import math
import os

import numpy as np
import pyrealsense2 as rs

W, H, FPS = 1280, 720, 30
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realsense_intrinsics.json")


class RealSense:
    def __init__(self, width=W, height=H, fps=FPS):
        self.pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        prof = self.pipe.start(cfg)
        i = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intr = {"width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy, "ppx": i.ppx, "ppy": i.ppy,
                     "model": str(i.model), "coeffs": list(i.coeffs),
                     "hfov": math.degrees(2 * math.atan(i.width / 2 / i.fx)),
                     "vfov": math.degrees(2 * math.atan(i.height / 2 / i.fy))}
        for _ in range(10):  # let auto-exposure settle
            self.pipe.wait_for_frames()

    @property
    def hfov(self): return self.intr["hfov"]

    @property
    def vfov(self): return self.intr["vfov"]

    def read(self) -> np.ndarray:
        """BGR uint8 image."""
        return np.asanyarray(self.pipe.wait_for_frames().get_color_frame().get_data()).copy()

    def stop(self):
        self.pipe.stop()


if __name__ == "__main__":
    cam = RealSense()
    print(json.dumps(cam.intr, indent=2))
    with open(OUT, "w") as f:
        json.dump(cam.intr, f, indent=2)
    print(f"hfov={cam.hfov:.2f} deg  vfov={cam.vfov:.2f} deg  -> saved {OUT}")
    cam.stop()
