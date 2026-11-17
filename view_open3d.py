"""
view_open3d.py - open the active-camera result in an Open3D 3D window.

Layers (tick/untick them in the panel on the right of the window):
  room            VGGT points in the photos' real colours, gaze footprint tinted red
  footprint       only the gaze footprint (red, darker = more likely)
  eye, gaze ray, gaze cone (95%)
  camera spot, chosen view (blue pyramid = what the turned camera sees)
  real photos     the directions the Turn Capture photos were taken in
Mouse: left-drag rotate, wheel zoom, right-drag (or shift+left) pan.

Usage (from sotsuron):
  ..\\.venv\\Scripts\\python.exe view_open3d.py outputs_active\\steps\\20261116_1906_Room_20_degree
  add --check to only build the geometry without opening a window.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402

import active_camera as A  # noqa: E402


def lineset(segments, color):
    pts, lines = [], []
    for a, b in segments:
        lines.append([len(pts), len(pts) + 1])
        pts += [a, b]
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(np.array(pts)), o3d.utility.Vector2iVector(np.array(lines)))
    ls.paint_uniform_color(color)
    return ls


def sphere(center, r, color):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=r)
    s.translate(center)
    s.compute_vertex_normals()
    s.paint_uniform_color(color)
    return s


def frustum(center, R_w2c, hfov, vfov, length):
    th, tv = np.tan(np.radians(hfov / 2)), np.tan(np.radians(vfov / 2))
    cs = [center + length * (R_w2c.T @ np.array([sx * th, sy * tv, 1.0])) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return [(center, c) for c in cs] + [(cs[k], cs[(k + 1) % 4]) for k in range(4)]


def build(d):
    S = np.load(os.path.join(d, "scene.npz"))
    P, rgb, w = S["P"].astype(np.float64), S["rgb"].astype(np.float64) / 255.0, S["w"]
    eye, gaze, half, cam, up, ref = S["eye"], S["gaze"], float(S["half"]), S["cam"], S["up"], S["ref"]
    hfov, vfov = float(S["hfov"]), float(S["vfov"])
    L = float(np.percentile(np.linalg.norm(P - cam, axis=1), 90))

    a = np.clip(w / max(w.max(), 1e-9), 0, 1)[:, None] * 0.85
    room = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    room.colors = o3d.utility.Vector3dVector((1 - a) * rgb + a * np.array([1.0, 0.05, 0.05]))
    f = w > 0.05
    foot = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[f]))
    t = (w[f] / w.max())[:, None]
    foot.colors = o3d.utility.Vector3dVector((1 - t) * np.array([1.0, 0.85, 0.8]) + t * np.array([0.7, 0.0, 0.0]))

    u = np.cross(gaze, up); u /= np.linalg.norm(u); v = np.cross(u, gaze)
    cone = []
    for k in range(16):
        ang = 2 * np.pi * k / 16
        dvec = np.cos(np.radians(half)) * gaze + np.sin(np.radians(half)) * (np.cos(ang) * u + np.sin(ang) * v)
        cone.append((eye, eye + 1.1 * L * dvec))
    R = A.rotation_looking_at(A.forward_from_pan_tilt(float(S["best_pan"]), float(S["best_tilt"]), ref, up), up)
    real = [(c, c + 0.25 * L * fw) for c, fw in zip(S["centers"], S["fwd"])]
    r = 0.015 * L
    geoms = [
        ("room", room, True),
        ("footprint", foot, False),
        ("eye", sphere(eye, r, [0, 0, 0]), True),
        ("gaze ray", lineset([(eye, eye + 1.1 * L * gaze)], [1, 0, 0]), True),
        (f"gaze cone +-{half:.0f} deg (95%)", lineset(cone, [1, 0.4, 0.4]), True),
        ("camera spot", sphere(cam, r, [0, 0, 1]), True),
        (f"chosen view pan {float(S['best_pan']):+.0f} deg, tilt {float(S['best_tilt']):+.0f} deg",
         lineset(frustum(cam, R, hfov, vfov, 0.8 * L), [0, 0.2, 1]), True),
        ("real photo directions", lineset(real, [0.2, 0.6, 1]), False),
        ("person-photo camera", lineset([(S["person_center"], S["person_center"] + 0.3 * L * S["person_fwd"])], [0.6, 0, 0.8]), False),
    ]
    view = dict(lookat=cam, eye=cam - 1.2 * L * ref + 0.9 * L * up, up=up)
    return geoms, view


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    d = args[0] if args else os.path.join(HERE, "outputs_active", "steps", "20261116_1906_Room_20_degree")
    geoms, view = build(d)
    if "--check" in sys.argv:
        for name, g, on in geoms:
            print(f"{name:40s} {type(g).__name__:14s} {'on' if on else 'off'}")
        return
    try:   # new viewer with a layer panel
        o3d.visualization.draw([{"name": n, "geometry": g, "is_visible": on} for n, g, on in geoms],
                               title=os.path.basename(d), show_ui=True, point_size=3,
                               lookat=view["lookat"], eye=view["eye"], up=view["up"], bg_color=(1, 1, 1, 1))
    except Exception as e:  # classic viewer (no layer panel)
        print(f"new viewer failed ({e}); using the classic one")
        o3d.visualization.draw_geometries([g for _, g, on in geoms if on], window_name=os.path.basename(d),
                                          lookat=view["lookat"], front=(view["eye"] - view["lookat"]) /
                                          np.linalg.norm(view["eye"] - view["lookat"]), up=view["up"], zoom=0.6)


if __name__ == "__main__":
    main()
