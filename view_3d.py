"""
view_3d.py - interactive 3D view of the active-camera result (opens in the web browser).

Reads outputs_active/steps/<circle>/scene.npz (written by active_camera_steps.py) and writes
  view3d.html  - rotate (drag), zoom (wheel), pan (right-drag); click legend items to hide/show layers
  room.ply     - the coloured room + footprint for MeshLab / CloudCompare (optional)
Usage (from sotsuron):  ..\\.venv\\Scripts\\python.exe -I view_3d.py outputs_active\\steps\\20261116_1906_Room_20_degree
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

import active_camera as A  # noqa: E402

MAX_POINTS = 160_000


def lines_trace(segments, name, color, width=4, dash=None, visible=True):
    xs, ys, zs = [], [], []
    for a, b in segments:
        xs += [a[0], b[0], None]; ys += [a[1], b[1], None]; zs += [a[2], b[2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=name, visible=visible,
                        line=dict(color=color, width=width, dash=dash), hoverinfo="name")


def frustum(center, R_w2c, hfov, vfov, length):
    """4 edges + far rectangle of a camera's field of view."""
    th, tv = np.tan(np.radians(hfov / 2)), np.tan(np.radians(vfov / 2))
    corners_c = [np.array([sx * th, sy * tv, 1.0]) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    corners = [center + length * (R_w2c.T @ c) / np.linalg.norm(c) for c in corners_c]
    segs = [(center, c) for c in corners] + [(corners[k], corners[(k + 1) % 4]) for k in range(4)]
    return segs


def write_ply(path, P, rgb):
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        rec = np.empty(len(P), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        rec["p"], rec["c"] = P, rgb
        f.write(rec.tobytes())


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "outputs_active", "steps", "20261116_1906_Room_20_degree")
    S = np.load(os.path.join(d, "scene.npz"))
    P, rgb, w = S["P"], S["rgb"].astype(np.float32) / 255.0, S["w"]
    eye, gaze, half, cam, up, ref = S["eye"], S["gaze"], float(S["half"]), S["cam"], S["up"], S["ref"]
    hfov, vfov = float(S["hfov"]), float(S["vfov"])
    L = float(np.percentile(np.linalg.norm(P - cam, axis=1), 90))

    # show the room in its real colours; the gaze footprint tinted red by its weight
    red = np.array([1.0, 0.05, 0.05])
    a = np.clip(w / max(w.max(), 1e-9), 0, 1)[:, None] * 0.85
    col = (1 - a) * rgb + a * red
    rng = np.random.default_rng(0)
    idx = rng.choice(len(P), min(MAX_POINTS, len(P)), replace=False)
    foot = np.where(w > 0.05)[0]

    def colors(c):
        return [f"rgb({int(255 * r)},{int(255 * g)},{int(255 * b)})" for r, g, b in c]

    traces = [
        go.Scatter3d(x=P[idx, 0], y=P[idx, 1], z=P[idx, 2], mode="markers", name="room (photo colours, footprint in red)",
                     marker=dict(size=1.6, color=colors(col[idx])), hoverinfo="skip"),
        go.Scatter3d(x=P[foot, 0], y=P[foot, 1], z=P[foot, 2], mode="markers", name="gaze footprint only",
                     marker=dict(size=2, color=w[foot], colorscale="Reds", cmin=0, cmax=1), hoverinfo="skip",
                     visible="legendonly"),
        go.Scatter3d(x=[eye[0]], y=[eye[1]], z=[eye[2]], mode="markers+text", text=["eye"], name="eye",
                     marker=dict(size=7, color="black")),
        lines_trace([(eye, eye + 1.1 * L * gaze)], "gaze ray (L2CS mean)", "red", 7),
    ]
    # cone surface lines every 30 degrees around the gaze
    u = np.cross(gaze, up); u /= np.linalg.norm(u); v = np.cross(u, gaze)
    cone = []
    for k in range(12):
        ang = np.radians(30 * k)
        dvec = np.cos(np.radians(half)) * gaze + np.sin(np.radians(half)) * (np.cos(ang) * u + np.sin(ang) * v)
        cone.append((eye, eye + 1.1 * L * dvec))
    traces.append(lines_trace(cone, f"gaze cone ±{half:.0f}° (95%)", "rgba(255,0,0,0.45)", 2, "dash"))

    traces.append(go.Scatter3d(x=[cam[0]], y=[cam[1]], z=[cam[2]], mode="markers+text", text=["camera spot"],
                               name="camera spot", marker=dict(size=7, color="blue", symbol="square")))
    R = A.rotation_looking_at(A.forward_from_pan_tilt(float(S["best_pan"]), float(S["best_tilt"]), ref, up), up)
    traces.append(lines_trace(frustum(cam, R, hfov, vfov, 0.8 * L),
                              f"chosen view: pan {float(S['best_pan']):+.0f}°, tilt {float(S['best_tilt']):+.0f}°", "blue", 5))
    real = []
    for c, f, yw, cp in zip(S["centers"], S["fwd"], S["yaws"], S["captured"]):
        real.append((c, c + 0.25 * L * f))
    traces.append(lines_trace(real, "real photo directions (Turn Capture)", "rgba(0,120,255,0.6)", 3, visible="legendonly"))
    traces.append(lines_trace([(S["person_center"], S["person_center"] + 0.3 * L * S["person_fwd"])],
                              "person-photo camera", "purple", 5, visible="legendonly"))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"{os.path.basename(d)}: gaze footprint (red) and the chosen camera view (blue)",
        scene=dict(aspectmode="data", xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                   camera=dict(up=dict(x=float(up[0]), y=float(up[1]), z=float(up[2])),
                               eye=dict(x=float(-ref[0] * 1.2 + up[0] * 1.2), y=float(-ref[1] * 1.2 + up[1] * 1.2),
                                        z=float(-ref[2] * 1.2 + up[2] * 1.2)))),
        legend=dict(itemsizing="constant", bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="white")
    out = os.path.join(d, "view3d.html")
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    write_ply(os.path.join(d, "room.ply"), P, (np.clip(col, 0, 1) * 255).astype(np.uint8))
    print(f"-> {out}\n-> {os.path.join(d, 'room.ply')}")


if __name__ == "__main__":
    main()
