"""PiPER forward kinematics from the joint origins in piper_description.urdf (agilexrobotics/piper_ros, humble).

The RealSense sits on the wrist just above the gripper, looking the same way as the gripper (photo, 2026-11-17).
We take the camera centre as the gripper_base origin moved CAM_OFFSET along the gripper's approach axis (its z).

    python robot/piper_fk.py 5 26.8 -62.6 10 40.7 0      # joints in degrees -> camera position, radius, view direction
"""
import sys

import numpy as np

# (xyz, rpy) of joint1..joint6 origins, then gripper_base (fixed, identity on link6). Axes are all local z.
JOINTS = [
    ((0, 0, 0.123), (0, 0, 0)),
    ((0, 0, 0), (1.5708, -0.1359, -3.1416)),
    ((0.28503, 0, 0), (0, 0, -1.7939)),
    ((-0.021984, -0.25075, 0), (1.5708, 0, 0)),
    ((0, 0, 0), (-1.5708, 0, 0)),
    ((8.8259e-05, -0.091, 0), (1.5708, 0, 0)),
]
CAM_OFFSET = np.array([0.0, 0.0, 0.03])   # metres in the gripper_base frame (z = towards the fingers); rough, from the photo


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _rz(q):
    c, s = np.cos(q), np.sin(q)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def gripper_pose(q_deg):
    """4x4 pose of gripper_base in the base_link frame (metres)."""
    T = np.eye(4)
    for (xyz, rpy), q in zip(JOINTS, np.deg2rad(np.asarray(q_deg, float))):
        A = np.eye(4); A[:3, :3] = _rpy(*rpy); A[:3, 3] = xyz
        Z = np.eye(4); Z[:3, :3] = _rz(q)
        T = T @ A @ Z
    return T


def camera(q_deg):
    """Camera centre (m, base frame), its horizontal distance from the J1 axis, and the gripper approach direction."""
    T = gripper_pose(q_deg)
    c = T[:3, :3] @ CAM_OFFSET + T[:3, 3]
    approach = T[:3, 2]
    elev = float(np.degrees(np.arcsin(np.clip(approach[2], -1, 1))))
    return dict(center=c, radius=float(np.hypot(c[0], c[1])), height=float(c[2]), approach=approach, elevation_deg=elev)


if __name__ == "__main__":
    q = [float(x) for x in sys.argv[1:7]] if len(sys.argv) >= 7 else [0, 0, 0, 0, 0, 0]
    r = camera(q)
    print(f"joints {q}")
    print(f"camera centre (m) {np.round(r['center'], 3)}  radius from base axis {r['radius']:.3f} m  height {r['height']:.3f} m")
    print(f"looking direction {np.round(r['approach'], 3)}  elevation {r['elevation_deg']:+.1f} deg")
