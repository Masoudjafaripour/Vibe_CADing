"""Articulated stick-figure skeleton and forward kinematics (FK).

Coordinates: Y up, the character faces +Z, its left is +X. Units: metres.
A pose is one local rotation per joint (XYZ Euler angles, radians) plus
the root (pelvis) position; FK turns that into world joint positions.
"""
import numpy as np

# name, parent index, offset from the parent joint in the rest pose
JOINTS = [
    ("pelvis",     -1, (0.0, 0.0, 0.0)),
    ("spine",       0, (0.0, 0.22, 0.0)),
    ("neck",        1, (0.0, 0.28, 0.0)),
    ("head",        2, (0.0, 0.18, 0.0)),
    ("l_shoulder",  2, (0.18, -0.03, 0.0)),
    ("l_elbow",     4, (0.0, -0.28, 0.0)),
    ("l_wrist",     5, (0.0, -0.25, 0.0)),
    ("r_shoulder",  2, (-0.18, -0.03, 0.0)),
    ("r_elbow",     7, (0.0, -0.28, 0.0)),
    ("r_wrist",     8, (0.0, -0.25, 0.0)),
    ("l_hip",       0, (0.10, -0.05, 0.0)),
    ("l_knee",     10, (0.0, -0.43, 0.0)),
    ("l_ankle",    11, (0.0, -0.43, 0.0)),
    ("l_toe",      12, (0.0, -0.05, 0.12)),
    ("r_hip",       0, (-0.10, -0.05, 0.0)),
    ("r_knee",     14, (0.0, -0.43, 0.0)),
    ("r_ankle",    15, (0.0, -0.43, 0.0)),
    ("r_toe",      16, (0.0, -0.05, 0.12)),
]
NAMES = [name for name, _, _ in JOINTS]
IDX = {name: i for i, name in enumerate(NAMES)}
PARENTS = np.array([parent for _, parent, _ in JOINTS])
OFFSETS = np.array([offset for _, _, offset in JOINTS], dtype=float)
NUM_JOINTS = len(JOINTS)
BONES = [(int(p), j) for j, p in enumerate(PARENTS) if p >= 0]
REST_HEIGHT = 0.96  # pelvis height that puts the toes on the ground (y = 0)


def _axis_rotation(a, axis):
    c, s = np.cos(a), np.sin(a)
    o, l = np.zeros_like(a), np.ones_like(a)
    rows = {
        0: [[l, o, o], [o, c, -s], [o, s, c]],
        1: [[c, o, s], [o, l, o], [-s, o, c]],
        2: [[c, -s, o], [s, c, o], [o, o, l]],
    }[axis]
    return np.stack([np.stack(r, -1) for r in rows], -2)


def euler_to_matrix(angles):
    """(..., 3) XYZ Euler angles -> (..., 3, 3) rotation R = Rx @ Ry @ Rz."""
    return (_axis_rotation(angles[..., 0], 0) @ _axis_rotation(angles[..., 1], 1)
            @ _axis_rotation(angles[..., 2], 2))


def forward_kinematics(root, angles):
    """root (T, 3), angles (T, J, 3) -> world joint positions (T, J, 3).

    G_j = G_parent(j) @ R(theta_j),   p_j = p_parent(j) + G_parent(j) @ offset_j
    """
    local = euler_to_matrix(angles)
    frames = root.shape[0]
    pos = np.zeros((frames, NUM_JOINTS, 3))
    glob = np.zeros((frames, NUM_JOINTS, 3, 3))
    for j, p in enumerate(PARENTS):  # parents always come before children
        if p < 0:
            pos[:, j], glob[:, j] = root, local[:, j]
        else:
            pos[:, j] = pos[:, p] + glob[:, p] @ OFFSETS[j]
            glob[:, j] = glob[:, p] @ local[:, j]
    return pos
