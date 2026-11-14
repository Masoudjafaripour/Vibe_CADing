"""C++ versions of the motion geometry, with the same signatures as the NumPy
originals (skeleton.forward_kinematics, metrics.foot_sliding / jerk,
control.stance_toes / lock_feet / stance_sliding), plus BVH export.

    pip install ./src/geometry_engine      # builds vibecad_motion (from the repo root)

    from motion_fast import forward_kinematics, lock_feet, write_bvh
    write_bvh("walk.bvh", root, angles)    # opens in Blender, Maya, Unity
"""
import numpy as np
import vibecad_motion as _m

from motions import FPS
from skeleton import IDX, NAMES, OFFSETS, PARENTS

_PARENTS = PARENTS.astype(np.int32)
_TOES = (IDX["l_toe"], IDX["r_toe"])
_LEGS = [tuple(IDX[f"{side}_{j}"] for j in ("hip", "knee", "ankle", "toe")) for side in "lr"]


def forward_kinematics(root, angles):
    return _m.forward_kinematics(root, angles, _PARENTS, OFFSETS)


def stance_toes(pos, contact_height=0.03, max_speed=1.0):
    return np.array(_m.stance_toes(pos, *_TOES, FPS, contact_height, max_speed))


def lock_feet(pos, contact_height=0.03, max_drift=0.35):
    return _m.lock_feet(pos, _LEGS[0], _LEGS[1], FPS, contact_height, max_drift)


def stance_sliding(pos, reference=None, fps=FPS):
    return _m.stance_sliding(pos, pos if reference is None else reference, *_TOES, fps)


def foot_sliding(pos, fps=FPS, contact_height=0.02):
    return _m.foot_sliding(pos, *_TOES, fps, contact_height)


def jerk(pos, fps=FPS):
    return _m.jerk(pos, fps)


def write_bvh(path, root, angles, fps=FPS):
    """Writes a clip as BVH (metres, Y up, XYZ Euler in degrees)."""
    _m.write_bvh(path, root, angles, _PARENTS, OFFSETS, NAMES, fps)
