"""Evaluation on FK joint positions (T, J, 3)."""
import numpy as np

from motions import FPS
from skeleton import IDX


def jerk(pos, fps=FPS):
    """Mean joint jerk |d^3 p / dt^3| in m/s^3; lower is smoother."""
    return float(np.linalg.norm(np.diff(pos, n=3, axis=0) * fps ** 3, axis=-1).mean())


def mpjpe(a, b):
    """Mean per-joint position error in metres."""
    return float(np.linalg.norm(a - b, axis=-1).mean())


def foot_sliding(pos, fps=FPS, contact_height=0.02):
    """Mean horizontal toe speed (m/s) while the toe touches the ground; 0 = planted feet."""
    toes = pos[:, [IDX["l_toe"], IDX["r_toe"]]]
    speed = np.linalg.norm(np.diff(toes[..., [0, 2]], axis=0), axis=-1) * fps
    contact = toes[1:, :, 1] < contact_height
    return float(speed[contact].mean()) if contact.any() else 0.0


def diversity(clips):
    """Mean pairwise MPJPE (m) between clips generated from the same text; 0 = deterministic."""
    pairs = [mpjpe(a, b) for i, a in enumerate(clips) for b in clips[i + 1:]]
    return float(np.mean(pairs)) if pairs else 0.0


def classify(pos):
    """Rule-based check of what a clip does: walk / wave / raise_arms / other."""
    forward = pos[-1, IDX["pelvis"], 2] - pos[0, IDX["pelvis"], 2]
    feet = np.sign(pos[:, IDX["l_ankle"], 2] - pos[:, IDX["r_ankle"], 2])
    steps = int((np.diff(feet[feet != 0]) != 0).sum())
    if forward > 0.8 and steps >= 2:
        return "walk"
    late = pos[len(pos) // 2:]
    head = late[:, IDX["head"], 1]
    l_up = (late[:, IDX["l_wrist"], 1] > head).mean() > 0.8
    r_up = (late[:, IDX["r_wrist"], 1] > head).mean() > 0.8
    if l_up and r_up:
        return "raise_arms"
    if r_up and not l_up and late[:, IDX["r_wrist"], 0].std() > 0.03:
        return "wave"
    return "other"
