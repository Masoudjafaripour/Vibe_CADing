"""Procedural motions (the baseline and the training data) and the frame features.

Every clip starts from the rest pose, so the first frames carry no
information about the command: the model has to get it from the text.
"""
import numpy as np

from skeleton import IDX, NUM_JOINTS, REST_HEIGHT

FPS = 30
SEQ_LEN = 90  # 3 s
LEG_LENGTH = 0.91

# command -> phrases used for training and evaluation (the whole vocabulary)
COMMANDS = {
    "walk": ["walk forward", "walk", "walk ahead", "go forward"],
    "wave": ["wave", "wave hello", "wave your right hand"],
    "raise_arms": ["raise both arms", "raise your arms", "arms up"],
}


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def procedural_motion(command, rng=None, length=SEQ_LEN, yaw_rate=0.0):
    """Hand-written motion for `command` -> (root (T, 3), angles (T, J, 3)).

    With `rng`, period / amplitude / speed are randomized (training data);
    without it, every parameter is its mid value (the baseline clip).
    `yaw_rate` (rad/s, positive = turn left) curves the walk; it is used
    by control.py and is 0 for the original clips.
    """
    u = (lambda lo, hi: rng.uniform(lo, hi)) if rng is not None else (lambda lo, hi: (lo + hi) / 2)
    t = np.arange(length) / FPS
    angles = np.zeros((length, NUM_JOINTS, 3))
    root = np.zeros((length, 3))
    root[:, 1] = REST_HEIGHT
    # arms slightly away from the body in every pose
    angles[:, IDX["l_shoulder"], 2] = 0.08
    angles[:, IDX["r_shoulder"], 2] = -0.08

    if command == "walk":
        period, amp, speed = u(0.9, 1.2), u(0.35, 0.55), u(0.9, 1.4)  # s, rad, m/s
        w = 2 * np.pi / period
        ramp = smoothstep(t / 0.4)
        s = ramp * np.sin(w * t)
        # hips swing in opposition (negative X rotation = leg forward), knees bend in swing
        angles[:, IDX["l_hip"], 0] = -amp * s
        angles[:, IDX["r_hip"], 0] = amp * s
        angles[:, IDX["l_knee"], 0] = ramp * 0.6 * amp * (1 + np.sin(w * t + 0.6))
        angles[:, IDX["r_knee"], 0] = ramp * 0.6 * amp * (1 - np.sin(w * t + 0.6))
        # arms swing against the legs, elbows slightly bent, torso counter-rotates
        angles[:, IDX["l_shoulder"], 0] = 0.8 * amp * s
        angles[:, IDX["r_shoulder"], 0] = -0.8 * amp * s
        angles[:, IDX["l_elbow"], 0] = -0.3 * ramp
        angles[:, IDX["r_elbow"], 0] = -0.3 * ramp
        angles[:, IDX["spine"], 1] = 0.15 * amp * s
        # heading turns at yaw_rate (pelvis yaw); velocity follows the heading
        yaw = np.cumsum(yaw_rate * ramp) / FPS
        angles[:, IDX["pelvis"], 1] = yaw
        root[:, 0] = np.cumsum(speed * ramp * np.sin(yaw)) / FPS
        root[:, 2] = np.cumsum(speed * ramp * np.cos(yaw)) / FPS
        root[:, 1] = REST_HEIGHT - LEG_LENGTH * (1 - np.cos(amp * s))  # keep feet near the ground
    elif command == "wave":
        freq, amp = u(1.5, 2.5), u(0.4, 0.6)  # Hz, rad
        lift = smoothstep(t / 0.6)
        osc = smoothstep((t - 0.6) / 0.3) * amp * np.sin(2 * np.pi * freq * (t - 0.6))
        angles[:, IDX["r_shoulder"], 2] = -0.08 - 2.5 * lift  # right arm up and out
        angles[:, IDX["r_elbow"], 2] = -0.4 * lift + osc      # forearm swings side to side
    elif command == "raise_arms":
        duration, height = u(0.8, 1.4), u(2.6, 2.9)  # s, rad
        lift = smoothstep(t / duration)
        angles[:, IDX["l_shoulder"], 2] = 0.08 + height * lift
        angles[:, IDX["r_shoulder"], 2] = -0.08 - height * lift
    else:
        raise ValueError(f"Unknown command {command!r}; choose from {list(COMMANDS)}")
    return root, angles


def to_features(root, angles):
    """Per-frame vector x_t = [v_x, y, v_z, theta_1..J (XYZ)]: horizontal root
    velocity (translation invariant), root height, joint rotations."""
    vel = np.diff(root[:, [0, 2]], axis=0, prepend=root[:1, [0, 2]]) * FPS
    return np.concatenate([vel[:, :1], root[:, 1:2], vel[:, 1:], angles.reshape(len(root), -1)], 1)


def from_features(x):
    root = np.stack([np.cumsum(x[:, 0]) / FPS, x[:, 1], np.cumsum(x[:, 2]) / FPS], 1)
    return root, x[:, 3:].reshape(len(x), NUM_JOINTS, 3)
