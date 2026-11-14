"""The C++ motion engine (motion_fast.py) must match the NumPy originals on real
clips, and BVH files must replay to the same joint positions.

    cd src/animation && python -m unittest test_motion_fast.py -v
"""
import os
import re
import tempfile
import unittest

import numpy as np

import control
import metrics
import motion_fast as fast
import skeleton
from motions import FPS, procedural_motion


def clips():
    """Baseline + randomized clips for every command, plus turning walks."""
    out = []
    for command in ("walk", "wave", "raise_arms"):
        out.append(procedural_motion(command))
        for seed in range(3):
            out.append(procedural_motion(command, np.random.default_rng(seed)))
    for yaw in (-0.6, 0.4):
        out.append(procedural_motion("walk", np.random.default_rng(7), length=150, yaw_rate=yaw))
    return out


def read_bvh(path):
    """Minimal independent BVH reader -> (names, parents, offsets, channels per frame)."""
    with open(path) as f:
        lines = f.read().split("\n")
    names, parents, offsets, stack, end_site = [], [], [], [], False
    i = 0
    while not lines[i].startswith("MOTION"):
        tok = lines[i].split()
        if tok and tok[0] in ("ROOT", "JOINT"):
            names.append(tok[1])
            parents.append(stack[-1] if stack else -1)
        elif tok and tok[0] == "End":
            end_site = True
        elif tok and tok[0] == "{":
            stack.append(len(names) - 1) if not end_site else None
        elif tok and tok[0] == "}":
            if end_site:
                end_site = False
            else:
                stack.pop()
        elif tok and tok[0] == "OFFSET" and not end_site:
            offsets.append([float(v) for v in tok[1:4]])
        i += 1
    frames = int(re.search(r"Frames:\s*(\d+)", lines[i + 1]).group(1))
    data = np.array([[float(v) for v in l.split()] for l in lines[i + 3:i + 3 + frames]])
    return names, np.array(parents), np.array(offsets), data


class TestParity(unittest.TestCase):
    def setUp(self):
        self.clips = clips()

    def test_forward_kinematics(self):
        for root, angles in self.clips:
            np.testing.assert_allclose(fast.forward_kinematics(root, angles),
                                       skeleton.forward_kinematics(root, angles), atol=1e-12)

    def test_metrics(self):
        for root, angles in self.clips:
            pos = skeleton.forward_kinematics(root, angles)
            self.assertAlmostEqual(fast.jerk(pos), metrics.jerk(pos), places=6)
            self.assertAlmostEqual(fast.foot_sliding(pos), metrics.foot_sliding(pos), places=10)
            np.testing.assert_array_equal(fast.stance_toes(pos), control.stance_toes(pos))
            self.assertAlmostEqual(fast.stance_sliding(pos), control.stance_sliding(pos), places=10)

    def test_lock_feet(self):
        locked_any = False
        for root, angles in self.clips:
            pos = skeleton.forward_kinematics(root, angles)
            want, got = control.lock_feet(pos), fast.lock_feet(pos)
            np.testing.assert_allclose(got, want, atol=1e-12)
            locked_any |= not np.allclose(want, pos)
            self.assertAlmostEqual(fast.stance_sliding(got, pos), control.stance_sliding(want, pos), places=10)
        self.assertTrue(locked_any, "foot locking never engaged; the test would prove nothing")


class TestBvh(unittest.TestCase):
    def test_round_trip(self):
        root, angles = procedural_motion("walk", np.random.default_rng(1), length=120, yaw_rate=0.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "walk.bvh")
            fast.write_bvh(path, root, angles)
            names, parents, offsets, data = read_bvh(path)
        self.assertEqual(names, skeleton.NAMES)
        np.testing.assert_array_equal(parents, skeleton.PARENTS)
        np.testing.assert_allclose(offsets, skeleton.OFFSETS, atol=1e-6)
        self.assertEqual(data.shape, (120, 3 + 3 * len(names)))
        replay = skeleton.forward_kinematics(data[:, :3], np.radians(data[:, 3:]).reshape(120, -1, 3))
        np.testing.assert_allclose(replay, skeleton.forward_kinematics(root, angles), atol=1e-5)

    def test_frame_time(self):
        root, angles = procedural_motion("wave")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wave.bvh")
            fast.write_bvh(path, root, angles)
            with open(path) as f:
                text = f.read()
        self.assertIn(f"Frame Time: {1 / FPS:.6f}", text)
        self.assertEqual(text.count("End Site"), 5)  # head, 2 wrists, 2 toes


if __name__ == "__main__":
    unittest.main()
