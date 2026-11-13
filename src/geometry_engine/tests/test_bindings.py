"""Tests for the vibecad_geom Python module (pip install ./src/geometry_engine).

    cd src/geometry_engine && python -m unittest tests/test_bindings.py -v
"""
import math
import os
import tempfile
import unittest

import numpy as np
import vibecad_geom as g


def grid(sdf, m=65, lo=-1.0, hi=1.0):
    x = np.linspace(lo, hi, m, dtype=np.float32)
    z, y, xx = np.meshgrid(x, x, x, indexing="ij")  # index order [z, y, x]
    return sdf(xx, y, z)


class TestBindings(unittest.TestCase):
    def test_sphere(self):
        v, t = g.marching_cubes(grid(lambda x, y, z: np.sqrt(x * x + y * y + z * z) - 0.6))
        self.assertEqual((v.dtype, t.dtype, v.shape[1], t.shape[1]), (np.float32, np.uint32, 3, 3))
        s = g.analyze(v, t)
        self.assertTrue(s["watertight"] and s["oriented"])
        self.assertEqual((s["components"], s["genus"]), (1, 0))
        self.assertAlmostEqual(s["volume"], 4 / 3 * math.pi * 0.6 ** 3, delta=0.01)

    def test_torus_genus(self):
        v, t = g.marching_cubes(grid(lambda x, y, z: np.hypot(np.hypot(x, z) - 0.5, y) - 0.2))
        self.assertEqual(g.analyze(v, t)["genus"], 1)

    def test_write_stl(self):
        v, t = g.marching_cubes(grid(lambda x, y, z: np.sqrt(x * x + y * y + z * z) - 0.6, m=17))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.stl")
            g.write_stl(v, t, path)
            self.assertEqual(os.path.getsize(path), 84 + 50 * len(t))

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            g.marching_cubes(np.zeros((4, 5, 6), np.float32))
        with self.assertRaises(ValueError):
            g.analyze(np.zeros((3, 3), np.float32), np.array([[0, 1, 7]], np.uint32))


if __name__ == "__main__":
    unittest.main()
