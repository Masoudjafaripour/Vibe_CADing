"""Pure-Python twin of src/marching_cubes.cpp, used as the speed baseline and
as an independent cross-check. Reads the same table from mc_tables.hpp.

    python bench/mc_python.py [cells per axis = 64]
"""
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "..", "include", "vibecad", "mc_tables.hpp")) as f:
    TRI = [[int(v) for v in row.split(",") if int(v) != -1] for row in re.findall(r"\{([^{}]*)\},", f.read())]

CORNER = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
EDGE = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def sphere(x, y, z, r):
    return math.sqrt(x * x + y * y + z * z) - r


def smooth_union(a, b, k):
    h = min(max(0.5 + 0.5 * (b - a) / k, 0.0), 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def snowman(x, y, z):  # same shape as shapes.hpp
    base = sphere(x, y + 0.55, z, 0.45)
    torso = sphere(x, y - 0.15, z, 0.33)
    head = sphere(x, y - 0.62, z, 0.23)
    return smooth_union(smooth_union(base, torso, 0.08), head, 0.06)


def sample(sdf, n, lo=-1.0, hi=1.0):
    h = (hi - lo) / n
    return [sdf(lo + i * h, lo + j * h, lo + k * h)
            for k in range(n + 1) for j in range(n + 1) for i in range(n + 1)]


def marching_cubes(values, n, lo=-1.0, hi=1.0, iso=0.0):
    m = n + 1
    h = (hi - lo) / n
    at = lambda i, j, k: values[(k * m + j) * m + i]
    vertices, triangles, edge_vertex = [], [], {}
    for k in range(n):
        for j in range(n):
            for i in range(n):
                mask = 0
                for c, (dx, dy, dz) in enumerate(CORNER):
                    if at(i + dx, j + dy, k + dz) < iso:
                        mask |= 1 << c
                if mask == 0 or mask == 255:
                    continue
                row = TRI[mask]
                ids = {}
                for e in set(row):
                    ca, cb = CORNER[EDGE[e][0]], CORNER[EDGE[e][1]]
                    low = (i + min(ca[0], cb[0]), j + min(ca[1], cb[1]), k + min(ca[2], cb[2]))
                    axis = next(d for d in range(3) if ca[d] != cb[d])
                    key = (low, axis)
                    if key not in edge_vertex:
                        high = list(low)
                        high[axis] += 1
                        a, b = at(*low), at(*high)
                        p = [lo + low[0] * h, lo + low[1] * h, lo + low[2] * h]
                        p[axis] += (iso - a) / (b - a) * h
                        edge_vertex[key] = len(vertices)
                        vertices.append(p)
                    ids[e] = edge_vertex[key]
                for t in range(0, len(row), 3):
                    triangles.append((ids[row[t]], ids[row[t + 2]], ids[row[t + 1]]))
    return vertices, triangles


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    t0 = time.perf_counter()
    values = sample(snowman, n)
    t1 = time.perf_counter()
    vertices, triangles = marching_cubes(values, n)
    t2 = time.perf_counter()
    print(f"snowman  {n}^3 cells  ->  {len(vertices)} vertices, {len(triangles)} triangles")
    print(f"time: sample {1000 * (t1 - t0):.1f} ms | marching cubes {1000 * (t2 - t1):.1f} ms (pure Python)")


if __name__ == "__main__":
    main()
