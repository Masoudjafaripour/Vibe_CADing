"""Renders the README demo images from STL files made by mc_demo.

    ./build/mc_demo snowman 12 build/snowman_12.stl   (also 24, 64; torus 32)
    python bench/render_demo.py      # needs numpy, matplotlib, pillow
"""
import io
import os
import struct

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "..", "build")
ASSETS = os.path.join(HERE, "..", "assets")
LIGHT = np.array([0.4, 0.8, 0.5]) / np.linalg.norm([0.4, 0.8, 0.5])


def load_stl(path):
    data = open(path, "rb").read()
    count = struct.unpack("<I", data[80:84])[0]
    rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    return np.frombuffer(data[84:], dtype=rec, count=count)["v"][:, :, [0, 2, 1]]  # Y up


def draw(ax, tris, azim, color, edges):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    shade = 0.3 + 0.7 * np.clip(n[:, [0, 2, 1]] @ LIGHT, 0, 1)
    faces = np.clip(np.outer(shade, color), 0, 1)
    ax.add_collection3d(Poly3DCollection(tris, facecolors=faces, edgecolors=(0, 0, 0, 0.35) if edges else "none",
                                         linewidths=0.3))
    for lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        lim(-0.9, 0.9)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(15, azim)
    ax.axis("off")


def resolution_strip():
    orange = np.array([0.95, 0.55, 0.25])
    fig = plt.figure(figsize=(10, 3.6))
    for col, n in enumerate((12, 24, 64)):
        tris = load_stl(os.path.join(BUILD, f"snowman_{n}.stl"))
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        draw(ax, tris, -60, orange, edges=n < 64)
        ax.set_title(f"{n}³ grid · {len(tris):,} triangles", fontsize=10, y=0.02)
    fig.subplots_adjust(0, 0, 1, 1, 0, 0)
    fig.savefig(os.path.join(ASSETS, "snowman_resolutions.png"), dpi=110)
    plt.close(fig)


def turntable(name, color, frames=24):
    tris = load_stl(os.path.join(BUILD, name))
    images = []
    for i in range(frames):
        fig = plt.figure(figsize=(3, 3))
        draw(fig.add_subplot(projection="3d"), tris, -60 + 360 * i / frames, color, edges=True)
        fig.subplots_adjust(0, 0, 1, 1)
        buf = io.BytesIO()
        fig.savefig(buf, dpi=90)
        plt.close(fig)
        images.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    out = os.path.join(ASSETS, name.replace(".stl", ".gif"))
    images[0].save(out, save_all=True, append_images=images[1:], duration=80, loop=0, optimize=True)


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    resolution_strip()
    turntable("snowman_24.stl", np.array([0.95, 0.55, 0.25]))
    turntable("torus_32.stl", np.array([0.35, 0.55, 0.95]))
    print("wrote", sorted(os.listdir(ASSETS)))
