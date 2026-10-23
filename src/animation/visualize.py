"""Animated 3D stick figure from a clip saved by generate_motion.py.

    python visualize.py outputs/walk_model.npz            # -> outputs/walk_model.gif
    python visualize.py outputs/*.npz                     # every clip
    python visualize.py outputs/wave_model.npz --show     # interactive window
"""
import argparse
import os
import sys

import matplotlib
import numpy as np

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from skeleton import NAMES

LABELS = {"ar": "MODEL-GENERATED (AR Transformer)", "diffusion": "MODEL-GENERATED (DDPM)",
          "procedural": "PROCEDURAL BASELINE (hand-written)"}


def bone_color(joint_name):
    """Left side blue, right side orange, spine/head grey."""
    return {"l_": "tab:blue", "r_": "tab:orange"}.get(joint_name[:2], "0.3")


def animate(path, show=False):
    clip = np.load(path)
    pos, parents, fps = clip["positions"], clip["parents"], int(clip["fps"])

    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=12, azim=-120)  # front-right of a character facing +Z
    ax.set_box_aspect((1, 1, 1), zoom=1.25)
    ax.set_title(f"{LABELS[str(clip['source'])]}\n\"{clip['text']}\"", fontsize=10)
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    # ground lines every 0.5 m, so forward motion is visible while the camera follows
    for g in np.arange(-2, 6, 0.5):
        ax.plot([-1.5, 1.5], [g, g], [0, 0], color="0.85", lw=0.8)
    bones = [(int(p), j) for j, p in enumerate(parents) if p >= 0]
    lines = [ax.plot([], [], [], lw=3, color=bone_color(NAMES[j]))[0] for _, j in bones]
    head = ax.plot([], [], [], "o", ms=10, color="0.3")[0]

    def update(f):
        # plot axes (x, y, z) = world (X, Z, Y): matplotlib's z is up
        for line, (p, j) in zip(lines, bones):
            line.set_data_3d(pos[f, [p, j], 0], pos[f, [p, j], 2], pos[f, [p, j], 1])
        h = pos[f, NAMES.index("head")]
        head.set_data_3d([h[0]], [h[2]], [h[1]])
        cx, cz = pos[f, 0, 0], pos[f, 0, 2]
        ax.set_xlim(cx - 1, cx + 1), ax.set_ylim(cz - 1, cz + 1), ax.set_zlim(0, 2)
        return lines + [head]

    anim = FuncAnimation(fig, update, frames=len(pos), interval=1000 / fps)
    if show:
        plt.show()
    else:
        out = os.path.splitext(path)[0] + ".gif"
        anim.save(out, writer=PillowWriter(fps=fps))
        print(f"Saved {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render saved motion clips as animated 3D skeletons.")
    parser.add_argument("clips", nargs="+", help=".npz files from generate_motion.py")
    parser.add_argument("--show", action="store_true", help="open a window instead of writing a GIF")
    args = parser.parse_args()
    for path in args.clips:
        animate(path, args.show)


if __name__ == "__main__":
    main()
