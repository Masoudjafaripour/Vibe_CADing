"""Speed of the C++ motion engine vs the NumPy originals on the same clips.

    python bench_motion.py          # needs: pip install ./src/geometry_engine
    python bench_motion.py --plot   # also writes assets/cpp_motion.png
"""
import os
import sys
import time

import numpy as np

import control
import metrics
import motion_fast as fast
import skeleton
from motions import procedural_motion
from skeleton import IDX


def best_ms(fn, args_list, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        for args in args_list:
            fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best * 1000


def main():
    rng = np.random.default_rng(0)
    clips = [procedural_motion("walk", rng, yaw_rate=rng.uniform(-0.6, 0.6)) for _ in range(200)]
    poses = [(skeleton.forward_kinematics(r, a),) for r, a in clips]
    rows = [
        ("forward kinematics", skeleton.forward_kinematics, fast.forward_kinematics, clips),
        ("foot locking (IK)", control.lock_feet, fast.lock_feet, poses),
        ("stance sliding", control.stance_sliding, fast.stance_sliding, poses),
        ("foot sliding", metrics.foot_sliding, fast.foot_sliding, poses),
        ("jerk", metrics.jerk, fast.jerk, poses),
    ]
    print(f"200 walking clips x 90 frames x 18 joints (best of 3)\n")
    print(f"| {'step':20s} | {'NumPy':>10s} | {'C++':>9s} | {'speed-up':>8s} |")
    print(f"|{'-' * 22}|{'-' * 12}|{'-' * 11}|{'-' * 10}|")
    timings = []
    for name, py_fn, cpp_fn, args in rows:
        py, cpp = best_ms(py_fn, args), best_ms(cpp_fn, args)
        timings.append((name, py, cpp))
        print(f"| {name:20s} | {py:7.1f} ms | {cpp:6.1f} ms | {py / cpp:7.0f}x |")
    if "--plot" in sys.argv:
        plot(timings)


def plot(timings):
    """Left: toe paths of a turning walk before / after C++ foot locking. Right: speed-ups."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root, angles = procedural_motion("walk", np.random.default_rng(1), length=150, yaw_rate=0.5)
    pos = fast.forward_kinematics(root, angles)
    locked = fast.lock_feet(pos)
    stance = fast.stance_toes(pos)
    before, after = fast.stance_sliding(pos), fast.stance_sliding(locked, pos)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    for clip, style, label in ((pos, dict(lw=1, alpha=0.45), "original"), (locked, dict(lw=2), "C++ foot locking")):
        for toe, color in ((IDX["l_toe"], "#d9692a"), (IDX["r_toe"], "#2f62c9")):
            ax.plot(clip[:, toe, 0], clip[:, toe, 2], color=color, **style)
        planted = np.nonzero(stance >= 0)[0]
        ax.scatter(clip[planted, stance[planted], 0], clip[planted, stance[planted], 2], s=10 if clip is locked else 4,
                   color="black" if clip is locked else "grey", zorder=3, label=f"planted toe, {label}")
    ax.set_title(f"Turning walk, top view: planted toes slide {before:.2f} -> {after:.2f} m/s", fontsize=11)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="best")

    names = [n for n, _, _ in timings][::-1]
    speedups = [py / cpp for _, py, cpp in timings][::-1]
    bars = bx.barh(names, speedups, color="#2f62c9")
    bx.bar_label(bars, labels=[f"{v:.0f}x" for v in speedups], padding=3)
    bx.set_xscale("log")
    bx.set_xlabel("speed-up of C++ over NumPy (log scale)")
    bx.set_title("200 clips x 90 frames x 18 joints", fontsize=11)
    bx.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "cpp_motion.png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
