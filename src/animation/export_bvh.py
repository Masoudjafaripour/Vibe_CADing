"""Text -> motion -> BVH file, via the C++ engine (motion_fast.py).

For each command, writes the procedural clip and, if trained checkpoints
exist (python generate_motion.py), the AR and DDPM clips generated from text:

    python export_bvh.py                      # -> outputs/bvh/<command>_<source>.bvh
    python export_bvh.py --text "walk forward" --method diffusion --seed 3

BVH stores joint rotations, so it holds the clip as generated; foot locking
(lock_feet) edits positions only and is not part of the file.
"""
import argparse
import os

import numpy as np
import torch

import motion_fast as fast
from generate_motion import OUT_DIR, Generator, ckpt_path
from metrics import jerk
from motions import COMMANDS, procedural_motion

BVH_DIR = os.path.join(OUT_DIR, "bvh")


def export(name, root, angles):
    path = os.path.join(BVH_DIR, name + ".bvh")
    fast.write_bvh(path, root, angles)
    pos = fast.forward_kinematics(root, angles)
    before, after = fast.stance_sliding(pos), fast.stance_sliding(fast.lock_feet(pos), pos)
    print(f"{os.path.basename(path):28s} {len(root)} frames | jerk {jerk(pos):6.1f} m/s^3 | "
          f"planted-foot sliding {before:.2f} -> {after:.2f} m/s with C++ foot locking")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", help="one instruction instead of every command")
    parser.add_argument("--method", choices=("ar", "diffusion"), default="diffusion")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(BVH_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods = [m for m in ("ar", "diffusion") if os.path.exists(ckpt_path(m))]
    if not methods:
        print("No trained checkpoints yet (run: python generate_motion.py); exporting procedural clips only.")
    generators = {m: Generator(m, device) for m in methods}

    if args.text:
        if args.method not in generators:
            raise SystemExit(f"no {args.method} checkpoint; run python generate_motion.py first")
        root, angles, _ = generators[args.method](args.text, seed=args.seed)
        export("_".join(args.text.lower().split()) + f"_{args.method}", root, angles)
        return

    for command, phrases in COMMANDS.items():
        export(f"{command}_procedural", *procedural_motion(command))
        for method, gen in generators.items():
            root, angles, _ = gen(phrases[0], seed=args.seed)
            export(f"{command}_{method}", root, angles)
    print(f"\nOpen in Blender: File > Import > Motion Capture (.bvh) -> {BVH_DIR}")


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
