"""Evidence that the AR / DDPM clips come from the trained networks, not the procedural code.

    python diagnostics.py      # -> outputs/walk_curves.png, outputs/diagnostics/*.npz
    python render_character.py --grid outputs/diagnostics/*.npz --out walk_seeds_untrained

The procedural baseline uses the mid-range parameters, i.e. the mean of
the training distribution. The AR model (deterministic, MSE) learns that
mean, so it looks almost identical to the baseline; DDPM samples spread
over the training range. An untrained network on the same code path
produces no recognizable motion.
"""
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_motion import OUT_DIR, Generator, make_model, save_clip
from metrics import classify
from motions import FPS, SEQ_LEN, procedural_motion
from skeleton import IDX, forward_kinematics

TEXT = "walk forward"
SEEDS = (0, 1, 2)


def hip_deg(angles):
    return np.degrees(angles[:, IDX["l_hip"], 0])


def untrained(method, device):
    gen = Generator(method, device)
    torch.manual_seed(0)
    gen.model = make_model(method, int(gen.ckpt["active"].sum()), len(gen.vocab.words)).to(device).eval()
    return gen


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ar, ddpm = Generator("ar", device), Generator("diffusion", device)
    out = os.path.join(OUT_DIR, "diagnostics")
    os.makedirs(out, exist_ok=True)
    t = np.arange(SEQ_LEN) / FPS
    _, base = procedural_motion("walk")

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.5))
    rng = np.random.default_rng(3)
    for i in range(3):
        left.plot(t, hip_deg(procedural_motion("walk", rng)[1]), color="0.75", lw=1,
                  label="training samples" if i == 0 else None)
    left.plot(t, hip_deg(base), "k--", lw=2, label="procedural baseline (mid params)")
    root, angles, _ = ar(TEXT)
    left.plot(t, hip_deg(angles), color="tab:blue", lw=2, label="AR")
    right.plot(t, hip_deg(angles) - hip_deg(base), color="tab:blue", label="AR")
    save_clip(os.path.join(out, "1_ar.npz"), root, angles, TEXT, "ar")
    for s in SEEDS:
        root, angles, _ = ddpm(TEXT, seed=s)
        left.plot(t, hip_deg(angles), color="tab:green", lw=1.2, alpha=0.8, label="DDPM seeds" if s == 0 else None)
        right.plot(t, hip_deg(angles) - hip_deg(base), color="tab:green", alpha=0.8,
                   label="DDPM seeds" if s == 0 else None)
        save_clip(os.path.join(out, f"2_ddpm_seed{s}.npz"), root, angles, f"{TEXT} (seed {s})", "diffusion")
    root, angles, _ = untrained("ar", device)(TEXT)
    save_clip(os.path.join(out, "3_untrained_ar.npz"), root, angles, TEXT, "untrained")
    print(f"untrained AR is classified as: {classify(forward_kinematics(root, angles))}")

    left.set(title=f"left-hip flexion, \"{TEXT}\"", xlabel="time (s)", ylabel="degrees")
    right.set(title="difference from the procedural baseline", xlabel="time (s)", ylabel="degrees")
    right.axhline(0, color="k", lw=0.5)
    left.legend(fontsize=7), right.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "walk_curves.png")
    fig.savefig(path, dpi=100)
    print(f"Saved {path} and clips in {out}/")


if __name__ == "__main__":
    main()
