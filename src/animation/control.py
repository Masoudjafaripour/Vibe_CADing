"""Controllable, continuous walking: a second DDPM conditioned on past frames and a 2D root path.

    python control.py            # train outputs/control.pt (~1 min GPU), evaluate, write outputs/control/
    python render_character.py --grid outputs/control/continuous_*.npz --out continuous_walk
    python render_character.py --grid outputs/control/traj_*.npz --out trajectory_compare

The original AR / DDPM models are untouched. This model reuses MotionDenoiser
with extra per-frame condition channels:

    cond_t = [ known_t * m_t,  m_t,  traj_t * h,  h ]
      m_t     1 for frames given in advance (the last frames of the previous segment)
      known_t those frames (normalized features)
      traj_t  target root (x, z) in the segment's own frame, h = 1 if a path is given

Training windows (96 frames) are cut at random offsets from long procedural
walks with random turn rates, and each window is expressed in its first
frame's local frame (origin at the root, heading +Z). That makes "start
mid-stride" and "continue from here" part of the training distribution.
At sampling time, known frames are also enforced by inpainting
(GaussianDiffusion.sample with known / mask), so the boundary is exact.
"""
import json
import os
import time

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion import GaussianDiffusion, MotionDenoiser
from metrics import foot_sliding, jerk
from model import Vocab
from motions import FPS, from_features, procedural_motion, smoothstep, to_features
from skeleton import IDX, PARENTS, forward_kinematics

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs", "control")
CKPT = os.path.join(HERE, "outputs", "control.pt")
WIN = 96          # frames per generated segment (3.2 s)
PREFIX = 12       # frames carried over from the previous segment
LONG = 192        # length of the procedural walks the windows are cut from
TRAJ_SCALE = 3.0  # metres -> condition units
SPEED = 1.15      # m/s, mid-range walking speed of the procedural data
COMMANDS = {      # phrases, training turn-rate range (rad/s, + = left)
    "straight": (["walk straight", "walk forward", "go straight"], (-0.05, 0.05)),
    "left": (["turn left", "walk left", "curve left"], (0.5, 1.0)),
    "right": (["turn right", "walk right", "curve right"], (-1.0, -0.5)),
}


# ---------- local (segment) frames ----------

def yaw_rotate(xz, a):
    """Rotates (..., 2) [x, z] vectors about +Y by `a` radians (same as Ry(a))."""
    c, s = np.cos(a), np.sin(a)
    return np.stack([c * xz[..., 0] + s * xz[..., 1], -s * xz[..., 0] + c * xz[..., 1]], -1)


def to_local(root, angles):
    """World clip -> clip in its first frame's frame (root at origin, heading +Z)."""
    origin, yaw0 = root[0, [0, 2]].copy(), angles[0, IDX["pelvis"], 1]
    r, a = root.copy(), angles.copy()
    r[:, [0, 2]] = yaw_rotate(root[:, [0, 2]] - origin, -yaw0)
    a[:, IDX["pelvis"], 1] -= yaw0
    return r, a, (origin, yaw0)


def to_world(root, angles, frame):
    origin, yaw0 = frame
    r, a = root.copy(), angles.copy()
    r[:, [0, 2]] = yaw_rotate(root[:, [0, 2]], yaw0) + origin
    a[:, IDX["pelvis"], 1] += yaw0
    return r, a


def path_from_controls(speed, yaw_rate, start=(0.0, 0.0), heading=0.0):
    """Integrates per-frame speed (m/s) and turn rate (rad/s) into an (x, z) path."""
    yaw = heading + np.cumsum(yaw_rate) / FPS
    step = np.stack([np.sin(yaw), np.cos(yaw)], -1) * speed[:, None] / FPS
    return np.asarray(start) + np.cumsum(step, 0)


def target_path(yaw_rate, length=WIN):
    """Mid-speed walk from rest with a constant turn rate (the same ramp as the data)."""
    speed = SPEED * smoothstep(np.arange(length) / FPS / 0.4)
    path = path_from_controls(speed, np.full(length, yaw_rate) * speed / SPEED)
    return path - path[0]


# ---------- data and training ----------

def build_windows(rng, vocab, clips_per_command=300, windows_per_clip=4):
    feats, words = [], []
    for phrases, (lo, hi) in COMMANDS.values():
        for _ in range(clips_per_command):
            root, angles = procedural_motion("walk", rng, length=LONG, yaw_rate=rng.uniform(lo, hi))
            for _ in range(windows_per_clip):
                # 25% of windows start from rest, the rest mid-walk
                start = 0 if rng.random() < 0.25 else int(rng.integers(1, LONG - WIN + 1))
                r, a, _ = to_local(root[start:start + WIN], angles[start:start + WIN])
                feats.append(to_features(r, a))
                words.append(vocab.encode(rng.choice(phrases)))
    return np.stack(feats).astype(np.float32), np.array(words)


def traj_from_features(x):
    """Root (x, z) path, in condition units, from raw (unnormalized) features (B, T, F)."""
    return np.cumsum(x[..., [0, 2]], 1) / FPS / TRAJ_SCALE


def make_cond(xn, traj, prefix_len, has_traj):
    """xn (B, T, F) normalized features; traj (B, T, 2); prefix_len, has_traj (B,)."""
    B, T, _ = xn.shape
    mask = (torch.arange(T, device=xn.device)[None] < prefix_len[:, None]).float()[..., None]
    h = has_traj.float()[:, None, None]
    cond = torch.cat([xn * mask, mask, traj * h, h.expand(B, T, 1)], -1)
    return cond, mask


def train(device, steps=8000, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    vocab = Vocab.from_phrases(p for phrases, _ in COMMANDS.values() for p in phrases)
    X, W = build_windows(rng, vocab)
    mean, std = X.mean((0, 1)), X.std((0, 1))
    active = std > 1e-6
    std[~active] = 1.0
    Xn = torch.tensor((X[..., active] - mean[active]) / std[active], device=device)
    P = torch.tensor(traj_from_features(X), dtype=torch.float32, device=device)
    W = torch.tensor(W, device=device)
    n = int(active.sum())
    model = MotionDenoiser(n, len(vocab.words), cond_dim=n + 4).to(device)
    diffusion = GaussianDiffusion(device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    print(f"[control] {len(X)} windows, {n} active features, {sum(p.numel() for p in model.parameters()):,} params")
    start = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(Xn), (64,), device=device)
        # half the batch continues from a known prefix of 8-16 frames; 80% get the path
        prefix = torch.where(torch.rand(64, device=device) < 0.5,
                             torch.randint(8, 17, (64,), device=device), torch.zeros(64, dtype=torch.long, device=device))
        cond, _ = make_cond(Xn[idx], P[idx], prefix, torch.rand(64, device=device) > 0.2)
        loss = diffusion.loss(model, Xn[idx], W[idx], cond=cond)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        ema.update_parameters(model)
        if step == 1 or step % 2000 == 0:
            print(f"  step {step:5d}  loss {loss.item():.4f}")
    torch.save({"state": ema.module.state_dict(), "words": vocab.words[1:], "mean": mean, "std": std,
                "active": active, "train_s": time.time() - start, "steps": steps, "windows": len(X)}, CKPT)


class Controller:
    def __init__(self, device):
        ckpt = torch.load(CKPT, map_location=device, weights_only=False)
        self.ckpt, self.device = ckpt, device
        self.vocab = Vocab(ckpt["words"])
        n = int(ckpt["active"].sum())
        self.model = MotionDenoiser(n, len(self.vocab.words), cond_dim=n + 4).to(device).eval()
        self.model.load_state_dict(ckpt["state"])
        self.diffusion = GaussianDiffusion(device=device)

    def generate(self, text, path=None, prefix=None, length=WIN, seed=0, guidance=2.0):
        """Local-frame segment -> (root, angles, seconds).

        path:   (length, 2) target root (x, z) in the segment frame, or None (text only)
        prefix: (root, angles) of the first frames, in the segment frame, kept fixed
        """
        mean, std, active = self.ckpt["mean"], self.ckpt["std"], self.ckpt["active"]
        dev = self.device
        n = int(active.sum())
        xn = torch.zeros(1, length, n, device=dev)
        k = 0
        if prefix is not None:
            k = len(prefix[0])
            xn[0, :k] = torch.tensor((to_features(*prefix)[:, active] - mean[active]) / std[active], device=dev)
        traj = torch.zeros(1, length, 2, device=dev)
        if path is not None:
            traj[0] = torch.tensor(path / TRAJ_SCALE, dtype=torch.float32, device=dev)
        cond, mask = make_cond(xn, traj, torch.tensor([k], device=dev), torch.tensor([path is not None], device=dev))
        words = torch.tensor([self.vocab.encode(text)], device=dev)
        gen = torch.Generator(dev).manual_seed(seed)
        start = time.perf_counter()
        x = self.diffusion.sample(self.model, words, (1, length, n), guidance, gen, cond=cond,
                                  known=xn if k else None, mask=mask if k else None)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        feats = np.tile(mean, (length, 1))
        feats[:, active] = x[0].cpu().numpy() * std[active] + mean[active]
        root, angles = from_features(feats)
        return root, angles, seconds


# ---------- continuation and post-processing ----------

def continue_segment(ctrl, root, angles, text, world_path=None, seed=0):
    """Appends one generated segment that starts from the last PREFIX frames of (root, angles)."""
    start = len(root) - PREFIX
    r_loc, a_loc, frame = to_local(root[start:], angles[start:])
    path = None
    if world_path is not None:
        # condition = the path actually walked for the known frames, then the target's
        # future steps continued from where the character really is
        target = yaw_rotate(world_path[start:start + WIN] - frame[0], -frame[1])
        path = np.concatenate([r_loc[:, [0, 2]],
                               r_loc[-1, [0, 2]] + np.cumsum(np.diff(target[PREFIX - 1:], axis=0), 0)])
    r2, a2, seconds = ctrl.generate(text, path, prefix=(r_loc, a_loc), seed=seed)
    r2, a2 = to_world(r2, a2, frame)
    return np.concatenate([root, r2[PREFIX:]]), np.concatenate([angles, a2[PREFIX:]]), seconds


def naive_concat(ctrl, root, angles, text, world_path=None, seed=0):
    """Baseline: the next segment is generated without the past frames and placed at the end
    position / heading of the first one (same model, no past-motion conditioning)."""
    frame = (root[-1, [0, 2]], angles[-1, IDX["pelvis"], 1])
    path = None
    if world_path is not None:
        seg = world_path[len(root) - 1:len(root) - 1 + WIN]
        path = yaw_rotate(seg - frame[0], -frame[1])
    r2, a2, seconds = ctrl.generate(text, path, seed=seed)
    r2, a2 = to_world(r2, a2, frame)
    n = WIN - PREFIX
    return np.concatenate([root, r2[:n]]), np.concatenate([angles, a2[:n]]), seconds


def stance_toes(pos, contact_height=0.03, max_speed=1.0):
    """Per frame: index of the planted toe joint, or -1. Planted = the slower of the two
    toes (horizontal speed), on the ground and slower than `max_speed`. Height alone
    can't separate stance from swing here: the procedural walk barely lifts the feet."""
    toes = np.array([IDX["l_toe"], IDX["r_toe"]])
    speed = np.linalg.norm(np.gradient(pos[:, toes][..., [0, 2]], axis=0), axis=-1) * FPS
    slow = np.argmin(speed, 1)
    rows = np.arange(len(pos))
    planted = (speed[rows, slow] < max_speed) & (pos[rows, toes[slow], 1] < contact_height)
    return np.where(planted, toes[slow], -1)


def lock_feet(pos, contact_height=0.03, max_drift=0.35):
    """POST-PROCESS (not learned): game-style foot locking with two-bone IK.

    Stance frames come from `stance_toes`. While in stance, toe and ankle are pinned where
    contact began (toe on the floor, y = 0) and the knee is re-solved from hip,
    pinned ankle and the fixed thigh / shin lengths, bending like the original
    knee. The lock is released when the leg can't reach or the original foot has
    moved more than `max_drift` away. The root path is not changed.
    """
    out = pos.copy()
    stance_toe = stance_toes(pos, contact_height)
    for side in "lr":
        hip, knee, ankle, toe = (IDX[f"{side}_{j}"] for j in ("hip", "knee", "ankle", "toe"))
        thigh = np.linalg.norm(pos[0, knee] - pos[0, hip])
        shin = np.linalg.norm(pos[0, ankle] - pos[0, knee])
        anchor = None
        for t in range(len(pos)):
            if stance_toe[t] != toe:
                anchor = None
                continue
            if anchor is None:
                anchor, foot = pos[t, toe] * [1, 0, 1], pos[t, ankle] - pos[t, toe]
            d = anchor + foot - out[t, hip]
            length = np.linalg.norm(d)
            if length > thigh + shin - 1e-3 or np.linalg.norm(pos[t, toe] - anchor) > max_drift:
                anchor = None
                continue
            along = (thigh ** 2 - shin ** 2 + length ** 2) / (2 * length)
            axis = d / length
            pole = pos[t, knee] - pos[t, hip]
            pole -= axis * pole.dot(axis)
            pole /= np.linalg.norm(pole) + 1e-9
            out[t, knee] = out[t, hip] + axis * along + pole * np.sqrt(max(thigh ** 2 - along ** 2, 0.0))
            out[t, ankle], out[t, toe] = anchor + foot, anchor
    return out


# ---------- metrics ----------

def boundary_jump(pos, angles, b):
    """Discontinuity at frame b (first newly generated frame)."""
    step = np.linalg.norm(pos[1:] - pos[:-1], axis=-1).mean(-1)          # m per frame
    acc = np.linalg.norm(pos[2:] - 2 * pos[1:-1] + pos[:-2], axis=-1).mean(-1) * FPS ** 2
    yaw = np.degrees(np.abs(np.diff(angles[:, IDX["pelvis"], 1])))
    return {"pos_jump_m": float(step[b - 1]), "median_step_m": float(np.median(step)),
            "accel_spike_ms2": float(acc[b - 2:b + 1].max()), "median_accel_ms2": float(np.median(acc)),
            "yaw_jump_deg": float(yaw[b - 1]), "max_yaw_step_deg": float(yaw.max())}


def stance_sliding(pos, reference=None, fps=FPS):
    """Mean horizontal speed (m/s) of the planted toe over stance frames. Stance frames
    are detected on `reference` (default: pos itself), so a corrected clip is scored
    on the same frames as its uncorrected version."""
    stance = stance_toes(pos if reference is None else reference)
    t = np.nonzero(stance[1:] >= 0)[0] + 1
    if len(t) == 0:
        return 0.0
    step = pos[t, stance[t]][:, [0, 2]] - pos[t - 1, stance[t]][:, [0, 2]]
    return float(np.linalg.norm(step, axis=-1).mean() * fps)


def path_error(pos, path):
    return float(np.linalg.norm(pos[:, IDX["pelvis"]][:, [0, 2]] - path, axis=-1).mean())


def save_positions(name, pos, text, source):
    np.savez(os.path.join(OUT_DIR, name + ".npz"), positions=pos, parents=PARENTS, fps=FPS, text=text, source=source)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(CKPT):
        train(device)
    ctrl = Controller(device)
    report = {"train": {k: ctrl.ckpt[k] for k in ("steps", "windows", "train_s")}}
    seeds = (0, 1, 2)

    # 1) trajectory conditioning: same text, paths with different curvature
    traj_rows, paths_plot = [], []
    for name, text, rate in [("straight", "walk straight", 0.0), ("left", "turn left", 0.6),
                             ("left", "turn left", 0.95), ("right", "turn right", -0.6),
                             ("right", "turn right", -0.95)]:
        path = target_path(rate)
        row = {"text": text, "turn_rate": rate}
        for mode, p in (("text+path", path), ("text only", None)):
            errs, slides = [], []
            for s in seeds:
                r, a, _ = ctrl.generate(text, p, seed=s)
                pos = forward_kinematics(r, a)
                errs.append(path_error(pos, path))
                slides.append(foot_sliding(pos))
                if s == 0:
                    paths_plot.append((text, rate, mode, pos[:, 0][:, [0, 2]], path))
                    if mode == "text+path" and abs(rate) in (0.0, 0.95):
                        save_positions(f"traj_{name}", pos, f"{text} + path", "ctrl")
            row[mode] = {"path_error_m": float(np.mean(errs)), "foot_sliding": float(np.mean(slides))}
        traj_rows.append(row)
    report["trajectory"] = traj_rows

    # 2) continuation: 6 s = walk straight (96 frames) + turn left (84 new frames)
    frames = np.arange(2 * WIN)  # a little past 6 s: the naive segment needs a full window of path
    speed = SPEED * smoothstep(frames / FPS / 0.4)
    turn = 0.75 * smoothstep((frames - WIN) / 20)  # ease the turn in over ~0.7 s
    world_path = path_from_controls(speed, turn)
    world_path -= world_path[0]
    r1, a1, t1 = ctrl.generate("walk straight", world_path[:WIN], seed=0)
    rc, ac, t2 = continue_segment(ctrl, r1, a1, "turn left", world_path, seed=1)
    rn, an, _ = naive_concat(ctrl, r1, a1, "turn left", world_path, seed=1)
    cont, naive = forward_kinematics(rc, ac), forward_kinematics(rn, an)
    locked = lock_feet(cont)
    save_positions("continuous_1_naive", naive, "straight -> left", "naive")
    save_positions("continuous_2_inpaint", cont, "straight -> left, continued", "ctrl")
    save_positions("continuous_3_footlock", locked, "straight -> left", "footlock")
    report["continuation"] = {
        name: {**boundary_jump(pos, ang, WIN), "jerk": jerk(pos), "foot_sliding": foot_sliding(pos),
               "stance_sliding": stance_sliding(pos, cont if name.endswith("footlock") else None),
               "path_error_m": path_error(pos, world_path[:len(pos)])}
        for name, pos, ang in (("naive", naive, an), ("continuation", cont, ac), ("continuation+footlock", locked, ac))}

    # 3) latency per segment
    ctrl.generate("walk straight", target_path(0.0))
    report["latency_ms_per_segment"] = {
        str(device): 1000 * float(np.mean([ctrl.generate("walk straight", target_path(0.0), seed=s)[2] for s in seeds]))}

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    plot_paths(paths_plot, world_path, rc, rn, locked)
    print_report(report)


def plot_paths(paths_plot, world_path, rc, rn, locked):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [2, 1]})
    colors = {}
    for text, rate, mode, xz, path in paths_plot:
        c = colors.setdefault((text, rate), f"C{len(colors)}")
        if mode == "text+path":
            axes[0].plot(path[:, 0], path[:, 1], "k:", lw=1)
            axes[0].plot(xz[:, 0], xz[:, 1], color=c, lw=2, label=f"{text}, path {rate:+.2f} rad/s")
        else:
            axes[0].plot(xz[:, 0], xz[:, 1], color=c, lw=1, ls="--", alpha=0.6)
    axes[0].set_title("Trajectory conditioning\nsolid: text + path   dashed: text only   dotted: target", fontsize=9)
    axes[1].plot(world_path[:len(rc), 0], world_path[:len(rc), 1], "k:", label="target")
    axes[1].plot(rn[:, 0], rn[:, 2], color="tab:orange", label="naive concat")
    axes[1].plot(rc[:, 0], rc[:, 2], color="tab:green", lw=2, label="continuation (inpainting)")
    axes[1].plot(*rc[WIN, [0, 2]], "ko", ms=5)
    axes[1].set_title("6 s continuation: straight -> left\n(dot = segment boundary)", fontsize=9)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("x (m, + = left)")
        ax.set_ylabel("z (m, forward)")
        ax.invert_xaxis()  # character's left drawn on the left, viewed from above-behind
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "paths.png")
    fig.savefig(path, dpi=100)
    print(f"Saved {path}")


def print_report(r):
    print("\nTrajectory conditioning (mean of 3 seeds; error = mean root distance to target path):")
    print("| text | target turn (rad/s) | path error, text+path (m) | path error, text only (m) |")
    print("|---|---|---|---|")
    for row in r["trajectory"]:
        print(f"| {row['text']} | {row['turn_rate']:+.2f} | {row['text+path']['path_error_m']:.3f} "
              f"| {row['text only']['path_error_m']:.3f} |")
    print("\n6 s continuation (boundary at frame 96):")
    print("| method | joint jump at boundary (m) | median step (m) | accel spike (m/s^2) | median accel | "
          "yaw jump (deg) | max yaw step | foot sliding (m/s) | stance sliding (m/s) | path error (m) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, c in r["continuation"].items():
        print(f"| {name} | {c['pos_jump_m']:.3f} | {c['median_step_m']:.3f} | {c['accel_spike_ms2']:.1f} "
              f"| {c['median_accel_ms2']:.1f} | {c['yaw_jump_deg']:.1f} | {c['max_yaw_step_deg']:.1f} | {c['foot_sliding']:.2f} | {c['stance_sliding']:.2f} | {c['path_error_m']:.3f} |")
    print(f"\nLatency per {WIN}-frame segment: {r['latency_ms_per_segment']}  (trained {r['train']['train_s']:.0f} s)")


if __name__ == "__main__":
    main()
