"""Text -> 3D stick-figure motion: procedural baseline, two tiny generative models, evaluation.

    python generate_motion.py                                 # train AR + diffusion, evaluate, write outputs/
    python generate_motion.py --method diffusion --retrain    # retrain one model
    python generate_motion.py --method ar --text "wave hello" # generate one clip with a saved model
    python visualize.py outputs/*.npz                         # -> GIFs

Methods: "ar" (autoregressive Transformer, model.py) and "diffusion"
(DDPM with a 1D temporal U-Net, diffusion.py). Both learn from the same
procedural clips (motions.py) and are scored with the same metrics.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from diffusion import GaussianDiffusion, MotionDenoiser
from metrics import classify, diversity, foot_sliding, jerk, mpjpe
from model import MotionTransformer, Vocab
from motions import COMMANDS, FPS, SEQ_LEN, from_features, procedural_motion, to_features
from skeleton import PARENTS, forward_kinematics

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
METHODS = ("ar", "diffusion")
DEFAULT_STEPS = {"ar": 1500, "diffusion": 6000}
DIVERSITY_SAMPLES = 4


def ckpt_path(method):
    return os.path.join(OUT_DIR, f"{method}.pt")


def save_clip(path, root, angles, text, source):
    np.savez(path, positions=forward_kinematics(root, angles), parents=PARENTS, fps=FPS,
             text=text, source=source)


def build_dataset(clips_per_command, rng, vocab):
    feats, words = [], []
    for command, phrases in COMMANDS.items():
        for _ in range(clips_per_command):
            feats.append(to_features(*procedural_motion(command, rng)))
            words.append(vocab.encode(rng.choice(phrases)))
    return np.stack(feats).astype(np.float32), np.array(words)


def make_model(method, feat_dim, vocab_size):
    return MotionTransformer(feat_dim, vocab_size) if method == "ar" else MotionDenoiser(feat_dim, vocab_size)


def train(method, args, device):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    vocab = Vocab.from_phrases(p for phrases in COMMANDS.values() for p in phrases)
    X, W = build_dataset(args.clips, rng, vocab)
    mean, std = X.mean((0, 1)), X.std((0, 1))
    # Only model the degrees of freedom that move; the rest stay at their constant value.
    active = std > 1e-6
    Xn = torch.tensor((X[..., active] - mean[active]) / std[active], device=device)
    W = torch.tensor(W, device=device)

    steps = args.steps or DEFAULT_STEPS[method]
    model = make_model(method, int(active.sum()), len(vocab.words)).to(device)
    diffusion = GaussianDiffusion(device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    # diffusion saves an exponential moving average of the weights (standard for DDPMs;
    # it hurt the short AR run, so AR saves its final weights)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    print(f"[{method}] training on {len(X)} procedural clips, {int(active.sum())}/{X.shape[-1]} active features, "
          f"{sum(p.numel() for p in model.parameters()):,} params, {device}")
    start = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(Xn), (64,), device=device)
        x, w = Xn[idx], W[idx]
        if method == "ar":
            # noisy inputs, so the rollout can recover from its own small errors
            inp = x[:, :-1] + args.input_noise * torch.randn_like(x[:, :-1])
            loss = torch.nn.functional.mse_loss(model(w, inp), x)
        else:
            loss = diffusion.loss(model, x, w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        ema.update_parameters(model)
        if step == 1 or step % 500 == 0:
            print(f"  step {step:5d}  loss {loss.item():.4f}")
    os.makedirs(OUT_DIR, exist_ok=True)
    saved = ema.module if method == "diffusion" else model
    torch.save({"method": method, "state": saved.state_dict(), "words": vocab.words[1:], "mean": mean,
                "std": std, "active": active, "train_s": time.time() - start, "steps": steps,
                "clips": len(X), "final_loss": loss.item()}, ckpt_path(method))


class Generator:
    """A trained model plus everything needed to turn text into (root, angles)."""

    def __init__(self, method, device, guidance=2.0):
        ckpt = torch.load(ckpt_path(method), map_location=device, weights_only=False)
        self.method, self.ckpt, self.guidance = method, ckpt, guidance
        self.vocab = Vocab(ckpt["words"])
        self.model = make_model(method, int(ckpt["active"].sum()), len(self.vocab.words)).to(device)
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()
        self.diffusion = GaussianDiffusion(device=device)

    def to(self, device):
        self.model.to(device)
        self.diffusion = GaussianDiffusion(device=device)
        return self

    def __call__(self, text, seed=0):
        """text -> (root, angles, seconds)."""
        device = next(self.model.parameters()).device
        words = torch.tensor([self.vocab.encode(text)], device=device)
        n_active = int(self.ckpt["active"].sum())
        start = time.perf_counter()
        if self.method == "ar":
            x = self.model.generate(words, SEQ_LEN)
        else:
            gen = torch.Generator(device).manual_seed(seed)
            x = self.diffusion.sample(self.model, words, (1, SEQ_LEN, n_active), self.guidance, gen)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        mean, std, active = self.ckpt["mean"], self.ckpt["std"], self.ckpt["active"]
        feats = np.tile(mean, (SEQ_LEN, 1))
        feats[:, active] = x[0].cpu().numpy() * std[active] + mean[active]
        root, angles = from_features(feats)
        return root, angles, seconds


def latency_ms(generator, runs=3):
    generator("walk")  # warm-up
    return 1000 * float(np.mean([generator("walk", seed=s)[2] for s in range(runs)]))


def motion_metrics(pos):
    return {"jerk": jerk(pos), "foot_sliding": foot_sliding(pos)}


def evaluate(generators):
    report = {"commands": {}, "latency_ms_per_clip": {}, "train": {}}
    for command, phrases in COMMANDS.items():
        root, angles = procedural_motion(command)
        base = forward_kinematics(root, angles)
        save_clip(os.path.join(OUT_DIR, f"{command}_procedural.npz"), root, angles, phrases[0], "procedural")
        row = {"procedural": {"label": classify(base), **motion_metrics(base)}}
        for method, gen in generators.items():
            clips = []  # one clip per phrase, plus extra seeds of the first phrase
            for i, phrase in enumerate(phrases):
                root, angles, _ = gen(phrase, seed=i)
                clips.append((phrase, forward_kinematics(root, angles)))
                if i == 0:
                    save_clip(os.path.join(OUT_DIR, f"{command}_{method}.npz"), root, angles, phrase, method)
            seeds = [clips[0][1]] + [forward_kinematics(*gen(phrases[0], seed=100 + s)[:2])
                                     for s in range(DIVERSITY_SAMPLES - 1)]
            labels = {phrase: classify(pos) for phrase, pos in clips}
            row[method] = {
                "follows": float(np.mean([label == command for label in labels.values()])),
                "labels": labels,
                "jerk": float(np.mean([jerk(pos) for _, pos in clips])),
                "foot_sliding": float(np.mean([foot_sliding(pos) for _, pos in clips])),
                "mpjpe_to_procedural": float(np.mean([mpjpe(pos, base) for _, pos in clips])),
                "diversity": diversity(seeds),
            }
        report["commands"][command] = row
    for method, gen in generators.items():
        lat = {"cpu": latency_ms(gen.to("cpu"))}
        if torch.cuda.is_available():
            lat["cuda"] = latency_ms(gen.to("cuda"))
        report["latency_ms_per_clip"][method] = lat
        report["train"][method] = {k: gen.ckpt[k] for k in ("steps", "clips", "train_s", "final_loss")}
    return report


def print_report(report):
    methods = [m for m in METHODS if m in report["latency_ms_per_clip"]]
    print("\n| command | source | follows | jerk (m/s^3) | foot sliding (m/s) | MPJPE to proc. (m) | diversity (m) |")
    print("|---|---|---|---|---|---|---|")
    for command, row in report["commands"].items():
        p = row["procedural"]
        print(f"| {command} | procedural | {p['label'] == command} | {p['jerk']:.1f} | {p['foot_sliding']:.2f} | - | 0 |")
        for m in methods:
            r = row[m]
            print(f"| {command} | {m} | {r['follows']:.0%} | {r['jerk']:.1f} | {r['foot_sliding']:.2f} "
                  f"| {r['mpjpe_to_procedural']:.3f} | {r['diversity']:.3f} |")
    print(f"\nLatency per {SEQ_LEN}-frame clip:")
    for m in methods:
        lat = report["latency_ms_per_clip"][m]
        print(f"  {m}: " + ", ".join(f"{k} {v:.0f} ms" for k, v in lat.items())
              + f"  (trained {report['train'][m]['train_s']:.0f} s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=METHODS + ("both",), default="both")
    parser.add_argument("--text", help="generate one clip from this instruction with a saved model")
    parser.add_argument("--name", default="custom", help="output stem for --text (default: custom)")
    parser.add_argument("--seed", type=int, default=0, help="training seed, and diffusion sampling seed for --text")
    parser.add_argument("--retrain", action="store_true", help="train even if a checkpoint exists")
    parser.add_argument("--steps", type=int, help=f"training steps (default: {DEFAULT_STEPS})")
    parser.add_argument("--clips", type=int, default=200, help="procedural clips per command")
    parser.add_argument("--input-noise", type=float, default=0.05, help="AR training input noise")
    parser.add_argument("--guidance", type=float, default=2.0, help="diffusion classifier-free guidance w")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods = METHODS if args.method == "both" else (args.method,)
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.text:
        for method in methods:
            root, angles, seconds = Generator(method, device, args.guidance)(args.text, args.seed)
            path = os.path.join(OUT_DIR, f"{args.name}_{method}.npz")
            save_clip(path, root, angles, args.text, method)
            print(f"[{method}] saved {path} ({seconds * 1000:.0f} ms); "
                  f"classified as: {classify(forward_kinematics(root, angles))}")
        return

    for method in methods:
        if args.retrain or not os.path.exists(ckpt_path(method)):
            train(method, args, device)
    report = evaluate({m: Generator(m, device, args.guidance) for m in methods})
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print_report(report)


if __name__ == "__main__":
    main()
