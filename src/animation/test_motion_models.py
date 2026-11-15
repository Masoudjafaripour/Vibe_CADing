"""Checks the C++ motion models (vibecad_motion.MotionModel) against PyTorch, and times both.

    python export_weights.py && python test_motion_models.py

1. AR Transformer: one forward pass on random frames, and a full 90-frame rollout.
2. Diffusion U-Net: denoiser outputs at t = 0, 50, 99, and a full 100-step sampling run with
   the same noise draws fed to both (so the comparison is exact, not statistical).
3. End to end: text -> root + joint angles, against generate_motion.Generator.
4. Speed on this CPU (and, if present, GPU).
"""
import os
import sys
import time

import numpy as np
import torch

import vibecad_motion as vm
from export_weights import WEB_DIR
from generate_motion import Generator
from motions import SEQ_LEN

TOL = {"forward": 1e-4, "rollout": 2e-3, "denoise": 1e-4, "sample": 2e-3, "clip": 2e-3}
failures = 0


def report(name, diff, tol):
    global failures
    ok = diff < tol
    failures += not ok
    print(f"  [{'pass' if ok else 'FAIL'}] {name:52s} max |C++ - torch| = {diff:.2e}  (< {tol:g})")


def timed(fn, runs=3):
    fn()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    return 1000 * (time.perf_counter() - t0) / runs


def main():
    torch.manual_seed(0)
    threads = torch.get_num_threads()
    rng = np.random.default_rng(0)
    speed = []
    for method in ("ar", "diffusion"):
        print(f"== {method}")
        gen = Generator(method, torch.device("cpu"))
        model = gen.model
        cpp = vm.MotionModel.load(os.path.join(WEB_DIR, f"{method}.vcw"))
        words = gen.vocab.encode("wave your right hand")
        assert cpp.encode("wave your right hand") == words
        W = torch.tensor([words])
        F = cpp.features
        with torch.no_grad():
            if method == "ar":
                frames = rng.standard_normal((40, F)).astype(np.float32)
                ref = model(W, torch.tensor(frames)[None])[0].numpy()
                report("forward pass (41 tokens, random frames)", np.abs(cpp.forward(words, frames) - ref).max(), TOL["forward"])
                ref = model.generate(W, SEQ_LEN)[0].numpy()
                report("90-frame autoregressive rollout", np.abs(cpp.sample(words) - ref).max(), TOL["rollout"])
                t_torch = timed(lambda: model.generate(W, SEQ_LEN))
                torch.set_num_threads(1)
                t_torch1 = timed(lambda: model.generate(W, SEQ_LEN))
                torch.set_num_threads(threads)
                t_cpp = timed(lambda: cpp.sample(words))
            else:
                diff = gen.diffusion
                x = rng.standard_normal((SEQ_LEN, F)).astype(np.float32)
                for t in (0, 50, 99):
                    ref = model(torch.tensor(x)[None], torch.tensor([t]), W)[0].numpy()
                    report(f"denoiser G(x_t, t={t}, text)", np.abs(cpp.denoise(x, t, words) - ref).max(), TOL["denoise"])
                null = [0] * len(words)
                ref = model(torch.tensor(x)[None], torch.tensor([50]), torch.tensor([null]))[0].numpy()
                report("denoiser without text (classifier-free branch)", np.abs(cpp.denoise(x, 50, null) - ref).max(), TOL["denoise"])
                # GaussianDiffusion.sample, with the noise taken from a list instead of a generator
                noise = torch.randn(diff.steps, SEQ_LEN, F)
                xt, draws = noise[0][None], iter(noise[1:])
                for i in reversed(range(diff.steps)):
                    tt = torch.full((1,), i)
                    x0c, x0u = model(torch.cat([xt, xt]), torch.cat([tt, tt]), torch.cat([W, torch.tensor([null])])).chunk(2)
                    x0 = x0u + 2.0 * (x0c - x0u)
                    if i == 0:
                        break
                    a, abar, abar_prev, beta = diff.alphas[i], diff.abar[i], diff.abar[i - 1], diff.betas[i]
                    mean = (abar_prev.sqrt() * beta * x0 + a.sqrt() * (1 - abar_prev) * xt) / (1 - abar)
                    xt = mean + (beta * (1 - abar_prev) / (1 - abar)).sqrt() * next(draws)[None]
                report("100-step sampling, same noise, guidance 2", np.abs(cpp.sample_with_noise(words, noise.numpy()) - x0[0].numpy()).max(), TOL["sample"])
                g = torch.Generator().manual_seed(0)
                t_torch = timed(lambda: diff.sample(model, W, (1, SEQ_LEN, F), 2.0, g), runs=1)
                torch.set_num_threads(1)
                t_torch1 = timed(lambda: diff.sample(model, W, (1, SEQ_LEN, F), 2.0, g), runs=1)
                torch.set_num_threads(threads)
                t_cpp = timed(lambda: cpp.sample(words), runs=1)
        # end to end: text -> root and joint angles (AR is deterministic, so the clips must agree)
        if method == "ar":
            root, angles, _ = gen("wave hello")
            out = cpp.generate("wave hello")
            report("text -> root + joint angles (\"wave hello\")", max(np.abs(out["root"] - root).max(), np.abs(out["angles"] - angles).max()), TOL["clip"])
        speed.append((method, t_torch, t_torch1, t_cpp))
    print("\nSpeed for one 3 s clip (90 frames), single sample, this CPU:")
    print(f"  {'':10s} {'PyTorch, ' + str(threads) + ' threads':>22s} {'PyTorch, 1 thread':>18s} {'C++, 1 thread':>14s}")
    for method, a, a1, b in speed:
        print(f"  {method:10s} {a:19.0f} ms {a1:15.0f} ms {b:11.0f} ms   ({a1 / b:.1f}x vs 1 thread)")
    print("all passed" if not failures else f"{failures} FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
