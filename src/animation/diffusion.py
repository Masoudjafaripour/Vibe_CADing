"""Tiny text-conditioned motion diffusion model (DDPM) over whole clips.

A clip X_0 in R^{T x F} (all frames at once) is noised and denoised:

    q(X_t | X_0) = N(sqrt(abar_t) X_0, (1 - abar_t) I),   X_t = sqrt(abar_t) X_0 + sqrt(1 - abar_t) eps
    loss         = || X_0 - G_theta(X_t, t, c) ||^2       (predict the clean clip, as in MDM)
    X_K ~ N(0, I)  ->  X_{K-1}  ->  ...  ->  X_0         (ancestral sampling, posterior q(X_{t-1} | X_t, X_0_hat))

Predicting X_0 rather than eps gives smoother motion and lets geometric
losses (velocity, foot contact, physics) act on X_0_hat directly.
G_theta is a 1D temporal U-Net (convolutions over time, features as
channels). The condition c is dropped 10% of the time in training, so
sampling can use classifier-free guidance:
    X_0_hat = G(X_t, t, 0) + w (G(X_t, t, c) - G(X_t, t, 0))
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Vocab


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], -1)


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, emb_dim):
        super().__init__()
        self.norm1, self.conv1 = nn.GroupNorm(8, c_in), nn.Conv1d(c_in, c_out, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * c_out)  # FiLM: scale and shift from (t, c)
        self.norm2, self.conv2 = nn.GroupNorm(8, c_out), nn.Conv1d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(emb)[..., None].chunk(2, 1)
        h = self.norm2(h) * (1 + scale) + shift
        return self.skip(x) + self.conv2(F.silu(h))


class MotionDenoiser(nn.Module):
    """G_theta(X_t, t, c) -> X_0_hat: (B, T, F) -> (B, T, F). One down/up level, T must be even.

    `cond_dim` > 0 adds per-frame condition channels (e.g. known past frames,
    a root trajectory; see control.py), concatenated to X_t at the input.
    """

    def __init__(self, feat_dim, vocab_size, ch=128, cond_dim=0):
        super().__init__()
        self.text = nn.EmbeddingBag(vocab_size, ch, mode="mean", padding_idx=Vocab.PAD)  # all-PAD -> zeros = "no condition"
        self.time = nn.Sequential(nn.Linear(ch, ch), nn.SiLU(), nn.Linear(ch, ch))
        self.inp = nn.Conv1d(feat_dim + cond_dim, ch, 3, padding=1)
        self.enc = ResBlock(ch, ch, ch)
        self.down = nn.Conv1d(ch, 2 * ch, 4, stride=2, padding=1)
        self.mid = ResBlock(2 * ch, 2 * ch, ch)
        self.up = nn.ConvTranspose1d(2 * ch, ch, 4, stride=2, padding=1)
        self.dec = ResBlock(2 * ch, ch, ch)
        self.out = nn.Conv1d(ch, feat_dim, 3, padding=1)
        nn.init.zeros_(self.out.weight), nn.init.zeros_(self.out.bias)

    def forward(self, x, t, words, cond=None):
        if cond is not None:
            x = torch.cat([x, cond], -1)
        emb = self.time(timestep_embedding(t, self.inp.out_channels)) + self.text(words)
        h = self.enc(self.inp(x.transpose(1, 2)), emb)
        m = self.up(self.mid(self.down(h), emb))
        return self.out(self.dec(torch.cat([h, m], 1), emb)).transpose(1, 2)


class GaussianDiffusion:
    def __init__(self, steps=100, device="cpu"):
        # cosine schedule (Nichol & Dhariwal 2021)
        s = torch.linspace(0, 1, steps + 1, device=device)
        abar = torch.cos((s + 0.008) / 1.008 * math.pi / 2) ** 2
        abar = abar / abar[0]
        self.betas = (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
        self.alphas = 1 - self.betas
        self.abar = torch.cumprod(self.alphas, 0)
        self.steps = steps

    def loss(self, model, x0, words, drop_cond=0.1, cond=None):
        t = torch.randint(0, self.steps, (len(x0),), device=x0.device)
        a = self.abar[t][:, None, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * torch.randn_like(x0)
        drop = torch.rand(len(x0), device=x0.device) < drop_cond
        words = torch.where(drop[:, None], torch.full_like(words, Vocab.PAD), words)
        return F.mse_loss(model(xt, t, words, cond), x0)

    @torch.no_grad()
    def sample(self, model, words, shape, guidance=2.0, generator=None, cond=None, known=None, mask=None):
        """Ancestral sampling. With `known` (B, T, F) and `mask` (B, T, 1), frames where
        mask = 1 are fixed to `known` (inpainting): X_0_hat is overwritten there and
        X_{t-1} gets the matching noised known frames, so the boundary stays exact."""
        x = torch.randn(shape, device=words.device, generator=generator)
        null = torch.full_like(words, Vocab.PAD)
        cond2 = torch.cat([cond, cond]) if cond is not None else None
        for i in reversed(range(self.steps)):
            t = torch.full((shape[0],), i, device=x.device)
            x0_c, x0_u = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([words, null]), cond2).chunk(2)
            x0 = x0_u + guidance * (x0_c - x0_u)
            if known is not None:
                x0 = mask * known + (1 - mask) * x0
            if i == 0:
                return x0
            a, abar, abar_prev, beta = self.alphas[i], self.abar[i], self.abar[i - 1], self.betas[i]
            # posterior q(X_{t-1} | X_t, X_0_hat): mean and variance
            mean = (abar_prev.sqrt() * beta * x0 + a.sqrt() * (1 - abar_prev) * x) / (1 - abar)
            var = beta * (1 - abar_prev) / (1 - abar)
            x = mean + var.sqrt() * torch.randn(shape, device=x.device, generator=generator)
            if known is not None:
                noise = torch.randn(shape, device=x.device, generator=generator)
                x_known = abar_prev.sqrt() * known + (1 - abar_prev).sqrt() * noise
                x = mask * x_known + (1 - mask) * x
        return x
