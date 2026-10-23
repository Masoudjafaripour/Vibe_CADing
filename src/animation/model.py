"""Tiny language-conditioned autoregressive Transformer over pose frames.

The instruction becomes one prefix token (the mean of its word
embeddings); every following token is one pose frame. With causal
attention, output k predicts frame k:

    p(x_1:T | c) = prod_t p(x_t | x_<t, c),  p = N(f_theta(x_<t, c), I)  ->  MSE loss

Swapping `head` for VQ-VAE token logits or a diffusion denoiser keeps
the conditioning and the data pipeline unchanged.
"""
import re

import torch
import torch.nn as nn


def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


class Vocab:
    PAD = 0

    def __init__(self, words):
        self.words = ["<pad>"] + sorted(set(words))
        self.index = {w: i for i, w in enumerate(self.words)}

    @classmethod
    def from_phrases(cls, phrases):
        return cls(w for p in phrases for w in tokenize(p))

    def encode(self, text, length=8):
        """Word ids, padded to `length`. Unknown words are dropped."""
        ids = [self.index[w] for w in tokenize(text) if w in self.index][:length]
        if not ids:
            raise ValueError(f"No known words in {text!r}. Known words: {', '.join(self.words[1:])}")
        return ids + [self.PAD] * (length - len(ids))


class MotionTransformer(nn.Module):
    def __init__(self, feat_dim, vocab_size, d_model=128, layers=3, heads=4, max_len=128):
        super().__init__()
        self.text = nn.EmbeddingBag(vocab_size, d_model, mode="mean", padding_idx=Vocab.PAD)
        self.inp = nn.Linear(feat_dim, d_model)
        self.pos = nn.Parameter(torch.randn(max_len + 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, feat_dim)

    def forward(self, words, frames):
        """words (B, L) ids, frames (B, T, F) -> (B, T + 1, F); output k predicts frame k."""
        n = frames.shape[1] + 1
        h = torch.cat([self.text(words)[:, None], self.inp(frames)], 1) + self.pos[:n]
        mask = nn.Transformer.generate_square_subsequent_mask(n, device=h.device)
        return self.head(self.encoder(h, mask=mask, is_causal=True))

    @torch.no_grad()
    def generate(self, words, steps):
        """Rolls out `steps` frames, feeding each prediction back in."""
        frames = torch.zeros(words.shape[0], 0, self.head.out_features, device=words.device)
        for _ in range(steps):
            frames = torch.cat([frames, self(words, frames)[:, -1:]], 1)
        return frames
