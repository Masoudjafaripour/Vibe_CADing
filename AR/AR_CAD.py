# simple AR-based CAD generation
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Masked convolution (type 'A' for first layer, 'B' for others) ---

class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, mask_center, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        kH, kW = self.kernel_size
        mask = torch.ones(out_channels, in_channels, kH, kW)

        # row-major ordering: pixels above and left are allowed, right & below are masked
        yc, xc = kH // 2, kW // 2
        mask[:, :, yc+1:, :] = 0          # all rows below
        mask[:, :, yc, xc+1:] = 0         # same row, columns to the right
        if mask_center:                   # for first layer: forbid current pixel itself
            mask[:, :, yc, xc] = 0
        self.register_buffer("mask", mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

# --- Simple PixelCNN block for CAD tokens ---

class SimplePixelCAD(nn.Module):
    def __init__(self, num_tokens, hidden_channels=64, kernel_size=7, depth=5):
        """
        num_tokens: number of discrete CAD cell types K
        """
        super().__init__()
        self.num_tokens = num_tokens

        # first masked conv: cannot see current pixel ("type A")
        self.input_conv = MaskedConv2d(
            in_channels=num_tokens,  # one-hot channels
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            mask_center=True,
        )

        # stack of masked convs that can see current feature ("type B")
        self.blocks = nn.ModuleList([
            MaskedConv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                mask_center=False,
            )
            for _ in range(depth)
        ])

        # 1x1 conv → logits over tokens
        self.out_conv = nn.Conv2d(hidden_channels, num_tokens, kernel_size=1)

    def forward(self, x):
        """
        x: (B, H, W) integers in [0, K-1]
        returns logits: (B, K, H, W)
        """
        # one-hot encode tokens to "channels"
        B, H, W = x.shape
        x_onehot = F.one_hot(x, num_classes=self.num_tokens).float()  # (B, H, W, K)
        x_onehot = x_onehot.permute(0, 3, 1, 2).contiguous()          # (B, K, H, W)

        h = F.relu(self.input_conv(x_onehot))
        for blk in self.blocks:
            h = F.relu(blk(h))
        logits = self.out_conv(h)
        return logits

    @torch.no_grad()
    def sample(self, shape, device="cuda"):
        """
        shape: (B, H, W), returns samples with ints in [0, K-1]
        """
        B, H, W = shape
        x = torch.zeros(B, H, W, dtype=torch.long, device=device)

        for i in range(H):
            for j in range(W):
                logits = self.forward(x)           # (B, K, H, W)
                probs  = F.softmax(logits[:, :, i, j], dim=-1)  # (B, K)
                x[:, i, j] = torch.multinomial(probs, 1).squeeze(-1)
        return x


# --- Training on dummy CAD grid data ---
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

K = 10
H, W = 32, 32
num_samples = 1000

cad_data = torch.randint(0, K, (num_samples, H, W))  # (N, H, W)
train_dataset = TensorDataset(cad_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimplePixelCAD(num_tokens=K).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10

for epoch in range(num_epochs):
    for (cad_grid,) in train_loader:   # <<< MAIN FIX
        cad_grid = cad_grid.to(device) # (B, H, W)

        logits = model(cad_grid)       # (B, K, H, W)
        loss = F.cross_entropy(logits, cad_grid, reduction="mean")

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


# --- Sampling new CAD designs ---
import matplotlib.pyplot as plt
import numpy as np

samples = model.sample((16, H, W), device=device).cpu().numpy()

plt.figure(figsize=(6,6))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(samples[i], vmin=0, vmax=K-1)
    plt.axis("off")
plt.tight_layout()
plt.show()
