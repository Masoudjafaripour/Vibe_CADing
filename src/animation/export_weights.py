"""Writes the trained motion models as .vcw files for the C++ inference engine
(src/geometry_engine/src/motion_models.cpp), which also runs in the browser.

    python export_weights.py            # outputs/{ar,diffusion}.pt -> website/public/motion/{ar,diffusion}.vcw

Format: b"VCW1", uint32 header length, JSON header, zero padding to 4 bytes, float32 data.
The header holds the method, vocabulary, feature normalization, skeleton, layer sizes and,
for every tensor, its shape and offset (in floats) into the data.
"""
import json
import os
import struct

import numpy as np
import torch

from generate_motion import OUT_DIR, ckpt_path
from motions import FPS, SEQ_LEN
from skeleton import NAMES, OFFSETS, PARENTS

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "..", "..", "website", "public", "motion")


def export(method, out_path):
    ckpt = torch.load(ckpt_path(method), map_location="cpu", weights_only=False)
    state = ckpt["state"]
    tensors, blobs, offset = {}, [], 0
    for name, t in state.items():
        a = t.detach().float().contiguous().numpy()
        tensors[name] = {"shape": list(a.shape), "offset": offset}
        blobs.append(a.ravel())
        offset += a.size
    if method == "ar":
        d = state["inp.weight"].shape[0]
        layers = len({k.split(".")[2] for k in state if k.startswith("encoder.layers.")})
        config = {"d_model": d, "layers": layers, "heads": 4, "ff": state["encoder.layers.0.linear1.weight"].shape[0]}
    else:
        config = {"ch": state["inp.weight"].shape[0], "steps": 100}
    header = {
        "method": method, "words": ckpt["words"], "seq_len": SEQ_LEN, "fps": FPS, "config": config,
        "mean": np.asarray(ckpt["mean"], float).tolist(), "std": np.asarray(ckpt["std"], float).tolist(),
        "active": [bool(a) for a in ckpt["active"]],
        "skeleton": {"names": NAMES, "parents": PARENTS.tolist(), "offsets": OFFSETS.tolist()},
        "tensors": tensors,
    }
    text = json.dumps(header).encode()
    pad = (-(8 + len(text))) % 4
    with open(out_path, "wb") as f:
        f.write(b"VCW1" + struct.pack("<I", len(text)) + text + b"\0" * pad)
        f.write(np.concatenate(blobs).astype("<f4").tobytes())
    return offset


def main():
    os.makedirs(WEB_DIR, exist_ok=True)
    for method in ("ar", "diffusion"):
        path = os.path.join(WEB_DIR, f"{method}.vcw")
        n = export(method, path)
        print(f"{method}: {n:,} parameters -> {os.path.relpath(path)} ({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
