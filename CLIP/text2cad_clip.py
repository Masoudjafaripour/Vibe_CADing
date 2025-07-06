import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

# Encode text query
text = clip.tokenize(["a red chair with armrests"]).to(device)
with torch.no_grad():
    text_feat = model.encode_text(text)
    text_feat /= text_feat.norm(dim=-1, keepdim=True)  # Normalize



from PIL import Image
import os
import numpy as np

def encode_mesh(folder_path):
    image_feats = []
    for fname in os.listdir(folder_path):
        if not fname.endswith(".png"): continue
        img = preprocess(Image.open(os.path.join(folder_path, fname))).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img)
            feat /= feat.norm(dim=-1, keepdim=True)
            image_feats.append(feat)
    return torch.mean(torch.stack(image_feats), dim=0)  # Mean-pooled feature

# Encode all mesh folders
mesh_features = []
mesh_names = []
for mesh_dir in sorted(os.listdir("data/")):
    folder = os.path.join("data/", mesh_dir)
    if os.path.isdir(folder):
        feat = encode_mesh(folder)
        mesh_features.append(feat)
        mesh_names.append(mesh_dir)

mesh_features = torch.cat(mesh_features, dim=0)  # Shape: (n_meshes, 512)


# text_feat: (1, 512), mesh_features: (n, 512)
similarities = (mesh_features @ text_feat.T).squeeze()  # Cosine sim
top_k = similarities.topk(5)

for i in range(5):
    idx = top_k.indices[i].item()
    score = top_k.values[i].item()
    print(f"{i+1}. {mesh_names[idx]} - Score: {score:.3f}")
