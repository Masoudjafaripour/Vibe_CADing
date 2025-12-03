# Vibe-CADing

**Vibe-CADing** is a unified generative design pipeline exploring *two complementary families* of models:

1. **Voxel-based Diffusion + VAE models** for 3D shape generation and refinement
2. **Autoregressive (AR) PixelCNN-style models** for 2D CAD layout generation (token grids that can be extruded into 3D)

Together, these components form the foundation of a "Vibe-CAD" workflow — an iterative, feedback-driven system for creating functional CAD-like geometry.

---

## Features

### 🔶 3D Generative Models

* **3D Voxel Generation** via DDPM (denoising diffusion)
* **VAE-based latent compression** for low‑resolution voxel representations
* **Simple 3D U‑Net** backbone for noise prediction

### 🔷 CAD Autoregressive Generator (NEW)

* PixelCNN‑style masked convolutions
* Generates **2D CAD layouts** as token grids
* Grids can be exported as **SVG/DXF** or **extruded to STL**
* Supports future conditioning on text, sketches, or constraints

### 🎯 Feedback‑Guided Refinement

* Symmetry scoring
* Center‑of‑mass and balance metrics
* Structural heuristics for manufacturability

### 🗂 RAG‑like Retrieval

* Retrieve the most similar CAD examples (voxel or layout)
* Use retrieved samples to prime generation or serve as constraints

---

## Roadmap

1. Voxel dataset generation (synthetic primitives + real parts)
2. DDPM training on 3D voxel shapes
3. Spatial + structural feedback scoring
4. Iterative refinement loop
5. Add text/sketch conditioning
6. Connect AR 2D CAD grid → voxel extrusion → diffusion refinement
7. Optional mesh-based branch (marching cubes → mesh diffusion)

---

## Installation

```bash
pip install torch numpy matplotlib scikit-image open3d
```

Additional dependencies for AR CAD model:

```bash
pip install torchvision
```

---

## Usage

### Diffusion / Voxel Models

* `train.py` – train the diffusion model
* `sample.py` – generate new 3D voxel parts
* `visualize.py` – render voxel grids or meshes

### AR CAD Model

* `ar_cad_train.py` – train PixelCNN‑style CAD generator
* `ar_cad_sample.py` – sample new CAD token grids
* Visualization via matplotlib or DXF/STL exporters

---

## Folder Structure

```
vibe-cading/
├── data/               # voxel data, CAD token grids
├── models/             # 3D U-Net, VAE, PixelCNN
├── diffusion/          # DDPM scheduler + sampling
├── ar/                 # autoregressive CAD generator
├── feedback/           # symmetry / CoM / structure scoring
├── train.py            # diffusion training
├── sample.py           # diffusion sampling
├── visualize.py
├── ar_cad_train.py     # AR CAD training
├── ar_cad_sample.py    # AR CAD sampling
└── README.md
```

---

## Citation

If you use **Vibe-CADing** in research or projects, please cite:

**Plain text:**

Masoud Jafaripour. *Vibe-CADing: Conditional CAD Generation and Retrieval for a Text-to-CAD Design Pipeline*. GitHub, 2025.

**BibTeX:**

@misc{vibecading2025,
author       = {Jafaripour, Masoud},
title        = {Vibe-CADing: Conditional CAD Generation and Retrieval for a Text-to-CAD Design Pipeline},
year         = {2025},
howpublished = {[https://github.com/Masoudjafaripour/Vibe_CADing}](https://github.com/Masoudjafaripour/Vibe_CADing}),
}

## License

MIT License

---

### Maintainer

**Masoud Jafaripour**
