# Vibe-CADing

**Vibe-CADing** is a unified, agentic generative design pipeline that transforms natural language prompts into manufacturable CAD-like geometry. The system integrates *multiple complementary components*:

* **Voxel-based 3D generators** (Diffusion, VAE, VQ-VAE, AR models)
* **Autoregressive PixelCNN-style 2D CAD layout generator**
* **Retrieval-Augmented Generation (RAG)** over open-source CAD libraries
* **Geometric feedback and manufacturability scoring**
* **LLM-powered multi-agent orchestration** (Planner / Retriever / Generator / Critic)

These modules form an iterative, feedback-driven workflow for creating, refining, and reusing 3D CAD parts.

---

## Features

### 🔶 3D Generative Models

* Voxel-based **diffusion models (DDPM)** for coarse-to-fine 3D synthesis
* **VAE / VQ-VAE** latent compression for efficient geometry modeling
* Optional **autoregressive 3D models** for conditional voxel generation
* Simple **3D U-Net** backbone for noise prediction

### 🔷 2D CAD Autoregressive Generator

* PixelCNN-style **masked convolutions** for sequence-free CAD grid modeling
* Generates **2D token grids** (sketch-like layouts)
* Export to **SVG/DXF** or **extrude → voxel → diffusion refinement**
* Future conditioning: **text, sketches, geometry constraints**

### 🧠 Multi-Agent Reasoning (Planner/Retriever/Generator/Critic)

* Planner: extracts specs and decomposes the user's intent
* Retriever: searches open-source CAD libraries via **text, vision, and geometric embeddings**
* Generator: produces candidate geometries using 2D/3D models
* Critic: evaluates manufacturability, symmetry, CoM, balance, structural validity

### 🎯 Feedback-Guided Refinement

* Symmetry scoring
* Center-of-mass / balance metrics
* Manufacturability heuristics (thin walls, overhangs, structure)
* Iterative correction loops driven by LLM agents

### 🗂 Retrieval-Augmented CAD (RAG)

* Retrieve nearest CAD examples (voxel/mesh/layout)
* Use retrieved shapes as priors, constraints, or templates
* Supports both **semantic** (text/vision) and **geometric** embedding search

---

## Roadmap

1. Voxel dataset generation (synthetic primitives + real CAD parts)
2. Train diffusion / VAE / VQ-VAE 3D models
3. Train PixelCNN CAD layout generator
4. Implement geometric + structural feedback scoring
5. Build the multi-agent (Planner/Retriever/Generator/Critic) loop
6. Text & sketch conditioning for all generative models
7. AR CAD grid → extrusion → diffusion refinement
8. Optional mesh branch: marching cubes → mesh diffusion
9. GUI-based interactive design agent

---

## Installation

`bash\pip install torch numpy matplotlib scikit-image open3d\`

Additional dependencies for AR CAD model:
`bash\pip install torchvision\`

---

## Usage

### Diffusion / Voxel Models

* `train.py` – train diffusion model
* `sample.py` – generate new voxel parts
* `visualize.py` – render voxel grids or meshes

### AR CAD Model

* `ar_cad_train.py` – train PixelCNN CAD generator
* `ar_cad_sample.py` – generate CAD token grids
* Visualization via matplotlib / DXF / STL tools

---

## Folder Structure

```
vibe-cading/
├── data/               # voxel datasets, token grids, retrieved CAD
├── models/             # U-Net, VAE, VQ-VAE, PixelCNN
├── diffusion/          # DDPM scheduler + sampling
├── ar/                 # autoregressive CAD generator
├── feedback/           # symmetry / CoM / manufacturability scoring
├── agents/             # planner / retriever / generator / critic
├── retrieval/          # text/vision/geometry embeddings + search
├── train.py            # diffusion training
├── sample.py           # diffusion sampling
├── visualize.py        
├── ar_cad_train.py     # AR layout training
├── ar_cad_sample.py    # AR layout sampling
└── README.md
```

---

## Citation

If you use **Vibe-CADing** in research or projects, please cite:

**Plain text:**

Masoud Jafaripour. *Vibe-CADing: Conditional CAD Generation and Retrieval for a Text-to-CAD Design Pipeline*. GitHub, 2025.

**BibTeX:**
```
@misc{vibecading2025,
author       = {Jafaripour, Masoud},
title        = {Vibe-CADing: Conditional CAD Generation and Retrieval for a Text-to-CAD Design Pipeline},
year         = {2025},
howpublished = {[https://github.com/Masoudjafaripour/Vibe_CADing}](https://github.com/Masoudjafaripour/Vibe_CADing}),
}
```
## License

MIT License

---

### Maintainer

**Masoud Jafaripour**
