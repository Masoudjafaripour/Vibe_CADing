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

```bash
pip install torch numpy matplotlib scikit-image open3d
```

Additional dependencies for AR CAD model:

```bash
pip install torchvision
```

> `requirements.txt` is currently empty — install per-module dependencies as listed in [Current Implementation Status](#current-implementation-status) below.

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

## Current Implementation Status

The sections above describe the target design. Below is the **actual state of the code in this repo** as of 2026-08-23, mapped to real file paths — some pieces are working end-to-end prototypes, others are stubs for future work.

| Module | File(s) | Status | Algorithm / Purpose |
|---|---|---|---|
| Text → Image → SVG | [src/pipeline/generator.py](src/pipeline/generator.py), [src/pipeline/postprocess.py](src/pipeline/postprocess.py) | ✅ Working | Stable Diffusion + ControlNet (Canny) turns a text prompt into an image; OpenCV Canny edge detection + contour tracing then vectorizes the image into an SVG. Entry point: [src/tests/run_pipeline.py](src/tests/run_pipeline.py) |
| AR 2D CAD generator | [src/AR/AR_CAD.py](src/AR/AR_CAD.py) | ✅ Working demo | PixelCNN-style masked convolutions (type A/B masks) model `p(grid) = Π p(cell \| cells above & left)` over a 2D grid of discrete CAD cell tokens, trained here on random dummy data; samples new grids autoregressively cell-by-cell |
| LLM → B-Rep (text-to-CAD) | [src/B-rep/LLM_B_rep.py](src/B-rep/LLM_B_rep.py) | ✅ Working demo | Qwen2.5-3B-Instruct converts a text prompt into a structured JSON CAD spec (e.g. `{"type":"cube","size":5}`); pythonOCC (OpenCascade) builds sketch → wire → face → extruded solid B-rep and displays it |
| B-Rep primitive example | [src/B-rep/example_B_rep.py](src/B-rep/example_B_rep.py) | ✅ Working demo | Hand-written pythonOCC pipeline: rectangle sketch → wire → face → prism extrusion → export to `.step`/`.igs` (sample outputs in [src/B-rep/results/](src/B-rep/results/)) |
| CLIP text/image retrieval | [src/CLIP/text2cad_clip.py](src/CLIP/text2cad_clip.py) | ✅ Working demo | Encodes a text query and rendered mesh images with CLIP ViT-B/32, ranks meshes by cosine similarity for retrieval-augmented CAD reuse |
| Multi-agent orchestration | [src/Agentic/multi_agent_cad.py](src/Agentic/multi_agent_cad.py) | 🚧 Prototype (not runnable as-is) | LangChain + LangGraph Planner → Retriever → Generator → Critic loop with a conditional edge that loops back to the planner until the critic approves; uses outdated LangChain/LangGraph APIs and needs a pre-built FAISS index (`cad_index`) |
| B-Rep RAG retrieval | [src/B-rep/B_rep_Retrieval.py](src/B-rep/B_rep_Retrieval.py) | 📝 Empty stub | Intended: encode B-rep topology/geometry, store embeddings, retrieve similar CAD models |
| CVAE model | [src/CVAE/model_cvae.py](src/CVAE/model_cvae.py) | 📝 Empty stub | Conditional VAE for latent 3D geometry compression |
| Diffusion backbone | [src/models/diffusion.py](src/models/diffusion.py) | 📝 Empty stub | Planned 3D U-Net / DDPM wrapper for voxel diffusion |
| AR training script | [src/AR/train_text2cad_ar.py](src/AR/train_text2cad_ar.py) | 📝 Empty stub | Planned text-conditioned AR CAD training loop |
| Pipeline validator/retrainer | [src/pipeline/validator.py](src/pipeline/validator.py), [src/pipeline/retrainer.py](src/pipeline/retrainer.py) | 📝 Empty stub | Planned manufacturability validation and feedback-driven retraining |
| Web app | [src/webapp/interface.py](src/webapp/interface.py) (Gradio), [src/webapp/main.py](src/webapp/main.py) (FastAPI) | 📝 Empty stub | Planned web UI / API for the pipeline |
| Streamlit demo | [src/vibe-cading-demo/streamlit_app.py](src/vibe-cading-demo/streamlit_app.py), [visualize.py](src/vibe-cading-demo/visualize.py) | 🚧 Prototype (not runnable as-is) | Streamlit UI that loads canned `.npy` voxel grids (files not included) and renders them with matplotlib `ax.voxels`; `visualize.py` is missing its `streamlit` import |
| Notebooks | [src/notebooks/](src/notebooks/) | 📝 Empty | Placeholders for diffusion testing and SVG conversion experiments |
| Data | [data/prompts.jsonl](data/prompts.jsonl) | 📝 Empty | Placeholder for prompt dataset |
| DVC pipeline | [dvc.yaml](dvc.yaml) | ✅ Present | Single `generate` stage running `python run_pipeline.py` |

## Commands to Run

```bash
# 1. Text → Image → SVG pipeline (needs: torch, diffusers, pillow, opencv-python, svgwrite; GPU recommended)
python -m src.tests.run_pipeline

# 2. Autoregressive PixelCNN CAD grid generator (needs: torch, matplotlib)
python src/AR/AR_CAD.py

# 3. LLM (Qwen2.5) → CAD spec → B-Rep solid (needs: torch, transformers, pythonocc-core)
python src/B-rep/LLM_B_rep.py

# 4. B-Rep primitive example → STEP/IGES export (needs: pythonocc-core)
python src/B-rep/example_B_rep.py

# 5. CLIP-based text/mesh retrieval (needs: torch, clip, pillow)
python src/CLIP/text2cad_clip.py

# DVC-tracked pipeline stage
dvc repro generate
```

Docker: the [Dockerfile](Dockerfile) sets up a `python:3.10` base image with `/app` as the working directory (dependency install and entrypoint are not yet defined — add them if containerizing).

For the B-Rep + LLM sub-pipeline specifically, see [src/B-rep/README.md](src/B-rep/README.md) for its own architecture notes.

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
