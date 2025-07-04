# Vibe-CADing

**Vibe-CADing** is a generative model (diffusion and VAE) pipeline for generating and refining 3D part designs in voxel space. The system integrates structural and aesthetic feedback to iteratively guide the generation of functional CAD-like shapes.

## Features

* **3D Voxel Generation** using a denoising diffusion probabilistic model (DDPM)
* **Custom Feedback Functions** for symmetry, balance, and design constraints
* **Refinement Loop** to improve samples based on feedback
* **Simple 3D U-Net** architecture for noise prediction
* **Synthetic Dataset Support** (cubes, spheres, etc.)

## Roadmap

1. Voxel-based dataset generation
2. DDPM training on 3D shapes
3. Feedback scoring (symmetry, CoM, etc.)
4. Guided refinement via feedback
5. Optional text-conditioning or mesh-based upgrades

## Installation

```bash
pip install torch numpy matplotlib scikit-image open3d
```

## Usage

* Run `train.py` to train the diffusion model
* Run `sample.py` to generate new 3D parts
* Visualize outputs using `visualize.py`

## Folder Structure

```
vibe-cading/
├── data/           # voxel data
├── models/         # 3D U-Net and helpers
├── diffusion/      # noise scheduler and DDPM logic
├── feedback/       # scoring functions
├── train.py
├── sample.py
├── visualize.py
└── README.md
```

## License

MIT License
