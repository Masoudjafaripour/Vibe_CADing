# SDF Harness

LLM → GLSL signed distance function → GPU render → retry on error.
Same loop as `../cadquery/` and `../build123d/`, but the shape is an
implicit surface written as GLSL (inspired by Keenan Crane's LLM-written
GLSL dragon) instead of a B-rep.

## How it works

1. The LLM writes `float sdf(vec3 p)`: negative inside, positive outside,
   shape inside the `[-1, 1]^3` box.
2. `render.py` sphere-traces it on the GPU (headless EGL) from 4 views
   into one PNG.
3. On a compile error, an empty render, or a render that fills the whole
   view (sign flipped), the error goes back to the LLM for a retry
   (up to `max_retries`).

Outputs (`<name>.glsl` + `<name>_preview.png`) go to `results/`.

| File | Purpose |
|---|---|
| `harness.py` | core loop (`run_harness`), model-agnostic (`llm_fn: str -> str`) |
| `render.py` | headless sphere-tracing renderer, also a CLI |
| `qwen_llm.py` | Qwen3-8B `llm_fn` (thinking off) |
| `test_harness.py` | tests with a fake LLM; no model needed |

**Status:** the loop works, but Qwen3-8B only catches *code* errors, and
its shapes are crude (e.g. the "mug" in `results/` is a sphere plus a
block). Next step: a vision critic (e.g. Qwen3-VL) that checks the
preview against the prompt.

## Commands

```bash
# Setup, from the repo root (needs an NVIDIA GPU with EGL)
uv venv sdf_venv
VIRTUAL_ENV=sdf_venv uv pip install moderngl numpy pillow torch transformers accelerate
source sdf_venv/bin/activate
cd src/CAD_Harness/sdf

# Tests
python -m unittest test_harness.py -v

# Generate with Qwen3-8B -> results/mug.glsl + results/mug_preview.png
python -c "
from harness import run_harness
from qwen_llm import qwen_llm
print(run_harness('a coffee mug with a handle', qwen_llm, output_path='mug.glsl'))
"

# Re-render an existing shader
python render.py results/mug.glsl results/mug_preview.png
```

Any other LLM works: pass any `callable(prompt: str) -> str` as `llm_fn`.
