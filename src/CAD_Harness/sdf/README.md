# LLM → SDF (GLSL) demo

An LLM writes a 3D shape as a few hundred lines of GLSL, a **signed
distance function**, and a GPU sphere tracer renders it to PNG and a
turntable GIF. Inspired by Keenan Crane's LLM-written GLSL dragon.

<p align="center"><img src="examples/dragon_gpt4o.gif" width="320" alt="GPT-4o dragon turntable"></p>

*Written entirely by GPT-4o: "A stylized dragon made of smooth geometric
primitives", then refined with "Make the wings larger". Each step
passed on the first attempt, and the result is 1.8 KB of GLSL
([`examples/dragon_gpt4o.glsl`](examples/dragon_gpt4o.glsl)). It's
crude (floating legs, heart-shaped wings), which shows where an
unassisted LLM gets to.*

| Snowman (`examples/snowman.glsl`) | Rubber duck (`examples/duck.glsl`) |
|---|---|
| ![snowman](examples/snowman.png) | ![duck](examples/duck.png) |
| ![snowman turntable](examples/snowman.gif) | ![duck turntable](examples/duck.gif) |

Both examples are hand-written reference shapes (about 1.5 KB each); they run without an LLM.

## Background

**SDF.** `sdf(p)` returns the distance from point `p` to the surface:
negative inside, zero on the surface, positive outside. A sphere is
`length(p) - r`. Shapes combine with plain math: `min` = union, `max` =
intersection, `max(a, -b)` = subtract, `opSmoothUnion` = blended union.
The surface is never stored; it is wherever `sdf = 0`.

**Sphere tracing.** For each pixel, march a ray: at point `p` it is safe
to step forward by `sdf(p)`, since nothing is closer. Stop when the
distance is tiny (hit) or the ray leaves the scene (miss). Normals come
from the SDF gradient, and soft shadows / ambient occlusion from more
SDF queries.

| | SDF (this demo) | Mesh | B-Rep (`../cadquery`, `../build123d`) |
|---|---|---|---|
| Stores | a function (code) | triangles | exact faces, edges, NURBS |
| Size | tiny (KB of code) | large | medium |
| Smooth blends | trivial | hard | hard (fillets) |
| Exact dimensions / STEP export | no | approximate | yes |
| Good for | organic, sculpted forms | games, rendering, printing | engineering CAD |

The output here is **procedural geometry for visualization, not
engineering-grade parametric CAD**: no exact dimensions, no STEP export.

## How it works

```
text ──► llm.py ──► harness.py ──► render.py (subprocess, timeout) ──► .glsl .png .gif
         (Qwen /     prompt +        compile, sphere-trace,
          OpenAI)    extract code    checks │
               ▲                             │ error: compile log + offending lines,
               └─────── repair (0-3x) ◄──────┘ nothing visible, inside-out, off-frame
```

Generation and rendering are separate: `llm.py` only turns a prompt into
text; `render.py` never calls an LLM. The LLM writes only `float
sdf(vec3 p)` (and optionally `vec3 sdf_color(vec3 p)`); the primitives in
`sdf_lib.glsl` (`sdSphere`, `sdBox`, `sdCapsule`, `sdEllipsoid`,
`sdRoundCone`, `opSmoothUnion`, `rotX/Y/Z`, ...) are prepended
automatically and listed in the prompt.

| File | Purpose |
|---|---|
| `demo.py` | CLI: example / render / generate / refine |
| `harness.py` | prompts, code extraction, render subprocess, repair loop |
| `llm.py` | backends: local Qwen, any OpenAI-compatible API |
| `render.py` | headless renderer (also its own CLI) |
| `sdf_lib.glsl` | shared SDF primitives and operators |
| `examples/` | snowman, duck (hand-written), dragon_gpt4o (LLM) + renders |
| `test_harness.py` | 24 tests; no LLM or API key needed |

**Rendering backend:** OpenGL 3.3 fragment shader via
[moderngl](https://github.com/moderngl/moderngl), offscreen. On Linux it
uses **EGL**, so no display or X server is needed (tested: Ubuntu, NVIDIA
RTX A6000). On other platforms it falls back to the native GL context.
That path should work on Windows and macOS but is **untested**.

**Untrusted code:** GLSL cannot touch files or the network. Each render
runs in a separate process with a timeout (`--timeout`, default 60 s)
and a 64 KB code limit, so a hanging or crashing shader can't take down
the caller.

## Install

```bash
# from the repo root
uv venv sdf_venv                     # or: python -m venv sdf_venv
source sdf_venv/bin/activate
uv pip install -r src/CAD_Harness/sdf/requirements.txt   # or: pip install -r ...
cd src/CAD_Harness/sdf
```

## Commands

Outputs go to `results/` (`<name>.glsl`, `.png`, `.json` metadata, and
`.gif` with `--gif`).

```bash
# No LLM: render a bundled example (snowman | duck | dragon_gpt4o)
python demo.py --example snowman --gif

# Generate with an OpenAI-compatible API (OpenAI, vLLM, Ollama, ...)
export OPENAI_API_KEY=sk-...
python demo.py --backend openai --model <model-name> \
    --prompt "A stylized dragon made of smooth geometric primitives" --name dragon --gif

# Refine an existing shape (saves dragon_v2, dragon_v3, ...)
python demo.py --backend openai --model <model-name> \
    --refine results/dragon.glsl --instruction "Make the wings larger" --gif

# Generate with the local model (default when no OpenAI env vars are set)
python demo.py --prompt "a snowman" --model Qwen/Qwen3-8B

# Re-render any .glsl (e.g. 4-view sheet, bigger image)
python demo.py --render results/dragon.glsl --views 4 --size 512

# Tests
python -m unittest test_harness.py -v
```

Useful flags: `--retries 0-3` (LLM repair rounds, default 2), `--size`,
`--views 1|4`, `--aa` (supersampling), `--no-shadows`, `--gif-frames`,
`--out-dir`. For a non-OpenAI server, also set
`OPENAI_BASE_URL=http://host:port/v1`. `SDF_LLM_MODEL` can replace `--model`.

## Status and limitations

- **Tested end to end:** GPT-4o generate + refine via the OpenAI API
  (dragon above); examples → PNG/GIF; compile-error repair and refine
  loops (with fake LLMs in tests); local Qwen generation.
- **Small local models aren't good enough.** Qwen3-8B and
  Qwen2.5-Coder-7B produce code that compiles but gives crude or
  off-frame shapes, and they often repeat the same mistake through every
  repair round. Expect better results from a strong API model.
- The checks catch *broken* code (compile errors, empty, inside-out,
  off-frame shapes), not *wrong* shapes. A vision critic comparing the
  render with the prompt is the natural next step.
- LLM SDFs are often not exact distances, so the tracer under-steps
  (0.9×). Very non-Lipschitz code can still show holes or artifacts.
- No mesh or STEP export.
