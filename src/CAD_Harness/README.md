# CAD Harness

Minimal generate → execute → retry loop for LLM-to-CAD: an LLM writes
CAD Python code, the code runs in a subprocess, and on failure the
traceback is fed back into the prompt for a fix (CADCodeVerify-style,
no vision critic). Two kernel variants, same loop shape, kept in
separate subfolders since `cadquery` and `build123d` pull in
conflicting OCP builds if installed in the same venv:

- `cadquery/` — **CadQuery** variant
- `build123d/` — **build123d** variant (kernel used by `../text-to-cad`'s `cadgen` skill)

Each subfolder is self-contained:
- `harness.py` — core loop (`run_harness`), model-agnostic (`llm_fn: str -> str`)
- `qwen_llm.py` — Qwen3-8B-backed `llm_fn` (lazy-loaded, HF `transformers`)
- `test_harness.py` — tests with a fake LLM; no model/GPU required

## How it works

```
                     ┌─────────────────────────────────────┐
                     │   instruction (e.g. "a cube of       │
                     │   size 5") + error from last attempt │
                     └───────────────────┬───────────────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │  build_prompt()        │
                              │  system prompt + task  │
                              │  (+ traceback if retry) │
                              └───────────┬─────────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │  llm_fn(prompt)         │
                              │  Qwen3-8B / any LLM     │
                              └───────────┬─────────────┘
                                          │  raw text
                                          ▼
                              ┌───────────────────────┐
                              │  extract_code()         │
                              │  pull ```python fence    │
                              └───────────┬─────────────┘
                                          │  CadQuery / build123d code
                                          ▼
                              ┌───────────────────────┐
                              │  execute_cadquery() /   │
                              │  execute_build123d()    │
                              │  run in subprocess,     │
                              │  export STEP file       │
                              └───────────┬─────────────┘
                                          │
                          success ◄───────┴───────► failure
                             │                          │
                             ▼                          ▼
                   ┌───────────────────┐   ┌─────────────────────────┐
                   │ return             │   │ stderr traceback fed     │
                   │ HarnessResult      │   │ back as `error`, loop     │
                   │ (success, code,    │   │ to build_prompt() again  │
                   │  attempts, path)   │   │ (up to max_retries)       │
                   └───────────────────┘   └─────────────┬─────────────┘
                                                          │
                                                 retries exhausted
                                                          │
                                                          ▼
                                          ┌───────────────────────────┐
                                          │ return HarnessResult        │
                                          │ (success=False, last error) │
                                          └───────────────────────────┘
```

## Install

```bash
# CadQuery variant (this repo's vb_cad_venv)
pip install cadquery transformers torch

# build123d variant — prefer a SEPARATE venv from the one above;
# cadquery and build123d pull in conflicting cadquery-ocp / OCP builds
# if installed together (this repo uses build123d_venv at repo root).
pip install build123d transformers torch
```

## Run tests

```bash
cd src/CAD_Harness/cadquery && python3 -m unittest test_harness.py -v
cd src/CAD_Harness/build123d && python3 -m unittest test_harness.py -v
```

## Run with Qwen3-8B

```bash
# CadQuery — simple part
cd src/CAD_Harness/cadquery
python3 -c "
from harness import run_harness
from qwen_llm import qwen_llm
result = run_harness('a cube of size 5', qwen_llm, output_path='cube.step')
print(result)
"

# CadQuery — a slightly more complex part (tests multi-step CadQuery chaining:
# box -> face select -> hole pattern -> edge select -> fillet)
python3 -c "
from harness import run_harness
from qwen_llm import qwen_llm
result = run_harness(
    'a 40x20x5mm mounting plate with two 4mm holes 15mm either side of '
    'center and 2mm filleted corners',
    qwen_llm,
    output_path='plate.step',
)
print(result)
"

# build123d — simple part
cd src/CAD_Harness/build123d
python3 -c "
from harness import run_harness
from qwen_llm import qwen_llm
result = run_harness('a cube of size 5', qwen_llm, output_path='cube.step')
print(result)
"

# build123d — the same mounting plate, via the builder API (tests
# BuildPart -> Locations -> Hole -> edge filter -> fillet)
python3 -c "
from harness import run_harness
from qwen_llm import qwen_llm
result = run_harness(
    'a 40x20x5mm mounting plate with two 4mm holes 15mm either side of '
    'center and 2mm filleted vertical edges',
    qwen_llm,
    output_path='plate.step',
)
print(result)
"
```

## View the result

`.step` files aren't human-readable directly, and only the `cadquery/`
venv has a viewer installed — but STEP is a neutral interchange format,
so it also renders STEP output from the `build123d/` variant. Run these
from `cadquery/` (copy or point at a `build123d/*.step` file as needed).
Both work with no display server (tested headless: the VTK warning below
is harmless, the render still succeeds):

```bash
# Quick vector preview (SVG, open in VS Code/browser)
python3 -c "
import cadquery as cq
shape = cq.importers.importStep('plate.step')
cq.exporters.export(shape, 'plate.svg')
"

# Rendered PNG snapshot (works headless via screenshot=..., interact=False;
# drop interact=False for an interactive window when a real display IS available)
python3 -c "
import cadquery as cq
from cadquery.vis import show
shape = cq.importers.importStep('plate.step')
show(shape, screenshot='plate_preview.png', interact=False)
"
```

## Run with any other LLM

Pass any `callable(prompt: str) -> str` as `llm_fn` (OpenAI, Ollama, etc.) to
either variant's `run_harness` — it doesn't care where the text comes from.
