# CAD Harness

Minimal generate → execute → retry loop for LLM-to-CAD: an LLM writes
CadQuery Python code, the code runs in a subprocess, and on failure the
traceback is fed back into the prompt for a fix (CADCodeVerify-style,
no vision critic).

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
                                          │  CadQuery code
                                          ▼
                              ┌───────────────────────┐
                              │  execute_cadquery()     │
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
pip install cadquery transformers torch
```

## Run tests

```bash
cd src/CAD_Harness
python3 -m unittest test_harness.py -v
```

## Run with Qwen3-8B

```bash
cd src/CAD_Harness
python3 -c "
from harness import run_harness
from qwen_llm import qwen_llm
result = run_harness('a cube of size 5', qwen_llm, output_path='cube.step')
print(result)
"
```

## Run with any other LLM

Pass any `callable(prompt: str) -> str` as `llm_fn` (OpenAI, Ollama, etc.) —
`run_harness` doesn't care where it comes from.
