# Live CAD v2 ("Fast", beta)

A faster, lighter pipeline for the website's "Try it live" section, built **next to** v1 ([`../live_cad/`](../live_cad/)),
which is unchanged. A switch above the prompt box picks **Standard** (v1, the default) or **Fast · beta** (v2). The
choice is remembered in the browser. Deleting v2 means deleting this folder and
[`website/src/scripts/fast/`](../../website/src/scripts/fast/), plus the switch in `TryLive.astro`.

## System design

```
                         v1 (Standard)                                  v2 (Fast)
                         ─────────────                                  ─────────
request "M8 nut"
   │
   ├─ prompt ─────────── every family (15) + free-form parts       C++ select_families(request, k=4)
   │                     language + 3 examples                     → hex_nut, bolt, star_knob, washer
   │                     ≈ 2,860 tokens                              + 2 examples          ≈ 830 tokens
   │
   ├─ schema ─────────── 15 family branches + custom parts         only the offered families
   │
   ├─ model ──────────── chosen in the menu                        suggested per device (WebGPU adapter,
   │                     context window 4,096                      deviceMemory): 4B or 1.7B; context 2,048
   │                                                               warm-up on text-box focus (cached model only)
   │
   ├─ while generating ─ 3D viewers keep drawing at 60 fps         viewers pause ("vibecad-gpu-busy" event)
   │
   ├─ C++ engine ─────── main thread (page pauses 0.3 to 0.5 s)    Web Worker (page stays responsive)
   │
   └─ repair loop, checks, STL, edits, library ───────── same in both ─────────────────────────────────►
```

```
┌──────────────── browser tab ────────────────────────────────────────────────────────────────┐
│  TryLive.astro (mode switch)                                                                │
│     │ v2                                                                                    │
│     ▼                                                                                       │
│  fast/live-cad-fast.ts ── selectFamilies / familiesGuide / build ──► fast/geometry-worker.ts│
│     │                                                                  └─ vibecad.wasm (C++)│
│     ├── messages + per-request schema ──► llm-worker.ts ──► WebLLM (WebGPU, Qwen3 q4f16)    │
│     │                                       context_window_size 2048                        │
│     ├── fast/device.ts: adapter info → 4B or 1.7B                                           │
│     └── window event "vibecad-gpu-busy" ──► mesh-viewer.ts pauses drawing                   │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Choosing the families** ([`families.cpp`](../geometry_engine/src/families.cpp) `rank_families`). Each family gets
a score: the sum, over the request's words, of the word's weight in that family's vocabulary. Name and synonyms
("Also: coat hook, …") weigh 3, description words 1, parameter names 0.5. Words are lower-cased and
singularized, and an "M8"-style size adds 2 to families with a `thread` parameter. The top 4 are offered. When
editing, the edited part's family is always offered. If no word matches, every family is offered.

## Results

Same prompts, checks and repair loop as v1 ([`eval_fast.py`](eval_fast.py)), 4-bit weights as in the browser
(`EVAL_Q4=1`):

| | v1 Qwen3-4B | **v2 Qwen3-4B** | v1 Qwen3-1.7B | v2 Qwen3-1.7B |
|---|---|---|---|---|
| Prompt length | 2,864 tokens | **828** (3.5× shorter) | 2,864 | 828 |
| Website suggestions | 12/12 | 12/12 | 12/12 | 12/12 |
| New wording (held-out) | 11/12 | 10/12 | 9/12 | 9/12 |
| Edits ("make it M16") | 8/8 | 8/8 | 6/8 | 6/8 |
| Right family offered (top 4) | n/a | 24/24 | n/a | 24/24 |

- **Accuracy:** v2 matches v1 except one held-out prompt on the 4B model, which is within the noise of 12 prompts.
  Both v2 misses put a size in the wrong parameter: "100 mm across" went to the bolt circle instead of the
  diameter, and "60 mm legs" set only one leg. The prompt was **not** tuned on these held-out prompts; that would
  make the held-out score meaningless.
- **Ablation:** dropping the `"name"` field (a few tokens shorter) cost one more held-out prompt on the 4B model
  (9/12): writing a name before the numbers helps the model place them. v2 keeps the name. Results are in
  `eval_results/` and `eval_results/no_name/`.
- **Scope:** v2 has no free-form ("custom") parts. Every answer is one of the 15 families. v1 still offers custom
  parts, which a 4B model rarely gets right.
- **Assemblies** use v1's assembly prompt in both modes.

**GPU memory.** The KV cache WebLLM reserves is layers × 2 (K, V) × KV heads × head size × 2 bytes per token
(from the models' configs):

| Model | per token | context 4,096 (v1) | context 2,048 (v2) | saved |
|---|---|---|---|---|
| Qwen3-4B | 0.147 MB | 604 MB | 302 MB | **0.30 GB** |
| Qwen3-1.7B | 0.115 MB | 470 MB | 235 MB | **0.24 GB** |

On top of that, picking the 1.7B model on integrated or small GPUs (about 2 GB instead of 3.4 GB in total) saves
about 1.4 GB there.

**Waiting time.** The prompt the model must read before its first token is 3.5× shorter, and warm-up hides model
loading when the model is cached. Real time-to-first-token depends on the visitor's GPU and couldn't be measured
here (this machine's headless browser has no usable WebGPU GPU). The page shows it after each run in both modes:
v1 "first token after … s", and v2 also the prompt token count from WebLLM's usage stats. Compare the two in
Chrome on a real machine.

## Files

| File | Role |
|---|---|
| [`prompt_fast.json`](prompt_fast.json) | v2 system prompt template, examples, edit template, k |
| [`fast_prompt.py`](fast_prompt.py) | builds v2 messages and the per-request schema (the TypeScript does the same) |
| [`eval_fast.py`](eval_fast.py) | v2 evaluation against v1's prompts and checks |
| [`../geometry_engine/src/families.cpp`](../geometry_engine/src/families.cpp) | `rank_families`, `select_families`, `families_guide` (C++, also in WebAssembly) |
| [`website/src/scripts/fast/live-cad-fast.ts`](../../website/src/scripts/fast/live-cad-fast.ts) | v2 pipeline: prompt, model (context 2,048), warm-up, repair loop |
| [`website/src/scripts/fast/geometry-worker.ts`](../../website/src/scripts/fast/geometry-worker.ts) | the C++ engine in a Web Worker |
| [`website/src/scripts/fast/device.ts`](../../website/src/scripts/fast/device.ts) | 4B or 1.7B from the WebGPU adapter and device memory |

```bash
EVAL_Q4=1 python src/live_cad_fast/eval_fast.py Qwen/Qwen3-4B Qwen/Qwen3-1.7B
```

## Not done yet

- A real-GPU timing comparison of v1 and v2 (time to first token, total time, GPU memory).
- The device heuristic is simple (vendor or architecture names, `deviceMemory`, buffer limits). It needs checking
  on a few real laptops.
- v2 for assemblies: the assembly prompt (about 1,100 tokens) is already short.
