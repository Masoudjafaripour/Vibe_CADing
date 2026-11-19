# Quantization study: accuracy, GPU memory and speed on the live-CAD task

How far can Qwen3-4B and Qwen3-1.7B be quantized before the live-CAD model gets worse, and what does each step
save? Every quantization runs the **same evaluation** as [`../live_cad/eval_models.py`](../live_cad/eval_models.py):
the v1 prompt and JSON schema (llama.cpp turns the schema into a grammar), the repair loop and the C++ checks.

- **Suggestions:** the 12 website prompts. The part must match (counts, genus, size).
- **Held-out:** 12 prompts in other wording. The family and every stated number must be right.
- **Edits:** 8 changes to recorded parts ("make it M16").

Setup: llama.cpp (commit `edec462`, CUDA, one RTX A6000), unsloth's GGUF files, context 4,096, temperature 0.
"VRAM" is the GPU memory the server took, including the KV cache and compute buffers.

## Results

**Qwen3-4B**

| Quantization | File | VRAM | Suggestions | Held-out | Edits | Read tok/s | Write tok/s |
|---|---|---|---|---|---|---|---|
| BF16 (reference) | 8.05 GB | 8.42 GB | 12/12 | 12/12 | 8/8 | 7,885 | 65 |
| Q8_0 | 4.28 GB | 4.92 GB | 12/12 | 12/12 | 8/8 | 7,590 | 99 |
| Q6_K | 3.31 GB | 4.01 GB | 12/12 | 12/12 | 8/8 | 6,823 | 116 |
| Q5_K_M | 2.89 GB | 3.62 GB | 12/12 | 12/12 | 8/8 | 7,091 | 124 |
| **Q4_K_M** | **2.50 GB** | **3.26 GB** | 12/12 | 12/12 | 8/8 | 7,457 | 140 |
| **Q3_K_M** | **2.08 GB** | **2.86 GB** | 12/12 | 12/12 | 8/8 | 6,474 | 141 |
| Q2_K | 1.67 GB | 2.48 GB | 12/12 | **8/12** | **6/8** | 5,558 | 152 |
| IQ2_M | 1.53 GB | 2.36 GB | 10/12 | 6/12 | 7/8 | 6,276 | 148 |
| IQ1_M | 1.14 GB | 2.00 GB | 0/12 | 0/12 | 0/8 | 5,863 | 124 |

**Qwen3-1.7B**

| Quantization | File | VRAM | Suggestions | Held-out | Edits | Read tok/s | Write tok/s |
|---|---|---|---|---|---|---|---|
| BF16 (reference) | 3.45 GB | 3.99 GB | 10/12 | 10/12 | 4/8 | 10,899 | 128 |
| Q8_0 | 1.83 GB | 2.50 GB | 10/12 | 10/12 | 5/8 | 11,276 | 184 |
| Q6_K | 1.42 GB | 2.11 GB | 10/12 | 10/12 | 5/8 | 12,173 | 215 |
| Q5_K_M | 1.26 GB | 1.96 GB | 10/12 | 11/12 | 3/8 | 11,540 | 221 |
| Q4_K_M | 1.11 GB | 1.82 GB | 11/12 | 9/12 | 3/8 | 12,730 | 228 |
| Q3_K_M | 0.94 GB | 1.66 GB | 11/12 | 7/12 | 0/8 | 12,421 | 225 |
| Q2_K | 0.78 GB | 1.51 GB | 12/12 | 8/12 | 4/8 | 10,839 | 227 |
| IQ2_M | 0.71 GB | 1.45 GB | 11/12 | 8/12 | 3/8 | 12,039 | 228 |
| IQ1_M | 0.56 GB | 1.31 GB | 0/12 | 0/12 | 0/8 | 10,199 | 189 |

**In the browser on CPU** (llama.cpp's WebAssembly build, [wllama](https://github.com/ngxson/wllama) 3.8.1, headless
Chromium, 8 threads, multi-threaded; the v2 prompt, 844 tokens):

| Model | Read tok/s | Write tok/s | One part | Native llama.cpp, same 8 threads |
|---|---|---|---|---|
| Qwen3-1.7B Q4_K_M | 17.9 | 6.7 | **51 s** | 183 read / 39 write tok/s |
| Qwen3-4B Q4_K_M | 6.6 | 2.4 | **138 s** | 61 read / 10 write tok/s |

Both answers were correct.

## What it means

1. **Qwen3-4B holds up down to 3 bits.** Q8 through Q3_K_M all score 12/12, 12/12, 8/8. Q2_K is where it breaks
   (held-out 8/12, edits 6/8), and 1-bit fails completely. Q3_K_M is 17% smaller than Q4_K_M (2.08 vs 2.50 GB)
   and uses 0.4 GB less GPU memory, at no measured accuracy cost.
2. **Qwen3-1.7B is the weak link, not its quantization.** Even unquantized it misses suggestions and half the
   edits, and below 4 bits edits drop further (Q3_K_M: 0/8). It is a "lighter, less accurate" option only.
3. **Speed on a GPU:** reading the prompt is about the same at every level (compute-bound, ~6,000 to 8,000 tok/s
   on the A6000). Writing gets faster as the model shrinks (65 tok/s BF16 to ~140 tok/s at 4 bits), because it is
   limited by memory bandwidth. Below 4 bits there is little extra speed.
4. **llama.cpp in the browser on CPU is not a usable fallback:** 51 to 138 s per part, 6 to 10× slower than the
   same code natively (WebAssembly has 128-bit SIMD only, no AVX). Visitors without WebGPU are better served by
   the recorded answers.
5. **llama.cpp Q4_K_M ≈ WebLLM q4f16:** 12/12 held-out here vs 11/12 for WebLLM's 4-bit format (simulated in
   `../live_cad/eval_models.py`), within the noise of 12 prompts.

## WebLLM's 3-bit format is not llama.cpp's

Finding 1 suggested a 3-bit Qwen3-4B for the website. WebLLM's 3-bit format is MLC's `q3f16_1`: plain symmetric
3-bit integers (7 levels) in groups of 40 with one fp16 scale. llama.cpp's Q3_K_M mixes 3-, 4- and 5-bit tensors
with finer scales. Simulating `q3f16_1` exactly in the evaluation (`EVAL_Q3=1`, partial groups zero-padded as MLC
does) shows a clear loss:

| Qwen3-4B | 4-bit `q4f16_1` (on the site) | 3-bit `q3f16_1` |
|---|---|---|
| v1 prompt: suggestions · held-out · edits | 12/12 · 11/12 · 8/8 | 9/12 · **6/12** · **4/8** |
| v2 prompt: suggestions · held-out · edits | 12/12 · 10/12 · 8/8 | 11/12 · **7/12** · **5/8** |

So the website keeps 4-bit. A smaller download would need a better 3-bit method in WebLLM (for example AWQ-style
calibrated scales), not plain group quantization. Building one would also need the exact compiler WebLLM 0.2.87
was built with (TVM `e345029`, MLC-LLM `44fd06d`; the newest prebuilt Linux MLC compiler is 0.26, while the
runtime is 0.28).

## Not done

- **WebLLM speed and memory in a real browser GPU** (q4f16 vs q4f32, and the 1.7B model): needs a machine with a
  real GPU in Chrome.
- 12 + 12 + 8 prompts is a small evaluation: differences of one prompt are noise.

## Reproduce

```bash
# llama.cpp with CUDA (no web UI)
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_UI_GZIP=OFF
cmake --build build -j --target llama-server llama-bench llama-gguf-split
# GGUF files from huggingface.co/unsloth/Qwen3-4B-GGUF and unsloth/Qwen3-1.7B-GGUF
python src/quant_study/eval_llamacpp.py --gpu 0 --models Qwen3-4B-Q4_K_M.gguf Qwen3-4B-Q3_K_M.gguf
# browser CPU: wllama/ (npm i @wllama/wllama@3.8.1; split files over 2 GB with llama-gguf-split)
python src/quant_study/wllama/serve.py 8099 & python src/quant_study/wllama/run_bench.py Qwen3-1.7B-Q4_K_M.gguf 8 v2
```
