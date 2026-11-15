#!/usr/bin/env bash
# Builds the text-to-motion models (C++ inference) to WebAssembly for the website (website/public/wasm/).
# Weights: python src/animation/export_weights.py -> website/public/motion/*.vcw
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=../../website/public/wasm
mkdir -p "$OUT"
em++ -O3 -std=c++17 -Iinclude -msimd128 -fwasm-exceptions -Wno-unknown-pragmas \
  src/nn.cpp src/motion_models.cpp src/motion.cpp src/bvh.cpp src/json.cpp wasm/motion_wasm.cpp \
  -lembind -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createMotion -sENVIRONMENT=web,worker \
  -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=64MB -sFILESYSTEM=0 \
  -o "$OUT/motion.mjs"
ls -la "$OUT"/motion.*
