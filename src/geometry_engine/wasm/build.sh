#!/usr/bin/env bash
# Builds the geometry engine to WebAssembly for the website (website/public/wasm/).
# Needs Emscripten: https://emscripten.org/docs/getting_started/downloads.html
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=../../website/public/wasm
mkdir -p "$OUT"
em++ -O3 -std=c++17 -Iinclude -fwasm-exceptions -Wno-unknown-pragmas \
  src/marching_cubes.cpp src/mesh_stats.cpp src/printability.cpp src/simplify.cpp src/mesh_ops.cpp \
  src/json.cpp src/csg.cpp src/families.cpp wasm/vibecad_wasm.cpp \
  -lembind -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createVibecad -sENVIRONMENT=web,worker \
  -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=64MB -sFILESYSTEM=0 \
  -o "$OUT/vibecad.mjs"
ls -la "$OUT"
