#!/usr/bin/env bash
# Independent strict-causal N/S/V/F extension.  Does not touch N/S/V runs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${1:?usage: bash scripts/run_nsvf_fourclass_3gpu.sh MANIFEST [OUT]}"
OUT="${2:-$ROOT/runs/nsvf_fourclass_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPUS="${GPUS:-0,1,2}"; EPOCHS="${EPOCHS:-15}"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"; mkdir -p "$OUT"; cd "$ROOT"

one_seed() {
  local seed="$1"
  local gpu="$2"
  local base="$OUT/seed${seed}"
  mkdir -p "$base"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/run_fourclass_nsvf.py train-cnn --manifest-dir "$MANIFEST" --out "$base/M1/backbone" --seed "$seed" --epochs "$EPOCHS" --device cuda >"$base/M1_train.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/run_fourclass_nsvf.py train-gru --manifest-dir "$MANIFEST" --backbone "$base/M1/backbone/fourclass_cnn.pt" --out "$base/M3/history" --seed "$seed" --epochs "$EPOCHS" --device cuda >"$base/M3_train.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/run_fourclass_nsvf.py evaluate --mode m1 --manifest-dir "$MANIFEST" --checkpoint "$base/M1/backbone/fourclass_cnn.pt" --out "$base/M1/DS2_FINAL_ONCE" --device cuda >"$base/M1_ds2.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/run_fourclass_nsvf.py evaluate --mode m3 --manifest-dir "$MANIFEST" --checkpoint "$base/M3/history/fourclass_gru.pt" --out "$base/M3/DS2_FINAL_ONCE" --device cuda >"$base/M3_ds2.log" 2>&1
}

for index in 0 1 2; do one_seed "$index" "${GPU_LIST[$index]}" >"$OUT/seed$index.log" 2>&1 & done
wait
echo "completed $OUT"
