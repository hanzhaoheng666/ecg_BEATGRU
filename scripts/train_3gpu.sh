#!/usr/bin/env bash
# Usage: GPUS=0,1,2 bash scripts/train_3gpu.sh /path/to/manifest [output_root]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${1:?usage: GPUS=0,1,2 bash scripts/train_3gpu.sh /path/to/manifest [output_root]}"
OUT="${2:-$ROOT/runs/causal_ds1ds2_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0,1,2}"; IFS=',' read -r GPU0 GPU1 GPU2 <<< "$GPUS"
DEVICE="${DEVICE:-cuda}"
mkdir -p "$OUT"
cd "$ROOT"

CUDA_VISIBLE_DEVICES="$GPU0" python -m beatgru.train train-cnn --manifest-dir "$MANIFEST" --out "$OUT/backbone" --seed 0 --device "$DEVICE" |& tee "$OUT/backbone.log"
BACKBONE="$OUT/backbone/derivative_cnn.pt"
run_seed() {
  local gpu="$1" seed="$2" dest="$OUT/gru_seed${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" python -m beatgru.train train-gru --manifest-dir "$MANIFEST" --backbone "$BACKBONE" --out "$dest" --seed "$seed" --device "$DEVICE" |& tee "$OUT/gru_seed${seed}.log"
  CUDA_VISIBLE_DEVICES="$gpu" python -m beatgru.train evaluate --manifest-dir "$MANIFEST" --checkpoint "$dest/beatgru.pt" --out "$dest/test" --seed "$seed" --device "$DEVICE" |& tee "$OUT/gru_seed${seed}.test.log"
}
run_seed "$GPU0" 0 & p0=$!
run_seed "$GPU1" 1 & p1=$!
run_seed "$GPU2" 2 & p2=$!
status=0; wait "$p0" || status=$?; wait "$p1" || status=$?; wait "$p2" || status=$?
if [[ "$status" -ne 0 ]]; then exit "$status"; fi
echo "Completed: $OUT"
