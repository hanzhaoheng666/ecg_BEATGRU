#!/usr/bin/env bash
# Evaluate the bundled seed-0 release checkpoint after building a local manifest.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${1:?usage: bash scripts/evaluate_release_checkpoint.sh /path/to/manifest [out_dir]}"
OUT="${2:-$ROOT/runs/release_seed0_evaluation}"
cd "$ROOT"
CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" python -m beatgru.train evaluate \
  --manifest-dir "$MANIFEST" --checkpoint checkpoints/beatgru_best_seed0.pt \
  --backbone checkpoints/derivative_cnn_seed0.pt --out "$OUT" --seed 0 \
  --device "${DEVICE:-cuda}" --target-fs 250 --lookback-ms 350 --delay-ms 200 \
  --hidden 128 --max-correction 0.75 --chunk 8192
