#!/usr/bin/env bash
# Evaluate the frozen MIT-BIH BEATGRU checkpoint on external target domains.
# The manifest must contain mit_bih_svdb.csv and st_petersburg_incart.csv in
# the strict-causal V4 CSV format used by beatgru.data.load_records().
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${1:?usage: GPUS=0,1 bash scripts/evaluate_external_zero_shot.sh /path/to/external_manifest [out_root]}"
OUT="${2:-$ROOT/runs/external_zero_shot_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0,1}"; IFS=',' read -r GPU0 GPU1 <<< "$GPUS"
mkdir -p "$OUT"; cd "$ROOT"
run_one() {
  local gpu="$1" dataset="$2" lead="$3" name="$4"
  CUDA_VISIBLE_DEVICES="$gpu" python -m beatgru.train evaluate \
    --manifest-dir "$MANIFEST" --dataset "$dataset" --target-split all --lead-selector "$lead" \
    --checkpoint checkpoints/beatgru_best_seed0.pt --backbone checkpoints/derivative_cnn_seed0.pt \
    --out "$OUT/$name" --seed 0 --device "${DEVICE:-cuda}" \
    --target-fs 250 --lookback-ms 350 --delay-ms 200 --hidden 128 --max-correction 0.75 --chunk 8192
}
run_one "$GPU0" mit_bih_svdb canonical_first svdb_ecg1 & p0=$!
# The released source model was trained on MIT-BIH MLII.  INCART names lead
# II explicitly, so this is a pre-specified physical acquisition match rather
# than a target-label-selected sensitivity run.
run_one "$GPU1" st_petersburg_incart source_matched_ii incart_source_matched_ii & p1=$!
status=0; wait "$p0" || status=$?; wait "$p1" || status=$?
if [[ "$status" -ne 0 ]]; then exit "$status"; fi
echo "Completed external frozen zero-shot tests: $OUT"
