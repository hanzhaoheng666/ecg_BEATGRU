#!/usr/bin/env bash
# Strict-causal dual-lead / embedding-history candidates, selected only in DS1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${1:?usage: GPUS=0,1,2 bash scripts/run_pair_embedding_6fold_and_frozen.sh DS1_MANIFEST EXTERNAL_MANIFEST [out_root]}"
EXTERNAL="${2:?external manifest required for frozen evaluation}"
OUT="${3:-$ROOT/runs/pair_embedding_$(date +%Y%m%d_%H%M%S)}"
CANDIDATES="$ROOT/experiments/vpair_embedding/candidates.json"
GPUS="${GPUS:-0,1,2}"; IFS=',' read -r -a GPU_LIST <<< "$GPUS"
EPOCHS="${EPOCHS:-15}"; DEVICE="${DEVICE:-cuda}"; FOLDS=6
mkdir -p "$OUT"; cd "$ROOT"
python tools/make_ds1_cv_folds.py --manifest-dir "$MANIFEST" --out "$OUT/folds" --folds "$FOLDS"

run_fold() {
  local name="$1" input="$2" swap="$3" hidden="$4" correction="$5" power="$6" lr="$7" fold="$8" gpu="$9"
  local work="$OUT/candidates/$name/fold$fold"; mkdir -p "$work"
  CUDA_VISIBLE_DEVICES="$gpu" python -m beatgru.train train-cnn --manifest-dir "$MANIFEST" --record-ids-file "$OUT/folds/fold${fold}_train.txt" --lead-selector source_matched_pair --lead-swap-prob "$swap" --out "$work/backbone" --seed "$fold" --device "$DEVICE" --epochs "$EPOCHS"
  CUDA_VISIBLE_DEVICES="$gpu" python -m beatgru.train train-gru --manifest-dir "$MANIFEST" --record-ids-file "$OUT/folds/fold${fold}_train.txt" --lead-selector source_matched_pair --history-input "$input" --backbone "$work/backbone/derivative_cnn.pt" --out "$work/gru" --seed "$fold" --device "$DEVICE" --epochs "$EPOCHS" --lr "$lr" --power "$power" --hidden "$hidden" --max-correction "$correction"
  CUDA_VISIBLE_DEVICES="$gpu" python -m beatgru.train evaluate --manifest-dir "$MANIFEST" --dataset mit_bih_arrhythmia --target-split train --record-ids-file "$OUT/folds/fold${fold}_val.txt" --lead-selector source_matched_pair --checkpoint "$work/gru/beatgru.pt" --backbone "$work/backbone/derivative_cnn.pt" --out "$work/validation" --seed "$fold" --device "$DEVICE" --hidden "$hidden" --max-correction "$correction"
}
mapfile -t CONFIGS < <(python -c 'import json,sys; [print("\t".join(map(str,(c["name"],c["history_input"],c["lead_swap_prob"],c["hidden"],c["max_correction"],c["power"],c["lr"])))) for c in json.load(open(sys.argv[1]))["candidates"]]' "$CANDIDATES")
active=0; job=0
for config in "${CONFIGS[@]}"; do
  IFS=$'\t' read -r name input swap hidden correction power lr <<< "$config"
  for fold in 0 1 2 3 4 5; do
    run_fold "$name" "$input" "$swap" "$hidden" "$correction" "$power" "$lr" "$fold" "${GPU_LIST[$((job % ${#GPU_LIST[@]}))]}" >"$OUT/${name}_fold${fold}.log" 2>&1 &
    active=$((active+1)); job=$((job+1)); if (( active >= ${#GPU_LIST[@]} )); then wait -n; active=$((active-1)); fi
  done
done
while (( active > 0 )); do wait -n; active=$((active-1)); done
python tools/select_ds1_cv_candidate.py --candidates "$CANDIDATES" --runs "$OUT/candidates" --out "$OUT/LOCKED_CONFIG.json" --folds "$FOLDS"
read -r NAME INPUT SWAP HIDDEN CORRECTION POWER LR < <(python -c 'import json,sys; c=json.load(open(sys.argv[1]))["selected"]["candidate"]; print(c["name"],c["history_input"],c["lead_swap_prob"],c["hidden"],c["max_correction"],c["power"],c["lr"])' "$OUT/LOCKED_CONFIG.json")
FINAL="$OUT/final_ds1_retrain/$NAME"; mkdir -p "$FINAL"
CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" python -m beatgru.train train-cnn --manifest-dir "$MANIFEST" --lead-selector source_matched_pair --lead-swap-prob "$SWAP" --out "$FINAL/backbone" --seed 0 --device "$DEVICE" --epochs "$EPOCHS" >"$FINAL/cnn.log" 2>&1
CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" python -m beatgru.train train-gru --manifest-dir "$MANIFEST" --lead-selector source_matched_pair --history-input "$INPUT" --backbone "$FINAL/backbone/derivative_cnn.pt" --out "$FINAL/gru" --seed 0 --device "$DEVICE" --epochs "$EPOCHS" --lr "$LR" --power "$POWER" --hidden "$HIDDEN" --max-correction "$CORRECTION" >"$FINAL/gru.log" 2>&1
CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" python -m beatgru.train evaluate --manifest-dir "$MANIFEST" --dataset mit_bih_arrhythmia --target-split test --lead-selector source_matched_pair --checkpoint "$FINAL/gru/beatgru.pt" --backbone "$FINAL/backbone/derivative_cnn.pt" --out "$FINAL/DS2_FINAL_ONCE" --seed 0 --device "$DEVICE" --hidden "$HIDDEN" --max-correction "$CORRECTION" >"$FINAL/ds2.log" 2>&1
CUDA_VISIBLE_DEVICES="${GPU_LIST[1]}" python -m beatgru.train evaluate --manifest-dir "$EXTERNAL" --dataset st_petersburg_incart --target-split all --lead-selector source_matched_pair --checkpoint "$FINAL/gru/beatgru.pt" --backbone "$FINAL/backbone/derivative_cnn.pt" --out "$FINAL/INCART_II_V1_ALL" --seed 0 --device "$DEVICE" --hidden "$HIDDEN" --max-correction "$CORRECTION" >"$FINAL/incart.log" 2>&1 & p0=$!
CUDA_VISIBLE_DEVICES="${GPU_LIST[2]}" python -m beatgru.train evaluate --manifest-dir "$EXTERNAL" --dataset mit_bih_svdb --target-split all --lead-selector source_matched_pair --checkpoint "$FINAL/gru/beatgru.pt" --backbone "$FINAL/backbone/derivative_cnn.pt" --out "$FINAL/SVDB_ECG1_ECG2_ALL" --seed 0 --device "$DEVICE" --hidden "$HIDDEN" --max-correction "$CORRECTION" >"$FINAL/svdb.log" 2>&1 & p1=$!
wait "$p0"; wait "$p1"
CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" python -m beatgru.audit --manifest-dir "$MANIFEST" --checkpoint "$FINAL/gru/beatgru.pt" --backbone "$FINAL/backbone/derivative_cnn.pt" --out "$FINAL/causality_audit.json" --device "$DEVICE" >"$FINAL/audit.log" 2>&1
echo "Completed dual-lead / embedding DS1-only selection and frozen evaluations: $FINAL"
