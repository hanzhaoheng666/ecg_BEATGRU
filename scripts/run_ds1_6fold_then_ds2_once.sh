#!/usr/bin/env bash
# Predeclared DS1-only six-fold selection followed by exactly one DS2 replay.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${1:?usage: GPUS=0,1,2 bash scripts/run_ds1_6fold_then_ds2_once.sh /path/to/ds1ds2_manifest [out_root]}"
OUT="${2:-$ROOT/runs/ds1_6fold_then_ds2_$(date +%Y%m%d_%H%M%S)}"
CANDIDATES="${CANDIDATES:-$ROOT/experiments/ds1_6fold_causal/candidates.json}"
GPUS="${GPUS:-0,1,2}"; IFS=',' read -r -a GPU_LIST <<< "$GPUS"
EPOCHS="${EPOCHS:-15}"; DEVICE="${DEVICE:-cuda}"; FOLDS=6
mkdir -p "$OUT"; cd "$ROOT"
python tools/make_ds1_cv_folds.py --manifest-dir "$MANIFEST" --out "$OUT/folds" --folds "$FOLDS"

run_cnn_fold() {
  local fold="$1" gpu="$2" work="$OUT/backbones/fold${fold}"
  CUDA_VISIBLE_DEVICES="$gpu" python -m beatgru.train train-cnn --manifest-dir "$MANIFEST" --record-ids-file "$OUT/folds/fold${fold}_train.txt" --out "$work" --seed "$fold" --device "$DEVICE" --epochs "$EPOCHS" --target-fs 250 --lookback-ms 350 --delay-ms 200
}
active=0
for fold in 0 1 2 3 4 5; do
  run_cnn_fold "$fold" "${GPU_LIST[$((fold % ${#GPU_LIST[@]}))]}" >"$OUT/cnn_fold${fold}.log" 2>&1 & active=$((active+1))
  if (( active >= ${#GPU_LIST[@]} )); then wait -n; active=$((active-1)); fi
done
while (( active > 0 )); do wait -n; active=$((active-1)); done

mapfile -t CONFIGS < <(python -c 'import json,sys; [print("\t".join(map(str,(c["name"],c["hidden"],c["max_correction"],c["power"],c["lr"])))) for c in json.load(open(sys.argv[1]))["candidates"]]' "$CANDIDATES")
run_gru_fold() {
  local name="$1" hidden="$2" correction="$3" power="$4" lr="$5" fold="$6" gpu="$7" work="$OUT/candidates/$name/fold$fold"
  mkdir -p "$work"
  CUDA_VISIBLE_DEVICES="$gpu" python -m beatgru.train train-gru --manifest-dir "$MANIFEST" --record-ids-file "$OUT/folds/fold${fold}_train.txt" --backbone "$OUT/backbones/fold${fold}/derivative_cnn.pt" --out "$work/gru" --seed "$fold" --device "$DEVICE" --epochs "$EPOCHS" --lr "$lr" --power "$power" --hidden "$hidden" --max-correction "$correction" --chunk 8192 --target-fs 250 --lookback-ms 350 --delay-ms 200
  CUDA_VISIBLE_DEVICES="$gpu" python -m beatgru.train evaluate --manifest-dir "$MANIFEST" --dataset mit_bih_arrhythmia --target-split train --record-ids-file "$OUT/folds/fold${fold}_val.txt" --checkpoint "$work/gru/beatgru.pt" --backbone "$OUT/backbones/fold${fold}/derivative_cnn.pt" --out "$work/validation" --seed "$fold" --device "$DEVICE" --hidden "$hidden" --max-correction "$correction" --chunk 8192 --target-fs 250 --lookback-ms 350 --delay-ms 200
}
active=0; job=0
for config in "${CONFIGS[@]}"; do
  IFS=$'\t' read -r name hidden correction power lr <<< "$config"
  for fold in 0 1 2 3 4 5; do
    run_gru_fold "$name" "$hidden" "$correction" "$power" "$lr" "$fold" "${GPU_LIST[$((job % ${#GPU_LIST[@]}))]}" >"$OUT/${name}_fold${fold}.log" 2>&1 & active=$((active+1)); job=$((job+1))
    if (( active >= ${#GPU_LIST[@]} )); then wait -n; active=$((active-1)); fi
  done
done
while (( active > 0 )); do wait -n; active=$((active-1)); done

python tools/select_ds1_cv_candidate.py --candidates "$CANDIDATES" --runs "$OUT/candidates" --out "$OUT/LOCKED_CONFIG.json" --folds "$FOLDS"
read -r NAME HIDDEN CORRECTION POWER LR < <(python -c 'import json,sys; c=json.load(open(sys.argv[1]))["selected"]["candidate"]; print(c["name"],c["hidden"],c["max_correction"],c["power"],c["lr"])' "$OUT/LOCKED_CONFIG.json")
FINAL="$OUT/final_ds1_retrain/$NAME"; mkdir -p "$FINAL"
CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" python -m beatgru.train train-cnn --manifest-dir "$MANIFEST" --out "$FINAL/backbone" --seed 0 --device "$DEVICE" --epochs "$EPOCHS" --target-fs 250 --lookback-ms 350 --delay-ms 200 >"$FINAL/cnn_train.log" 2>&1
CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" python -m beatgru.train train-gru --manifest-dir "$MANIFEST" --backbone "$FINAL/backbone/derivative_cnn.pt" --out "$FINAL/gru" --seed 0 --device "$DEVICE" --epochs "$EPOCHS" --lr "$LR" --power "$POWER" --hidden "$HIDDEN" --max-correction "$CORRECTION" --chunk 8192 --target-fs 250 --lookback-ms 350 --delay-ms 200 >"$FINAL/gru_train.log" 2>&1
CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" python -m beatgru.train evaluate --manifest-dir "$MANIFEST" --dataset mit_bih_arrhythmia --target-split test --checkpoint "$FINAL/gru/beatgru.pt" --backbone "$FINAL/backbone/derivative_cnn.pt" --out "$FINAL/DS2_FINAL_ONCE" --seed 0 --device "$DEVICE" --hidden "$HIDDEN" --max-correction "$CORRECTION" --chunk 8192 --target-fs 250 --lookback-ms 350 --delay-ms 200 >"$FINAL/ds2_final.log" 2>&1
echo "Completed DS1-only six-fold selection and one final DS2 evaluation: $FINAL/DS2_FINAL_ONCE"
