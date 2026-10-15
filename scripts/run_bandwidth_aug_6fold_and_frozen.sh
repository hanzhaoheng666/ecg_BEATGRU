#!/usr/bin/env bash
# One predeclared 128-Hz bandwidth-augmentation experiment; no threshold search.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${1:?usage: GPUS=0,1,2 bash scripts/run_bandwidth_aug_6fold_and_frozen.sh /path/to/ds1ds2_manifest /path/to/external_manifest [out_root]}"
EXTERNAL="${2:?external strict-causal manifest required}"
OUT="${3:-$ROOT/runs/bandwidth128_aug_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0,1,2}"; IFS=',' read -r -a GPU_LIST <<< "$GPUS"
EPOCHS="${EPOCHS:-15}"; DEVICE="${DEVICE:-cuda}"; PROB="${BANDWIDTH_AUG_PROB:-0.5}"
mkdir -p "$OUT"; cd "$ROOT"
python tools/make_ds1_cv_folds.py --manifest-dir "$MANIFEST" --out "$OUT/folds" --folds 6
run_fold() {
  local fold="$1" gpu="$2" work="$OUT/fold${fold}"
  CUDA_VISIBLE_DEVICES="$gpu" python -m beatgru.train train-cnn --manifest-dir "$MANIFEST" --record-ids-file "$OUT/folds/fold${fold}_train.txt" --out "$work/backbone" --seed "$fold" --device "$DEVICE" --epochs "$EPOCHS" --bandwidth-augment-prob "$PROB" --bandwidth-augment-rate 128 --target-fs 250 --lookback-ms 350 --delay-ms 200
  CUDA_VISIBLE_DEVICES="$gpu" python -m beatgru.train train-gru --manifest-dir "$MANIFEST" --record-ids-file "$OUT/folds/fold${fold}_train.txt" --backbone "$work/backbone/derivative_cnn.pt" --out "$work/gru" --seed "$fold" --device "$DEVICE" --epochs "$EPOCHS" --lr .0003 --power .40 --hidden 128 --max-correction .75 --chunk 8192 --target-fs 250 --lookback-ms 350 --delay-ms 200
  CUDA_VISIBLE_DEVICES="$gpu" python -m beatgru.train evaluate --manifest-dir "$MANIFEST" --dataset mit_bih_arrhythmia --target-split train --record-ids-file "$OUT/folds/fold${fold}_val.txt" --checkpoint "$work/gru/beatgru.pt" --backbone "$work/backbone/derivative_cnn.pt" --out "$work/validation" --seed "$fold" --device "$DEVICE" --hidden 128 --max-correction .75 --chunk 8192 --target-fs 250 --lookback-ms 350 --delay-ms 200
}
active=0
for fold in 0 1 2 3 4 5; do
  run_fold "$fold" "${GPU_LIST[$((fold % ${#GPU_LIST[@]}))]}" >"$OUT/fold${fold}.log" 2>&1 & active=$((active+1))
  if (( active >= ${#GPU_LIST[@]} )); then wait -n; active=$((active-1)); fi
done
while (( active > 0 )); do wait -n; active=$((active-1)); done

FINAL="$OUT/final"; mkdir -p "$FINAL"
CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" python -m beatgru.train train-cnn --manifest-dir "$MANIFEST" --out "$FINAL/backbone" --seed 0 --device "$DEVICE" --epochs "$EPOCHS" --bandwidth-augment-prob "$PROB" --bandwidth-augment-rate 128 --target-fs 250 --lookback-ms 350 --delay-ms 200 >"$FINAL/cnn.log" 2>&1
CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" python -m beatgru.train train-gru --manifest-dir "$MANIFEST" --backbone "$FINAL/backbone/derivative_cnn.pt" --out "$FINAL/gru" --seed 0 --device "$DEVICE" --epochs "$EPOCHS" --lr .0003 --power .40 --hidden 128 --max-correction .75 --chunk 8192 --target-fs 250 --lookback-ms 350 --delay-ms 200 >"$FINAL/gru.log" 2>&1
CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" python -m beatgru.train evaluate --manifest-dir "$MANIFEST" --dataset mit_bih_arrhythmia --target-split test --checkpoint "$FINAL/gru/beatgru.pt" --backbone "$FINAL/backbone/derivative_cnn.pt" --out "$FINAL/ds2" --seed 0 --device "$DEVICE" --hidden 128 --max-correction .75 --chunk 8192 --target-fs 250 --lookback-ms 350 --delay-ms 200 >"$FINAL/ds2.log" 2>&1
CUDA_VISIBLE_DEVICES="${GPU_LIST[1]}" python -m beatgru.train evaluate --manifest-dir "$EXTERNAL" --dataset st_petersburg_incart --target-split all --lead-selector source_matched_ii --checkpoint "$FINAL/gru/beatgru.pt" --backbone "$FINAL/backbone/derivative_cnn.pt" --out "$FINAL/incart_ii" --seed 0 --device "$DEVICE" --hidden 128 --max-correction .75 --chunk 8192 --target-fs 250 --lookback-ms 350 --delay-ms 200 >"$FINAL/incart.log" 2>&1 & p0=$!
CUDA_VISIBLE_DEVICES="${GPU_LIST[2]}" python -m beatgru.train evaluate --manifest-dir "$EXTERNAL" --dataset mit_bih_svdb --target-split all --lead-selector canonical_first --checkpoint "$FINAL/gru/beatgru.pt" --backbone "$FINAL/backbone/derivative_cnn.pt" --out "$FINAL/svdb_ecg1" --seed 0 --device "$DEVICE" --hidden 128 --max-correction .75 --chunk 8192 --target-fs 250 --lookback-ms 350 --delay-ms 200 >"$FINAL/svdb.log" 2>&1 & p1=$!
wait "$p0"; wait "$p1"
echo "Completed fixed 128-Hz bandwidth augmentation: $OUT"
