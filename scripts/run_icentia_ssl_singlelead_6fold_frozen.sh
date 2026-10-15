#!/usr/bin/env bash
# Label-free Icentia morphology initialization, strict single-lead DS1 training.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${1:?usage: GPUS=0,1,2 bash scripts/run_icentia_ssl_singlelead_6fold_frozen.sh DS1_MANIFEST EXTERNAL_MANIFEST [out]}"
EXTERNAL="${2:?external manifest required}"
PRETRAIN="${PRETRAIN:-/path/to/ssl_icentia400x128_e12_seed17.pt}"
OUT="${3:-$ROOT/runs/icentia_ssl_singlelead_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPUS="${GPUS:-0,1,2}"; IFS=',' read -r -a GPU_LIST <<< "$GPUS"; EPOCHS="${EPOCHS:-15}"
BASE="${BASE_RUN:-$ROOT/runs/ds1_6fold_then_ds2_20260918_170355}"
mkdir -p "$OUT"; cd "$ROOT"
"$PYTHON_BIN" tools/make_ds1_cv_folds.py --manifest-dir "$MANIFEST" --out "$OUT/folds" --folds 6
run_fold() {
  local fold="$1" gpu="$2" work="$OUT/fold$fold"; mkdir -p "$work"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m beatgru.train train-cnn --manifest-dir "$MANIFEST" --record-ids-file "$OUT/folds/fold${fold}_train.txt" --out "$work/backbone" --seed "$fold" --device cuda --epochs "$EPOCHS" --pretrained-encoder "$PRETRAIN" --lead-selector canonical_first >"$work/cnn.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m beatgru.train train-gru --manifest-dir "$MANIFEST" --record-ids-file "$OUT/folds/fold${fold}_train.txt" --backbone "$work/backbone/derivative_cnn.pt" --out "$work/gru" --seed "$fold" --device cuda --epochs "$EPOCHS" --lr .0003 --power .4 --hidden 128 --max-correction .75 --chunk 8192 --lead-selector canonical_first >"$work/gru.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m beatgru.train evaluate --manifest-dir "$MANIFEST" --dataset mit_bih_arrhythmia --target-split train --record-ids-file "$OUT/folds/fold${fold}_val.txt" --checkpoint "$work/gru/beatgru.pt" --backbone "$work/backbone/derivative_cnn.pt" --out "$work/validation" --seed "$fold" --device cuda --hidden 128 --max-correction .75 --chunk 8192 --lead-selector canonical_first >"$work/eval.log" 2>&1
}
active=0
for fold in 0 1 2 3 4 5; do run_fold "$fold" "${GPU_LIST[$((fold%${#GPU_LIST[@]}))]}" & active=$((active+1)); if ((active>=${#GPU_LIST[@]})); then wait -n; active=$((active-1)); fi; done
while ((active>0)); do wait -n; active=$((active-1)); done
"$PYTHON_BIN" - "$OUT" <<'PY'
import json,sys,numpy as np
from pathlib import Path
r=Path(sys.argv[1]); m=[json.loads((r/f'fold{i}/validation/test_metrics.json').read_text())['metrics'] for i in range(6)]
out={'protocol':'Icentia label-free causal SSL conv initialization; DS1 six-fold validation; DS2/external not loaded','mean':{k:float(np.mean([x[k] for x in m])) for k in ('macro_f1','sv_f1','N_f1','S_f1','V_f1','accuracy')},'std':{k:float(np.std([x[k] for x in m],ddof=1)) for k in ('macro_f1','sv_f1','S_f1','V_f1')}}
(r/'DS1_CV_SUMMARY.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
PY
FINAL="$OUT/final_ds1_retrain"; mkdir -p "$FINAL"
CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" "$PYTHON_BIN" -m beatgru.train train-cnn --manifest-dir "$MANIFEST" --out "$FINAL/backbone" --seed 0 --device cuda --epochs "$EPOCHS" --pretrained-encoder "$PRETRAIN" --lead-selector canonical_first >"$FINAL/cnn.log" 2>&1
CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" "$PYTHON_BIN" -m beatgru.train train-gru --manifest-dir "$MANIFEST" --backbone "$FINAL/backbone/derivative_cnn.pt" --out "$FINAL/gru" --seed 0 --device cuda --epochs "$EPOCHS" --lr .0003 --power .4 --hidden 128 --max-correction .75 --chunk 8192 --lead-selector canonical_first >"$FINAL/gru.log" 2>&1
CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" "$PYTHON_BIN" -m beatgru.train evaluate --manifest-dir "$MANIFEST" --dataset mit_bih_arrhythmia --target-split test --checkpoint "$FINAL/gru/beatgru.pt" --backbone "$FINAL/backbone/derivative_cnn.pt" --out "$FINAL/DS2_FINAL_ONCE" --seed 0 --device cuda --hidden 128 --max-correction .75 --chunk 8192 --lead-selector canonical_first >"$FINAL/ds2.log" 2>&1
CUDA_VISIBLE_DEVICES="${GPU_LIST[1]}" "$PYTHON_BIN" -m beatgru.train evaluate --manifest-dir "$EXTERNAL" --dataset st_petersburg_incart --target-split all --lead-selector source_matched_ii --checkpoint "$FINAL/gru/beatgru.pt" --backbone "$FINAL/backbone/derivative_cnn.pt" --out "$FINAL/INCART_II_ALL" --seed 0 --device cuda --hidden 128 --max-correction .75 --chunk 8192 >"$FINAL/incart.log" 2>&1 & p0=$!
CUDA_VISIBLE_DEVICES="${GPU_LIST[2]}" "$PYTHON_BIN" -m beatgru.train evaluate --manifest-dir "$EXTERNAL" --dataset mit_bih_svdb --target-split all --lead-selector canonical_first --checkpoint "$FINAL/gru/beatgru.pt" --backbone "$FINAL/backbone/derivative_cnn.pt" --out "$FINAL/SVDB_ECG1_ALL" --seed 0 --device cuda --hidden 128 --max-correction .75 --chunk 8192 >"$FINAL/svdb.log" 2>&1 & p1=$!
wait "$p0" "$p1"
echo "Completed Icentia SSL single-lead frozen evaluation: $FINAL"
