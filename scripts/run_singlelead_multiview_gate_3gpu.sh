#!/usr/bin/env bash
# DS1-only bounded fusion-gate selection, then frozen DS2/external evaluation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${1:?usage: GPUS=0,1,2 bash scripts/run_singlelead_multiview_gate_3gpu.sh DS1_MANIFEST EXTERNAL_MANIFEST [out]}"
EXTERNAL="${2:?external manifest required}"; OUT="${3:-$ROOT/runs/singlelead_multiview_gate_$(date +%Y%m%d_%H%M%S)}"
BASE="${BASE_RUN:-$ROOT/runs/ds1_6fold_then_ds2_20260918_170355}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPUS="${GPUS:-0,1,2}"; IFS=',' read -r -a GPU_LIST <<< "$GPUS"; EPOCHS="${EPOCHS:-20}"
mkdir -p "$OUT"; cd "$ROOT"
run_fold() {
  local name="$1" corr="$2" power="$3" fold="$4" gpu="$5"; local work="$OUT/$name/fold$fold"; mkdir -p "$work"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/train_singlelead_multiview_gate.py train --manifest-dir "$MANIFEST" --record-ids-file "$BASE/folds/fold${fold}_train.txt" --checkpoint "$BASE/candidates/gru_h128_c075_p040/fold${fold}/gru/beatgru.pt" --backbone "$BASE/backbones/fold${fold}/derivative_cnn.pt" --out "$work/gate" --seed "$fold" --device cuda --epochs "$EPOCHS" --max-correction "$corr" --power "$power"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/train_singlelead_multiview_gate.py evaluate --manifest-dir "$MANIFEST" --target-split train --record-ids-file "$BASE/folds/fold${fold}_val.txt" --gate "$work/gate/view_gate.pt" --out "$work/validation" --device cuda --max-correction "$corr"
}
active=0; job=0
for spec in 'gate_c025_p040 .25 .4' 'gate_c050_p040 .50 .4'; do
  read -r name corr power <<< "$spec"
  for fold in 0 1 2 3 4 5; do
    run_fold "$name" "$corr" "$power" "$fold" "${GPU_LIST[$((job % ${#GPU_LIST[@]}))]}" >"$OUT/${name}_fold${fold}.log" 2>&1 &
    active=$((active+1)); job=$((job+1)); if ((active>=${#GPU_LIST[@]})); then wait -n; active=$((active-1)); fi
  done
done
while ((active>0)); do wait -n; active=$((active-1)); done
"$PYTHON_BIN" - "$OUT" <<'PY'
import json,sys,numpy as np
from pathlib import Path
out=Path(sys.argv[1]); candidates=[]
for name in ('gate_c025_p040','gate_c050_p040'):
 m=[json.loads((out/name/f'fold{i}'/'validation'/'test_metrics.json').read_text())['metrics'] for i in range(6)]
 candidates.append({'name':name,'mean':{k:float(np.mean([x[k] for x in m])) for k in ('macro_f1','sv_f1','S_f1','V_f1')}})
# DS1 predeclared safety constraint: retain only gates whose mean S-F1 is at
# least the already-validated two-view arithmetic fusion S-F1 (0.215563).
eligible=[x for x in candidates if x['mean']['S_f1'] >= .215563]
selected=max(eligible or candidates,key=lambda x:(x['mean']['macro_f1'],x['mean']['sv_f1']))
(out/'LOCKED_GATE.json').write_text(json.dumps({'selection':'DS1 six-fold Macro-F1, constrained S-F1 >= 0.215563','candidates':candidates,'selected':selected},indent=2))
print(selected['name'])
PY
NAME="$("$PYTHON_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1]))["selected"]["name"])' "$OUT/LOCKED_GATE.json")"
CORR="$(case "$NAME" in gate_c025_p040) echo .25;; gate_c050_p040) echo .50;; esac)"
FINAL="$OUT/final_frozen/$NAME"; mkdir -p "$FINAL"
CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" "$PYTHON_BIN" tools/train_singlelead_multiview_gate.py train --manifest-dir "$MANIFEST" --checkpoint "$BASE/final_ds1_retrain/gru_h128_c075_p040/gru/beatgru.pt" --backbone "$BASE/final_ds1_retrain/gru_h128_c075_p040/backbone/derivative_cnn.pt" --out "$FINAL/gate" --seed 0 --device cuda --epochs "$EPOCHS" --max-correction "$CORR" --power .4 >"$FINAL/train.log" 2>&1
CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" "$PYTHON_BIN" tools/train_singlelead_multiview_gate.py evaluate --manifest-dir "$MANIFEST" --target-split test --gate "$FINAL/gate/view_gate.pt" --out "$FINAL/DS2_FINAL_ONCE" --device cuda --max-correction "$CORR" >"$FINAL/ds2.log" 2>&1
CUDA_VISIBLE_DEVICES="${GPU_LIST[1]}" "$PYTHON_BIN" tools/train_singlelead_multiview_gate.py evaluate --manifest-dir "$EXTERNAL" --dataset st_petersburg_incart --target-split all --gate "$FINAL/gate/view_gate.pt" --out "$FINAL/INCART_II_V1_ALL" --device cuda --max-correction "$CORR" >"$FINAL/incart.log" 2>&1 & p0=$!
CUDA_VISIBLE_DEVICES="${GPU_LIST[2]}" "$PYTHON_BIN" tools/train_singlelead_multiview_gate.py evaluate --manifest-dir "$EXTERNAL" --dataset mit_bih_svdb --target-split all --second-selector file_second --gate "$FINAL/gate/view_gate.pt" --out "$FINAL/SVDB_ECG1_ECG2_ALL" --device cuda --max-correction "$CORR" >"$FINAL/svdb.log" 2>&1 & p1=$!
wait "$p0" "$p1"
echo "Completed DS1-selected fusion gate: $FINAL"
