#!/usr/bin/env bash
# Non-causal control NC1 / NC2, run under the locked DS1 six-fold protocol.
#
#   bash scripts/run_noncausal_control.sh <manifest> <out_root> [stage]
#
# stage = folds | nc1 | nc2 | all   (default all)
#
# NC1 is executed at full strength: its own six record-disjoint DS1 folds, its
# own three-candidate selection, a complete DS1 retrain and exactly one DS2
# pass.  NC2 is executed at the minimum strength permitted by the protocol:
# the hyperparameters already selected for M3, one complete DS1 retrain and
# one DS2 pass, which is stated explicitly in the report.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${1:?usage: bash scripts/run_noncausal_control.sh /path/to/manifest /path/to/out [stage]}"
OUT="${2:?output root required}"
STAGE="${3:-all}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-15}"
FOLDS="${FOLDS:-6}"
CANDIDATES="${CANDIDATES:-$ROOT/experiments/ds1_6fold_causal/candidates.json}"
LOCKED_ROOT="$ROOT/runs/ds1_6fold_then_ds2_20260918_170355"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$OUT/logs"

make_folds() {
  # The locked six-fold partition is reused verbatim rather than regenerated.
  # tools/make_ds1_cv_folds.py drops DS1 records whose manifest rows carry
  # split=val, and the shipped v4 manifest marks five DS1 records that way, so
  # regenerating now would change fold membership.  Reusing the locked files is
  # also what "same treatment as the main result" requires.
  mkdir -p "$OUT/folds"
  cp "$LOCKED_ROOT"/folds/fold*_*.txt "$LOCKED_ROOT"/folds/folds.json "$OUT/folds/"
  "$PYTHON_BIN" - "$OUT/folds" "$LOCKED_ROOT/folds" <<'PY'
import hashlib, json, sys
from pathlib import Path
generated, locked = Path(sys.argv[1]), Path(sys.argv[2])
for fold in range(6):
    train = (generated / f"fold{fold}_train.txt").read_text().split()
    val = (generated / f"fold{fold}_val.txt").read_text().split()
    if set(train) & set(val):
        raise SystemExit(f"fold{fold} is not record-disjoint")
    if train != (locked / f"fold{fold}_train.txt").read_text().split():
        raise SystemExit(f"fold{fold}_train is not identical to the locked main run")
digest = hashlib.sha256((generated / "fold0_train.txt").read_bytes()).hexdigest()
json.dump({"source": str(locked.resolve()), "fold0_train_sha256": digest,
           "note": "locked six-fold partition reused verbatim; DS2 never loaded during selection"},
          open(generated / "PROVENANCE.json", "w"), ensure_ascii=False, indent=2)
print(f"folds identical to the locked main run; fold0_train sha256={digest}")
PY
}

candidate_rows() {
  "$PYTHON_BIN" -c 'import json,sys; [print("\t".join(map(str,(c["name"],c["hidden"],c["max_correction"],c["power"],c["lr"])))) for c in json.load(open(sys.argv[1]))["candidates"]]' "$CANDIDATES"
}

run_nc1() {
  local base="$OUT/nc1"
  mkdir -p "$base/backbones"
  for fold in 0 1 2 3 4 5; do
    "$PYTHON_BIN" tools/train_noncausal_control.py train-cnn --variant nc1 --manifest-dir "$MANIFEST" \
      --record-ids-file "$OUT/folds/fold${fold}_train.txt" --out "$base/backbones/fold${fold}" \
      --seed "$fold" --epochs "$EPOCHS" --device cuda >"$OUT/logs/nc1_cnn_fold${fold}.log" 2>&1
  done
  while IFS=$'\t' read -r name hidden correction power lr; do
    for fold in 0 1 2 3 4 5; do
      local work="$base/candidates/$name/fold$fold"
      mkdir -p "$work"
      "$PYTHON_BIN" tools/train_noncausal_control.py train-gru --variant nc1 --manifest-dir "$MANIFEST" \
        --record-ids-file "$OUT/folds/fold${fold}_train.txt" --backbone "$base/backbones/fold${fold}/derivative_cnn.pt" \
        --out "$work/gru" --seed "$fold" --epochs "$EPOCHS" --lr "$lr" --power "$power" \
        --hidden "$hidden" --max-correction "$correction" --chunk 8192 --device cuda \
        >"$OUT/logs/nc1_${name}_fold${fold}_gru.log" 2>&1
      "$PYTHON_BIN" tools/train_noncausal_control.py evaluate --variant nc1 --manifest-dir "$MANIFEST" \
        --dataset mit_bih_arrhythmia --target-split train --record-ids-file "$OUT/folds/fold${fold}_val.txt" \
        --checkpoint "$work/gru/history.pt" --backbone "$base/backbones/fold${fold}/derivative_cnn.pt" \
        --out "$work/validation" --seed "$fold" \
        --device cuda >"$OUT/logs/nc1_${name}_fold${fold}_eval.log" 2>&1
    done
  done < <(candidate_rows)
  "$PYTHON_BIN" tools/select_ds1_cv_candidate.py --candidates "$CANDIDATES" --runs "$base/candidates" \
    --out "$base/LOCKED_CONFIG_NC1.json" --folds "$FOLDS" >"$OUT/logs/nc1_select.log" 2>&1
  read -r NAME HIDDEN CORRECTION POWER LR < <("$PYTHON_BIN" -c 'import json,sys; c=json.load(open(sys.argv[1]))["selected"]["candidate"]; print(c["name"],c["hidden"],c["max_correction"],c["power"],c["lr"])' "$base/LOCKED_CONFIG_NC1.json")
  local final="$base/final_ds1_retrain/$NAME"
  mkdir -p "$final"
  "$PYTHON_BIN" tools/train_noncausal_control.py train-cnn --variant nc1 --manifest-dir "$MANIFEST" \
    --out "$final/backbone" --seed 0 --epochs "$EPOCHS" --device cuda >"$OUT/logs/nc1_final_cnn.log" 2>&1
  "$PYTHON_BIN" tools/train_noncausal_control.py train-gru --variant nc1 --manifest-dir "$MANIFEST" \
    --backbone "$final/backbone/derivative_cnn.pt" --out "$final/gru" --seed 0 --epochs "$EPOCHS" \
    --lr "$LR" --power "$POWER" --hidden "$HIDDEN" --max-correction "$CORRECTION" --chunk 8192 --device cuda \
    >"$OUT/logs/nc1_final_gru.log" 2>&1
  "$PYTHON_BIN" tools/train_noncausal_control.py evaluate --variant nc1 --manifest-dir "$MANIFEST" \
    --dataset mit_bih_arrhythmia --target-split test --checkpoint "$final/gru/history.pt" \
    --backbone "$final/backbone/derivative_cnn.pt" --out "$final/DS2_FINAL_ONCE" --seed 0 \
    --device cuda >"$OUT/logs/nc1_final_ds2.log" 2>&1
  "$PYTHON_BIN" tools/train_noncausal_control.py audit --variant nc1 --manifest-dir "$MANIFEST" \
    --checkpoint "$final/gru/history.pt" --backbone "$final/backbone/derivative_cnn.pt" --split test \
    --out "$final/causality_audit_nc1.json" --device cuda >"$OUT/logs/nc1_audit.log" 2>&1
  echo "NC1 finished: $final/DS2_FINAL_ONCE"
}

run_nc2() {
  local base="$OUT/nc2"
  local final="$base/final_ds1_retrain/m3_hyperparameters_minimum_strength"
  mkdir -p "$final"
  read -r HIDDEN CORRECTION POWER LR < <("$PYTHON_BIN" -c 'import json,sys; c=json.load(open(sys.argv[1]))["selected"]["candidate"]; print(c["hidden"],c["max_correction"],c["power"],c["lr"])' "$LOCKED_ROOT/LOCKED_CONFIG.json")
  "$PYTHON_BIN" -c 'import json,sys; c=json.load(open(sys.argv[1]))["selected"]["candidate"]; json.dump({"protocol":"NC2 minimum strength: M3 hyperparameters reused, no independent selection","selected":c}, open(sys.argv[2],"w"), ensure_ascii=False, indent=2)' "$LOCKED_ROOT/LOCKED_CONFIG.json" "$final/NC2_SELECTION_DISCLOSURE.json"
  "$PYTHON_BIN" tools/train_noncausal_control.py train-cnn --variant nc2 --manifest-dir "$MANIFEST" \
    --out "$final/backbone" --seed 0 --epochs "$EPOCHS" --device cuda >"$OUT/logs/nc2_final_cnn.log" 2>&1
  "$PYTHON_BIN" tools/train_noncausal_control.py train-gru --variant nc2 --manifest-dir "$MANIFEST" \
    --backbone "$final/backbone/derivative_cnn.pt" --out "$final/gru" --seed 0 --epochs "$EPOCHS" \
    --lr "$LR" --power "$POWER" --hidden "$HIDDEN" --max-correction "$CORRECTION" --chunk 65536 --device cuda \
    >"$OUT/logs/nc2_final_gru.log" 2>&1
  "$PYTHON_BIN" tools/train_noncausal_control.py evaluate --variant nc2 --manifest-dir "$MANIFEST" \
    --dataset mit_bih_arrhythmia --target-split test --checkpoint "$final/gru/history.pt" \
    --backbone "$final/backbone/derivative_cnn.pt" --out "$final/DS2_FINAL_ONCE" --seed 0 \
    --device cuda >"$OUT/logs/nc2_final_ds2.log" 2>&1
  "$PYTHON_BIN" tools/train_noncausal_control.py audit --variant nc2 --manifest-dir "$MANIFEST" \
    --checkpoint "$final/gru/history.pt" --backbone "$final/backbone/derivative_cnn.pt" --split test \
    --out "$final/causality_audit_nc2.json" --device cuda >"$OUT/logs/nc2_audit.log" 2>&1
  echo "NC2 finished: $final/DS2_FINAL_ONCE"
}

case "$STAGE" in
  folds) make_folds ;;
  nc1) make_folds; run_nc1 ;;
  nc2) run_nc2 ;;
  all) make_folds; run_nc1; run_nc2 ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
