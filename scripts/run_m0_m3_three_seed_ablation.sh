#!/usr/bin/env bash
# Paired M0-M3 strict-causal ablation; one seed per GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${1:?usage: GPUS=0,1,2 bash scripts/run_m0_m3_three_seed_ablation.sh DS1_MANIFEST EXTERNAL_MANIFEST [out]}"
EXTERNAL="${2:?external manifest required}"; OUT="${3:-$ROOT/runs/m0_m3_three_seed_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"; GPUS="${GPUS:-0,1,2}"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"; EPOCHS="${EPOCHS:-15}"; mkdir -p "$OUT"; cd "$ROOT"
run_seed() {
  local seed="$1" gpu="$2" s="$OUT/seed${seed}"; mkdir -p "$s"
  # M0: morphology-only; M1 backbone is shared by M1/M2/M3.
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/train_m0_morphology.py --mode train --manifest-dir "$MANIFEST" --out "$s/M0/backbone" --seed "$seed" --device cuda --epochs "$EPOCHS" >"$s/M0_train.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m beatgru.train train-cnn --manifest-dir "$MANIFEST" --out "$s/M1/backbone" --seed "$seed" --device cuda --epochs "$EPOCHS" --lead-selector canonical_first >"$s/M1_cnn_train.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/run_history_ablation.py train --mode short32 --manifest-dir "$MANIFEST" --backbone "$s/M1/backbone/derivative_cnn.pt" --out "$s/M2/history" --seed "$seed" --device cuda --epochs "$EPOCHS" --hidden 128 --max-correction .75 >"$s/M2_train.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/run_history_ablation.py train --mode full --manifest-dir "$MANIFEST" --backbone "$s/M1/backbone/derivative_cnn.pt" --out "$s/M3/history" --seed "$seed" --device cuda --epochs "$EPOCHS" --hidden 128 --max-correction .75 >"$s/M3_train.log" 2>&1
  # DS2 exactly once for each frozen seed/method.
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/train_m0_morphology.py --mode evaluate --manifest-dir "$MANIFEST" --dataset mit_bih_arrhythmia --target-split test --checkpoint "$s/M0/backbone/m0_morphology.pt" --out "$s/M0/DS2_FINAL_ONCE" --device cuda >"$s/M0_ds2.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/run_history_ablation.py evaluate --mode m1 --manifest-dir "$MANIFEST" --dataset mit_bih_arrhythmia --target-split test --backbone "$s/M1/backbone/derivative_cnn.pt" --out "$s/M1/DS2_FINAL_ONCE" --device cuda >"$s/M1_ds2.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/run_history_ablation.py evaluate --mode short32 --manifest-dir "$MANIFEST" --dataset mit_bih_arrhythmia --target-split test --checkpoint "$s/M2/history/history.pt" --out "$s/M2/DS2_FINAL_ONCE" --device cuda >"$s/M2_ds2.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/run_history_ablation.py evaluate --mode full --manifest-dir "$MANIFEST" --dataset mit_bih_arrhythmia --target-split test --checkpoint "$s/M3/history/history.pt" --out "$s/M3/DS2_FINAL_ONCE" --device cuda >"$s/M3_ds2.log" 2>&1
  # Only M1 and M3 are required for external paired comparison.
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/run_history_ablation.py evaluate --mode m1 --manifest-dir "$EXTERNAL" --dataset st_petersburg_incart --target-split all --lead-selector source_matched_ii --backbone "$s/M1/backbone/derivative_cnn.pt" --out "$s/M1/INCART_II_ALL" --device cuda >"$s/M1_incart.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/run_history_ablation.py evaluate --mode full --manifest-dir "$EXTERNAL" --dataset st_petersburg_incart --target-split all --lead-selector source_matched_ii --checkpoint "$s/M3/history/history.pt" --out "$s/M3/INCART_II_ALL" --device cuda >"$s/M3_incart.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/run_history_ablation.py evaluate --mode m1 --manifest-dir "$EXTERNAL" --dataset mit_bih_svdb --target-split all --lead-selector canonical_first --backbone "$s/M1/backbone/derivative_cnn.pt" --out "$s/M1/SVDB_ECG1_ALL" --device cuda >"$s/M1_svdb.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/run_history_ablation.py evaluate --mode full --manifest-dir "$EXTERNAL" --dataset mit_bih_svdb --target-split all --lead-selector canonical_first --checkpoint "$s/M3/history/history.pt" --out "$s/M3/SVDB_ECG1_ALL" --device cuda >"$s/M3_svdb.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m beatgru.audit --manifest-dir "$MANIFEST" --checkpoint "$s/M3/history/history.pt" --backbone "$s/M1/backbone/derivative_cnn.pt" --out "$s/M3/causality_audit.json" --device cuda >"$s/M3_audit.log" 2>&1
}
active=0; i=0
for seed in 0 1 2; do run_seed "$seed" "${GPU_LIST[$i]}" >"$OUT/seed${seed}.log" 2>&1 & active=$((active+1)); i=$((i+1)); done
wait
"$PYTHON_BIN" tools/summarize_ablation.py --root "$OUT" --out "$OUT/summary" --baseline-root "$ROOT/runs/ds1_6fold_then_ds2_20260918_170355" --external
echo "Completed M0-M3 three-seed strict-causal ablation: $OUT"
