#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${1:?usage: bash scripts/audit.sh /path/to/manifest /path/to/beatgru.pt [out.json] [backbone.pt]}"
CHECKPOINT="${2:?usage: bash scripts/audit.sh /path/to/manifest /path/to/beatgru.pt [out.json] [backbone.pt]}"
OUT="${3:-$(dirname "$CHECKPOINT")/causality_audit.json}"
BACKBONE="${4:-}"
cd "$ROOT"
extra=()
if [[ -n "$BACKBONE" ]]; then extra+=(--backbone "$BACKBONE"); fi
CUDA_VISIBLE_DEVICES="${AUDIT_GPU:-0}" python -m beatgru.audit --manifest-dir "$MANIFEST" --checkpoint "$CHECKPOINT" --out "$OUT" --device "${DEVICE:-cuda}" "${extra[@]}"
