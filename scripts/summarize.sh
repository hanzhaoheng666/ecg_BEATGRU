#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ROOT="${1:?usage: bash scripts/summarize.sh /path/to/run_root [summary.md]}"
OUT="${2:-$RUN_ROOT/summary.md}"
cd "$ROOT"
python -m beatgru.summary --runs-root "$RUN_ROOT" --out "$OUT"
echo "Wrote $OUT"
