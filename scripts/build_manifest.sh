#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW_ROOT="${1:?usage: bash scripts/build_manifest.sh /path/to/mit-bih-arrhythmia-database-1.0.0 [manifest_dir]}"
MANIFEST="${2:-$ROOT/data/mitbih_ds1ds2}"
cd "$ROOT"
python -c "from beatgru.data import build_manifest; build_manifest('$RAW_ROOT', '$MANIFEST')"
echo "Manifest written to $MANIFEST"
