"""Create a concise Markdown table from independently reproduced runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FIELDS = ("accuracy", "balanced_accuracy", "N_f1", "S_f1", "V_f1", "macro_f1", "sv_f1")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--runs-root", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.runs_root.glob("gru_seed*/test/test_metrics.json"))
    if not files: raise FileNotFoundError(f"no test metrics under {args.runs_root}")
    rows = [(path.parents[1].name, json.loads(path.read_text(encoding="utf-8"))["metrics"]) for path in files]
    lines = ["# BEATGRU reproduction results", "", "| seed | " + " | ".join(FIELDS) + " |", "|---|" + "---:|" * len(FIELDS)]
    for name, values in rows:
        lines.append("| " + name.removeprefix("gru_seed") + " | " + " | ".join(f"{values[field]:.6f}" for field in FIELDS) + " |")
    lines += ["", "## Mean ± sample standard deviation", "", "| metric | value |", "|---|---:|"]
    for field in FIELDS:
        values = np.asarray([row[field] for _, row in rows], dtype=float); std = values.std(ddof=1) if len(values) > 1 else 0.0
        lines.append(f"| {field} | {values.mean():.6f} ± {std:.6f} |")
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
