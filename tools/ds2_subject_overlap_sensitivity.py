#!/usr/bin/env python3
"""Sensitivity of DS2 metrics to the MIT-BIH subject shared by 201 and 202.

Records 201 (DS1, training) and 202 (DS2, test) belong to the same subject, so
the de Chazal inter-patient split is patient-disjoint for every subject except
this pair.  This tool recomputes the pooled N/S/V metrics with record 202
included and excluded, and also reports record 202's own metrics so the size of
the contribution is visible.  No training, no re-selection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from beatgru.metrics import metric

KEYS = ("macro_f1", "sv_f1", "N_f1", "S_f1", "V_f1", "balanced_accuracy", "accuracy")


def load(path: str, excluded: str) -> dict[str, object]:
    data = np.load(path, allow_pickle=False)
    ids = [str(x) for x in data["record_ids"]]
    offsets = data["record_offsets"].astype(int)
    if len(ids) != len(offsets) - 1:
        raise ValueError(f"{path}: record_ids/offsets mismatch")
    labels, predictions = data["labels"], data["predictions"]
    keep_all = np.ones(len(labels), dtype=bool)
    keep_excl = keep_all.copy()
    target = np.zeros(0, dtype=bool)
    if excluded in ids:
        index = ids.index(excluded)
        start, stop = offsets[index], offsets[index + 1]
        keep_excl[start:stop] = False
        target = np.zeros(len(labels), dtype=bool)
        target[start:stop] = True
    return {
        "path": str(Path(path).resolve()),
        "n_records": len(ids),
        "excluded_record": excluded if target.any() else None,
        "included": {key: float(metric(labels[keep_all], predictions[keep_all])[key]) for key in KEYS},
        "excluded": {key: float(metric(labels[keep_excl], predictions[keep_excl])[key]) for key in KEYS},
        "n_beats_included": int(keep_all.sum()),
        "n_beats_excluded": int(keep_excl.sum()),
        "record_itself": ({key: float(metric(labels[target], predictions[target])[key]) for key in KEYS}
                          if target.any() else None),
        "n_beats_in_record": int(target.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="name=npz")
    parser.add_argument("--exclude", default="202", help="record id removed from the pooled metrics")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = {"excluded_record": args.exclude, "metric_scope": "pooled over scored N/S/V beats", "models": {}}
    for entry in args.model:
        name, _, path = entry.partition("=")
        report["models"][name.strip()] = load(path.strip(), args.exclude)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "subject_overlap_sensitivity.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# 同受试者记录 `{args.exclude}` 的敏感性（零训练，仅重算指标）", "",
             "MIT-BIH 的 201（DS1 训练）与 202（DS2 测试）来自同一受试者，是 de Chazal 划分中",
             "唯一一处并非严格患者不重叠的位置。下表给出剔除该记录前后的 pooled 指标。", "",
             "| 模型 | 记录数 | 心拍数 | Macro-F1 含 | Macro-F1 剔除 | Δ | S-F1 含 | S-F1 剔除 | Δ | V-F1 含 | V-F1 剔除 | Δ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, payload in report["models"].items():
        a, b = payload["included"], payload["excluded"]
        lines.append(f"| {name} | {payload['n_records']} | {payload['n_beats_included']} | "
                     f"{a['macro_f1']:.6f} | {b['macro_f1']:.6f} | {b['macro_f1'] - a['macro_f1']:+.6f} | "
                     f"{a['S_f1']:.6f} | {b['S_f1']:.6f} | {b['S_f1'] - a['S_f1']:+.6f} | "
                     f"{a['V_f1']:.6f} | {b['V_f1']:.6f} | {b['V_f1'] - a['V_f1']:+.6f} |")
    if any(payload["record_itself"] for payload in report["models"].values()):
        lines += ["", f"## 记录 `{args.exclude}` 自身的指标", "",
                  "| 模型 | 心拍数 | Macro-F1 | S-F1 | V-F1 |", "|---|---:|---:|---:|---:|"]
        for name, payload in report["models"].items():
            own = payload["record_itself"]
            if own is None:
                continue
            lines.append(f"| {name} | {payload['n_beats_in_record']} | {own['macro_f1']:.6f} | "
                         f"{own['S_f1']:.6f} | {own['V_f1']:.6f} |")
    (out / "subject_overlap_sensitivity.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
