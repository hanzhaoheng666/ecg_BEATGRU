#!/usr/bin/env python3
"""Record-cluster bootstrap intervals and paired contrasts for locked runs.

The resampling unit is a complete ECG record, never a beat: beats inside one
record are strongly dependent, so a beat-level bootstrap would treat ~50k
beats as independent draws and produce intervals that are far too narrow.  A
record-level interval answers "what happens if a different set of patients is
drawn", which is the question the inter-patient protocol actually asks.

Every replicate draws the same set of records for both members of a pair, so
the paired interval removes the shared record-composition variation and is the
sensitive test for "did this change help".

The point estimate is always the locked run's own value, never the bootstrap
mean.
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
sys.path.insert(0, str(ROOT / "tools"))

from bootstrap_record_metrics import records, record_confusions, values_from_confusion  # noqa: E402

TABLE_A_METRICS = ("macro_f1", "S_f1", "V_f1", "S_recall", "S_precision")


def recall_precision(confusions: np.ndarray) -> dict[str, np.ndarray]:
    """Recall/precision layers on top of the shared confusion helper."""
    diag = np.diagonal(confusions, axis1=-2, axis2=-1).astype(float)
    predicted = confusions.sum(axis=-2)
    actual = confusions.sum(axis=-1)
    return {
        "S_recall": np.divide(diag[..., 1], actual[..., 1], out=np.zeros(actual.shape[:-1]), where=actual[..., 1] > 0),
        "S_precision": np.divide(diag[..., 1], predicted[..., 1], out=np.zeros(predicted.shape[:-1]), where=predicted[..., 1] > 0),
    }


def all_metrics(confusions: np.ndarray) -> dict[str, np.ndarray]:
    return {**values_from_confusion(confusions), **recall_precision(confusions)}


def interval(samples: np.ndarray) -> list[float]:
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def load_pair(template: str, seed: int) -> tuple[str, list[str], np.ndarray]:
    path = template.format(seed=seed)
    ids, rows = records(path)
    if not ids:
        raise ValueError(f"no records in {path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"a record appears twice in {path}")
    return path, ids, record_confusions(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True,
                        help="name=path template, {seed} is replaced per seed")
    parser.add_argument("--pair", action="append", default=[],
                        help="left-right contrast, e.g. M1-M0")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    if args.replicates < 100:
        raise ValueError("--replicates must be >=100")

    models = {}
    for entry in args.model:
        name, _, template = entry.partition("=")
        if not template:
            raise ValueError(f"--model expects name=path, got {entry!r}")
        models[name.strip()] = template.strip()
    seeds = [int(value) for value in args.seeds.split(",")]

    report: dict[str, object] = {
        "domain": args.domain, "bootstrap_unit": "complete ECG record",
        "replicates": args.replicates, "seed": args.seed, "seeds": seeds,
        "point_estimate": "locked run value", "resampling": "records drawn with replacement",
    }

    loaded: dict[tuple[str, int], tuple[str, list[str], np.ndarray]] = {}
    table_a: dict[str, dict[str, object]] = {}
    for name, template in models.items():
        per_seed: dict[str, object] = {}
        for seed in seeds:
            path, ids, confusions = load_pair(template, seed)
            loaded[(name, seed)] = (path, ids, confusions)
            rng = np.random.default_rng(args.seed)
            draws = rng.integers(0, len(ids), size=(args.replicates, len(ids)))
            sampled = all_metrics(confusions[draws].sum(axis=1))
            point = all_metrics(confusions.sum(axis=0))
            per_seed[str(seed)] = {
                "path": path, "n_records": len(ids), "n_beats": int(confusions.sum()),
                "point": {key: float(np.asarray(point[key])) for key in TABLE_A_METRICS + ("N_f1", "sv_f1", "balanced_accuracy", "accuracy")},
                "bootstrap95": {key: interval(sampled[key]) for key in TABLE_A_METRICS + ("N_f1", "sv_f1", "balanced_accuracy", "accuracy")},
            }
        aggregated = {}
        for key in TABLE_A_METRICS:
            values = [per_seed[str(seed)]["point"][key] for seed in seeds]
            aggregated[key] = {"mean_over_seeds": float(np.mean(values)),
                               "sd_over_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
        table_a[name] = {"per_seed": per_seed, "across_seeds": aggregated}
    report["table_a"] = table_a

    table_b: dict[str, object] = {}
    for contrast in args.pair:
        left, _, right = contrast.partition("-")
        if not left or not right or left not in models or right not in models:
            raise ValueError(f"--pair expects known-model names, got {contrast!r}")
        entry: dict[str, object] = {"delta": f"{right}-{left}"}
        for seed in seeds:
            _, ids_a, cm_a = loaded[(left, seed)]
            _, ids_b, cm_b = loaded[(right, seed)]
            if ids_a != ids_b:
                raise ValueError(f"paired files disagree on record order: {left} vs {right}, seed {seed}")
            rng = np.random.default_rng(args.seed + 17 * seed)
            draws = rng.integers(0, len(ids_a), size=(args.replicates, len(ids_a)))
            sampled_a = all_metrics(cm_a[draws].sum(axis=1))
            sampled_b = all_metrics(cm_b[draws].sum(axis=1))
            point_a = all_metrics(cm_a.sum(axis=0))
            point_b = all_metrics(cm_b.sum(axis=0))
            per_metric = {}
            for key in ("macro_f1", "S_f1", "V_f1"):
                difference = sampled_b[key] - sampled_a[key]
                low, high = interval(difference)
                per_metric[key] = {
                    "point_delta": float(np.asarray(point_b[key]) - np.asarray(point_a[key])),
                    "bootstrap95": [low, high],
                    "includes_zero": bool(low <= 0.0 <= high),
                }
            entry[str(seed)] = per_metric
        table_b[f"{left}_to_{right}"] = entry
    report["table_b"] = table_b

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "record_cluster_bootstrap.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# 按记录聚类的自举区间：{args.domain}", "",
             f"- 重采样单元：整条记录；每次抽取的记录数与该域记录数相同；B = {args.replicates}；固定种子 {args.seed}",
             "- 点估计取锁定运行的原始值，不取自举均值", "",
             "## 表 A　各模型点估计与 95% 区间（pooled 混淆矩阵）", "",
             "| 模型 | 记录数 | 心拍数 | Macro-F1 [95% CI] | S-F1 [95% CI] | V-F1 [95% CI] | S 召回 [95% CI] | S 精确率 [95% CI] |",
             "|---|---:|---:|---|---|---|---|---|"]
    for name, payload in table_a.items():
        for seed in seeds:
            row = payload["per_seed"][str(seed)]
            cells = []
            for key in TABLE_A_METRICS:
                low, high = row["bootstrap95"][key]
                cells.append(f"{row['point'][key]:.4f} [{low:.4f}, {high:.4f}]")
            lines.append(f"| {name} (seed {seed}) | {row['n_records']} | {row['n_beats']} | " + " | ".join(cells) + " |")
    lines += ["", "### 跨种子波动（点估计的均值 ± 标准差）", "", "| 模型 | Macro-F1 | S-F1 | V-F1 |", "|---|---|---|---|"]
    for name, payload in table_a.items():
        cells = [f"{payload['across_seeds'][key]['mean_over_seeds']:.4f} ± {payload['across_seeds'][key]['sd_over_seeds']:.4f}"
                 for key in ("macro_f1", "S_f1", "V_f1")]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    if table_b:
        lines += ["", "## 表 B　成对差值（同一重采样内计算，消除记录组成的共同波动）", "",
                  "| 对比 | seed | Macro-F1 差值 [95% CI] | 含零 | S-F1 差值 [95% CI] | 含零 | V-F1 差值 [95% CI] | 含零 |",
                  "|---|---:|---|---:|---|---:|---|---:|"]
        for contrast, entry in table_b.items():
            for seed in seeds:
                cells, flags = [], []
                for key in ("macro_f1", "S_f1", "V_f1"):
                    value = entry[str(seed)][key]
                    low, high = value["bootstrap95"]
                    cells.append(f"{value['point_delta']:+.4f} [{low:+.4f}, {high:+.4f}]")
                    flags.append("是" if value["includes_zero"] else "否")
                lines.append(f"| {contrast} | {seed} | " + " | ".join(f"{c} | {f}" for c, f in zip(cells, flags)) + " |")
    (out / "record_cluster_bootstrap.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
