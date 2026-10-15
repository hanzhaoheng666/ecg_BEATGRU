#!/usr/bin/env python3
"""Record-cluster bootstrap and paired comparison for saved N/S/V predictions.

The unit resampled is a complete ECG record, never individual beats.  This is
appropriate for the fixed DS2 record set, whose beat counts are highly unequal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from beatgru.metrics import metric


# ``metric`` preserves the canonical AAMI letters in its per-class keys.
KEYS = ("macro_f1", "sv_f1", "N_f1", "S_f1", "V_f1", "balanced_accuracy", "accuracy")


def records(path: str) -> tuple[list[str], list[tuple[np.ndarray, np.ndarray]]]:
    data = np.load(path, allow_pickle=False)
    required = {"labels", "predictions", "record_ids", "record_offsets"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    ids = [str(x) for x in data["record_ids"]]
    offsets = data["record_offsets"].astype(int)
    if len(offsets) != len(ids) + 1 or offsets[0] != 0 or offsets[-1] != len(data["labels"]):
        raise ValueError(f"{path}: invalid record offsets")
    return ids, [(data["labels"][offsets[i]:offsets[i + 1]], data["predictions"][offsets[i]:offsets[i + 1]]) for i in range(len(ids))]


def record_confusions(rows: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Return one 3x3 confusion matrix per record without beat-level resampling."""
    matrices = []
    for truth, pred in rows:
        matrices.append(np.asarray(metric(truth, pred)["confusion_matrix"], dtype=np.int64))
    return np.stack(matrices)


def values_from_confusion(cm: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized metrics for (..., 3, 3) confusion matrices."""
    diag = np.diagonal(cm, axis1=-2, axis2=-1).astype(float)
    predicted = cm.sum(axis=-2)
    actual = cm.sum(axis=-1)
    precision = np.divide(diag, predicted, out=np.zeros_like(diag), where=predicted > 0)
    recall = np.divide(diag, actual, out=np.zeros_like(diag), where=actual > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(diag), where=(precision + recall) > 0)
    n = cm.sum(axis=(-2, -1))
    return {"macro_f1": f1.mean(axis=-1), "sv_f1": f1[..., 1:].mean(axis=-1), "N_f1": f1[..., 0],
            "S_f1": f1[..., 1], "V_f1": f1[..., 2], "balanced_accuracy": recall.mean(axis=-1),
            "accuracy": np.divide(diag.sum(axis=-1), n, out=np.zeros_like(n, dtype=float), where=n > 0)}


def point(values: dict[str, np.ndarray]) -> dict[str, float]:
    return {key: float(np.asarray(value)) for key, value in values.items()}


def ci(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(values.mean()), "ci95_low": float(np.quantile(values, .025)), "ci95_high": float(np.quantile(values, .975))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-a", required=True, help="reference/model-A NPZ")
    parser.add_argument("--predictions-b", default="", help="optional paired model-B NPZ")
    parser.add_argument("--out", required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    if args.replicates < 100:
        raise ValueError("--replicates must be >=100")
    ids_a, rows_a = records(args.predictions_a)
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(rows_a), size=(args.replicates, len(rows_a)))
    cm_a = record_confusions(rows_a)
    samples_a = values_from_confusion(cm_a[indices].sum(axis=1))
    result: dict[str, object] = {
        "unit": "complete ECG record (cluster bootstrap)", "n_records": len(rows_a), "record_ids": ids_a,
        "replicates": args.replicates, "seed": args.seed, "model_a": str(Path(args.predictions_a).resolve()),
        "point_estimate_a": point(values_from_confusion(cm_a.sum(axis=0))),
        "bootstrap_a": {key: ci(values) for key, values in samples_a.items()},
    }
    if args.predictions_b:
        ids_b, rows_b = records(args.predictions_b)
        if ids_a != ids_b or any(not np.array_equal(a[0], b[0]) for a, b in zip(rows_a, rows_b)):
            raise ValueError("paired prediction files must have identical ordered record IDs and labels")
        cm_b = record_confusions(rows_b)
        samples_b = values_from_confusion(cm_b[indices].sum(axis=1))
        result.update({"model_b": str(Path(args.predictions_b).resolve()), "point_estimate_b": point(values_from_confusion(cm_b.sum(axis=0))),
                       "paired_difference_b_minus_a": {key: ci(samples_b[key] - samples_a[key]) for key in KEYS}})
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
