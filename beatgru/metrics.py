"""N/S/V metrics with no sklearn dependency."""
from __future__ import annotations

import numpy as np


CLASS_NAMES = ("N", "S", "V")


def metric(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape or not len(y_true):
        raise ValueError("labels and predictions must be equal-length nonempty vectors")
    cm = np.zeros((3, 3), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred):
        if 0 <= actual < 3 and 0 <= predicted < 3:
            cm[actual, predicted] += 1
    precision = np.divide(np.diag(cm), cm.sum(0), out=np.zeros(3), where=cm.sum(0) > 0)
    recall = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(3), where=cm.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=(precision + recall) > 0)
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "sv_f1": float(f1[1:].mean()),
        **{f"{name}_precision": float(precision[i]) for i, name in enumerate(CLASS_NAMES)},
        **{f"{name}_recall": float(recall[i]) for i, name in enumerate(CLASS_NAMES)},
        **{f"{name}_f1": float(f1[i]) for i, name in enumerate(CLASS_NAMES)},
        "confusion_matrix": cm.tolist(),
    }
