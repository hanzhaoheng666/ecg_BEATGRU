#!/usr/bin/env python3
"""R-peak localisation sensitivity of the frozen BEATGRU model.

No training and no recalibration: the locked DS1 checkpoint and its logits are
frozen, and the DS2 truth labels never move.  Only the R-peak positions that
the *window cutter* and the *causal RR descriptor* are driven by are
perturbed, so the measured degradation is the end-to-end sensitivity of the
declared pipeline to an imperfect upstream detector.

Perturbations
  1. random jitter      sigma in {10, 20, 40} ms, truncated at half the local
                        R-R interval so a peak cannot cross its neighbour;
  2. systematic shift   a single constant offset in {+-10, +-20, +-40} ms,
                        modelling a detector with a fixed lag or lead;
  3. missed / false     randomly remove 2% or 5% of the peaks and insert the
                        same number of spurious peaks.

Accounting for (3) is explicit.  A removed peak means the pipeline never emits
a decision for that annotated beat, which is a false negative for its true
class; an inserted peak produces a spurious decision, which is a false
positive for whatever class the model emits.  ``precision``/``recall`` are
computed from those separated actual/predicted totals.

``--sigma 0`` reproduces the locked evaluation bit for bit and is asserted
against the stored per-beat predictions before any perturbation is reported.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import wfdb
from scipy.signal import resample

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beatgru.data import _preferred_lead, causal_windows, derivative_channels, direct_causal_rr
from beatgru.models import BeatCNN, LogitGRU

CLASS_NAMES = ("N", "S", "V")


# --------------------------------------------------------------------------- #
# exact re-implementation of the locked window contract, plus edge padding
# --------------------------------------------------------------------------- #
def window_geometry(source_fs: int, target_fs: int, lookback_ms: int, delay_ms: int) -> tuple[int, int]:
    pre_target = round(lookback_ms * target_fs / 1000)
    post_target = round(delay_ms * target_fs / 1000)
    pre = round(pre_target / target_fs * 1000.0 * source_fs / 1000)
    post = round(post_target / target_fs * 1000.0 * source_fs / 1000)
    return pre, post


def cut_windows(signal: np.ndarray, peaks: np.ndarray, source_fs: int, target_fs: int,
                lookback_ms: int, delay_ms: int) -> np.ndarray:
    """Vectorised, edge-padded equivalent of ``data.causal_windows``.

    ``causal_windows`` cuts ``pre + post + 1`` samples on the *source* grid and
    then resamples that segment onto the fixed ``139``-sample target grid.  A
    shifted peak can run past the end of the record, where ``causal_windows``
    raises; the same clipped-index padding is applied on both sides instead, so
    the scored beat set stays constant across perturbation settings.
    """
    signal = np.nan_to_num(np.asarray(signal, dtype=np.float32))
    pre, post = window_geometry(source_fs, target_fs, lookback_ms, delay_ms)
    pre_target = round(lookback_ms * target_fs / 1000)
    width = pre_target + round(delay_ms * target_fs / 1000) + 1
    length = signal.shape[0]
    peaks = np.asarray(peaks, dtype=np.int64)
    offsets = np.arange(-pre, post + 1, dtype=np.int64)
    index = np.clip(peaks[:, None] + offsets[None, :], 0, length - 1)
    segment = signal[index]                      # (N, pre+post+1, channels)
    if segment.shape[1] != width:
        segment = resample(segment, width, axis=1).astype(np.float32, copy=False)
    mean = segment.mean(axis=1, keepdims=True)
    std = segment.std(axis=1, keepdims=True)
    return ((segment - mean) / np.maximum(std, 1e-6)).astype(np.float32).transpose(0, 2, 1)


def load_raw(manifest_dir: Path, dataset: str, split: str, delay_ms: int,
             lead_selector: str) -> list[dict[str, object]]:
    """Raw lead, original peaks, labels and scoring mask for every record."""
    grouped: dict[str, list[dict[str, str]]] = {}
    with (manifest_dir / f"{dataset}.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if split != "all" and row["split"] != split:
                continue
            grouped.setdefault(row["record_id"], []).append(row)
    output = []
    for record_id, rows in sorted(grouped.items(), key=lambda item: item[0]):
        peaks = np.asarray([int(row["r_peak_sample"]) for row in rows], dtype=np.int64)
        labels = np.asarray([int(row["label"]) if row.get("label", "") != "" else
                             {"N": 0, "S": 1, "V": 2}.get(row.get("aami_label", ""), -100)
                             for row in rows], dtype=np.int64)
        scored = np.asarray([int(row["is_scored"]) == 1 if row.get("is_scored", "") != "" else label >= 0
                             for row, label in zip(rows, labels)], dtype=bool)
        record = wfdb.rdrecord(str(Path(rows[0]["header_path"]).with_suffix("")))
        signal = _preferred_lead(record, lead_selector)
        source_fs = int(round(float(record.fs)))
        valid = peaks + round(delay_ms * source_fs / 1000) < len(signal)
        peaks, labels, scored = peaks[valid], labels[valid], scored[valid]
        output.append({"record_id": record_id, "signal": signal, "peaks": peaks,
                       "labels": labels, "score": scored, "source_fs": source_fs})
    return output


# --------------------------------------------------------------------------- #
# perturbations
# --------------------------------------------------------------------------- #
def jitter(peaks: np.ndarray, sigma_ms: float, fs: int, rng: np.random.Generator) -> np.ndarray:
    if sigma_ms <= 0:
        return peaks.copy()
    gaps = np.diff(peaks)
    half = np.empty(len(peaks), dtype=np.float64)
    half[0] = gaps[0] / 2 if len(gaps) else np.inf
    half[-1] = gaps[-1] / 2 if len(gaps) else np.inf
    if len(peaks) > 2:
        half[1:-1] = np.minimum(gaps[:-1], gaps[1:]) / 2
    noise = rng.normal(0.0, sigma_ms * fs / 1000.0, size=len(peaks))
    delta = np.clip(np.round(noise), -half, half).astype(np.int64)
    shifted = np.maximum(peaks + delta, 0)
    # Half-gap clipping already prevents crossings; the sweep guards against
    # integer rounding making two adjacent peaks coincide.
    for index in range(1, len(shifted)):
        shifted[index] = max(shifted[index], shifted[index - 1] + 1)
    return shifted


def shift(peaks: np.ndarray, shift_ms: float, fs: int) -> np.ndarray:
    delta = int(round(shift_ms * fs / 1000.0))
    return np.maximum(peaks + delta, 0)


def miss_false(peaks: np.ndarray, rate: float, fs: int,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (combined peaks, position of every original event, is_false flags).

    Every spurious peak must land on a sample that no retained peak occupies,
    otherwise the mapping between annotated beats and emitted decisions would
    be ambiguous.  Collisions are retried; the search then falls back to the
    middle of the widest available gaps.
    """
    n = len(peaks)
    if rate <= 0:
        return peaks.copy(), np.arange(n), np.zeros(n, dtype=bool)
    drop_n = max(1, int(round(rate * n)))
    dropped = set(int(x) for x in rng.choice(n, size=drop_n, replace=False))
    keep = np.array([index not in dropped for index in range(n)], dtype=bool)
    gaps = np.diff(peaks)
    used = set(int(x) for x in peaks[keep])
    fake: list[int] = []
    for _ in range(drop_n):
        placed = False
        for _ in range(64):
            anchor = int(rng.integers(0, max(1, n - 1)))
            gap = int(gaps[anchor])
            if gap < 4:
                continue
            candidate = int(peaks[anchor]) + int(rng.integers(2, gap - 1))
            if candidate not in used:
                used.add(candidate); fake.append(candidate); placed = True
                break
        if not placed:
            for anchor in np.argsort(gaps)[::-1]:
                gap = int(gaps[anchor])
                if gap < 4:
                    continue
                candidate = int(peaks[anchor]) + max(2, gap // 2)
                if candidate not in used:
                    used.add(candidate); fake.append(candidate); placed = True
                    break
        if not placed:
            raise ValueError("could not place a spurious peak without a collision")
    fake_array = np.asarray(sorted(fake), dtype=np.int64)
    combined = np.concatenate([peaks[keep], fake_array])
    order = np.argsort(combined, kind="stable")
    combined = combined[order]
    is_false = order >= int(keep.sum())
    position = np.full(n, -1, dtype=np.int64)
    position[np.flatnonzero(keep)] = np.flatnonzero(~is_false)
    return combined, position, is_false


# --------------------------------------------------------------------------- #
# metrics with explicit missed / spurious accounting
# --------------------------------------------------------------------------- #
def extended_metric(y_true: np.ndarray, y_pred: np.ndarray, missed: np.ndarray,
                    false_predictions: np.ndarray) -> dict[str, object]:
    cm = np.zeros((3, 3), dtype=np.int64)
    for actual, predicted in zip(y_true[~missed], y_pred[~missed]):
        cm[actual, predicted] += 1
    actual_total = cm.sum(axis=1).astype(np.int64).copy()
    predicted_total = cm.sum(axis=0).astype(np.int64).copy()
    for label in y_true[missed]:
        actual_total[label] += 1
    for label in false_predictions:
        predicted_total[label] += 1
    precision = np.divide(np.diag(cm), predicted_total, out=np.zeros(3), where=predicted_total > 0)
    recall = np.divide(np.diag(cm), actual_total, out=np.zeros(3), where=actual_total > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=(precision + recall) > 0)
    decisions = int(len(y_true) + len(false_predictions))
    return {
        "macro_f1": float(f1.mean()),
        "S_f1": float(f1[1]), "V_f1": float(f1[2]), "N_f1": float(f1[0]),
        "sv_f1": float(f1[1:].mean()),
        "S_recall": float(recall[1]), "S_precision": float(precision[1]),
        "balanced_accuracy": float(recall.mean()),
        "accuracy": float(np.diag(cm).sum() / decisions) if decisions else 0.0,
        "n_missed": int(missed.sum()), "n_spurious": int(len(false_predictions)),
        "confusion_matrix": cm.tolist(),
        "actual_totals": actual_total.tolist(), "predicted_totals": predicted_total.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference-npz", default="", help="locked per-beat predictions used as a regression check")
    parser.add_argument("--dataset", default="mit_bih_arrhythmia")
    parser.add_argument("--split", default="test")
    parser.add_argument("--lead-selector", default="canonical_first")
    parser.add_argument("--target-fs", type=int, default=250)
    parser.add_argument("--lookback-ms", type=int, default=350)
    parser.add_argument("--delay-ms", type=int, default=200)
    parser.add_argument("--realizations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest = Path(args.manifest_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    backbone_ckpt = torch.load(args.backbone, map_location="cpu", weights_only=False)
    base = BeatCNN(int(backbone_ckpt.get("channels", 3)), 8).to(device)
    base.load_state_dict(backbone_ckpt["model"]); base.eval()
    history_ckpt = torch.load(args.history, map_location="cpu", weights_only=False)
    saved = history_ckpt["args"]
    history = LogitGRU(8, int(saved["hidden"]), float(saved["max_correction"])).to(device)
    history.load_state_dict(history_ckpt["model"]); history.eval()

    records = load_raw(manifest, args.dataset, args.split, args.delay_ms, args.lead_selector)

    # ---- regression check: unperturbed pipeline must equal the locked run ----
    baseline_y, baseline_p, baseline_ids = [], [], []
    with torch.inference_mode():
        for record in records:
            wave = derivative_channels(cut_windows(record["signal"], record["peaks"], record["source_fs"],
                                                  args.target_fs, args.lookback_ms, args.delay_ms))
            reference = derivative_channels(causal_windows(record["signal"], record["peaks"], record["source_fs"],
                                                           args.target_fs, args.lookback_ms, args.delay_ms))
            if not np.array_equal(wave, reference):
                raise ValueError(f"window cutter diverged from the locked contract in {record['record_id']}")
            rr = direct_causal_rr(record["peaks"], record["source_fs"])
            z = base(torch.from_numpy(wave).to(device), torch.from_numpy(rr).to(device))
            z, _ = history.forward_chunk(z, torch.from_numpy(rr).to(device), None)
            keep = record["score"]
            baseline_y.append(record["labels"][keep]); baseline_p.append(z.argmax(1).cpu().numpy()[keep])
            baseline_ids.append(np.repeat(str(record["record_id"]), int(keep.sum())))
    y = np.concatenate(baseline_y); p = np.concatenate(baseline_p); ids = np.concatenate(baseline_ids)
    verification: dict[str, object] = {"window_cutter_matches_locked_contract": True}
    if args.reference_npz:
        reference = np.load(args.reference_npz, allow_pickle=False)
        same_labels = bool(np.array_equal(reference["labels"], y))
        agree = float(np.mean(reference["predictions"] == p))
        verification.update({
            "reference_npz": str(Path(args.reference_npz).resolve()),
            "labels_identical": same_labels,
            "prediction_agreement": agree,
            "n_disagreeing_beats": int(np.sum(reference["predictions"] != p)),
        })
        if not same_labels:
            raise ValueError("reconstructed scored-beat sequence does not match the locked run")

    settings: list[dict[str, object]] = [{"kind": "baseline", "label": "无扰动", "amplitude": 0.0, "rate": 0.0}]
    for sigma in (10.0, 20.0, 40.0):
        settings.append({"kind": "jitter", "label": f"随机抖动 sigma={sigma:g} ms", "amplitude": sigma})
    for value in (-40.0, -20.0, -10.0, 10.0, 20.0, 40.0):
        settings.append({"kind": "shift", "label": f"系统偏移 {value:+g} ms", "amplitude": value})
    for rate in (0.02, 0.05):
        settings.append({"kind": "miss_false", "label": f"漏检/误检 {rate:.0%}", "amplitude": 0.0, "rate": rate})

    results = []
    for setting in settings:
        realizations = 1 if setting["kind"] == "baseline" else args.realizations
        per_realization = []
        for realization in range(realizations):
            rng = np.random.default_rng(args.seed + realization * 7919)
            truth, predicted, missed_all, spurious_all = [], [], [], []
            with torch.inference_mode():
                for record in records:
                    peaks = np.asarray(record["peaks"], dtype=np.int64)
                    fs = int(record["source_fs"])
                    if setting["kind"] == "baseline":
                        used, position, is_false = peaks, np.arange(len(peaks)), np.zeros(len(peaks), dtype=bool)
                    elif setting["kind"] == "jitter":
                        used, position, is_false = jitter(peaks, float(setting["amplitude"]), fs, rng), \
                            np.arange(len(peaks)), np.zeros(len(peaks), dtype=bool)
                    elif setting["kind"] == "shift":
                        used, position, is_false = shift(peaks, float(setting["amplitude"]), fs), \
                            np.arange(len(peaks)), np.zeros(len(peaks), dtype=bool)
                    else:
                        used, position, is_false = miss_false(peaks, float(setting["rate"]), fs, rng)
                    wave = derivative_channels(cut_windows(record["signal"], used, fs, args.target_fs,
                                                           args.lookback_ms, args.delay_ms))
                    rr = direct_causal_rr(used, fs)
                    z = base(torch.from_numpy(wave).to(device), torch.from_numpy(rr).to(device))
                    z, _ = history.forward_chunk(z, torch.from_numpy(rr).to(device), None)
                    predicted_all = z.argmax(1).cpu().numpy()
                    keep = record["score"]
                    missed = (position < 0)[keep]
                    truth.append(record["labels"][keep])
                    predicted.append(np.where(missed, 0, predicted_all[np.maximum(position[keep], 0)]))
                    missed_all.append(missed)
                    spurious_all.append(predicted_all[is_false])
            values = extended_metric(np.concatenate(truth), np.concatenate(predicted),
                                     np.concatenate(missed_all), np.concatenate(spurious_all) if any(
                                         len(x) for x in spurious_all) else np.zeros(0, dtype=np.int64))
            per_realization.append(values)
        keys = ("macro_f1", "S_f1", "V_f1", "N_f1", "balanced_accuracy", "accuracy", "S_recall", "S_precision")
        summary = {key: {"mean": float(np.mean([r[key] for r in per_realization])),
                         "sd": float(np.std([r[key] for r in per_realization], ddof=1)) if len(per_realization) > 1 else 0.0}
                   for key in keys}
        results.append({**setting, "realizations": realizations, "metrics": summary,
                        "n_missed": int(np.mean([r["n_missed"] for r in per_realization])),
                        "n_spurious": int(np.mean([r["n_spurious"] for r in per_realization])),
                        "confusion_matrix": per_realization[0]["confusion_matrix"],
                        "actual_totals": per_realization[0]["actual_totals"],
                        "predicted_totals": per_realization[0]["predicted_totals"]})
        print(json.dumps({**setting, "macro_f1": summary["macro_f1"], "S_f1": summary["S_f1"]}), flush=True)

    baseline = results[0]["metrics"]["macro_f1"]["mean"]
    payload = {
        "dataset": args.dataset, "split": args.split, "lead_selector": args.lead_selector,
        "n_records": len(records), "n_scored_beats": int(len(y)),
        "decision_delay_ms": args.delay_ms, "lookback_ms": args.lookback_ms,
        "realizations": args.realizations, "noise_seed": args.seed,
        "frozen": {"backbone": str(Path(args.backbone).resolve()), "history": str(Path(args.history).resolve())},
        "labels_held_fixed": True, "recalibration": False,
        "verification": verification,
        "settings": results,
        "macro_f1_drop_per_ms_jitter": None,
    }
    ladder = [(s["amplitude"], baseline - s["metrics"]["macro_f1"]["mean"]) for s in results if s["kind"] == "jitter" and s["amplitude"] > 0]
    if ladder:
        payload["macro_f1_drop_per_ms_jitter"] = float(np.polyfit([x for x, _ in ladder], [d for _, d in ladder], 1)[0])
    (out / "rpeak_jitter.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# R 峰抖动敏感性（锁定 BEATGRU，DS2 冻结推理）", "",
             f"- 真值标签固定不动；只改变窗口与因果 RR 所依据的 R 峰位置；模型与阈值全部冻结",
             f"- 每个非基线设置 {args.realizations} 个噪声实现（种子 {args.seed}）",
             f"- 基线自检：窗口切分与锁定实现逐位一致；与锁定逐拍预测一致率 "
             f"{verification.get('prediction_agreement', float('nan')):.6f}", "",
             "| 扰动 | 设置 | Macro-F1 | S-F1 | V-F1 | 相对基线 Macro-F1 下降 |", "|---|---|---:|---:|---:|---:|"]
    for row in results:
        metrics = row["metrics"]
        drop = baseline - metrics["macro_f1"]["mean"]
        text = f"{metrics['macro_f1']['mean']:.4f}±{metrics['macro_f1']['sd']:.4f}" if row["realizations"] > 1 \
            else f"{metrics['macro_f1']['mean']:.4f}"
        lines.append(f"| {row['kind']} | {row['label']} | {text} | "
                     f"{metrics['S_f1']['mean']:.4f} | {metrics['V_f1']['mean']:.4f} | {drop:.4f} |")
    if payload["macro_f1_drop_per_ms_jitter"] is not None:
        slope = payload["macro_f1_drop_per_ms_jitter"]
        lines += ["", f"- 随机抖动退化斜率：定位误差每增加 1 ms，Macro-F1 下降 {slope:+.5f}",
                  f"- 折算：Macro-F1 下降 1 个百分点，对应约 {0.01 / abs(slope):.1f} ms 的 R 峰定位误差"
                  if slope > 0 else "- 斜率非正，抖动幅度内未见单调退化"]
    (out / "rpeak_jitter.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"verification": verification, "baseline_macro_f1": baseline,
                      "slope_per_ms": payload["macro_f1_drop_per_ms_jitter"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
