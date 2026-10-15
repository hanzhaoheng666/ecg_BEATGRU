#!/usr/bin/env python3
"""Evaluate the causal beat front-end R-peak detector on beat manifests.

The detector is intentionally evaluated separately from BEATGRU classification.
BEATGRU remains conditional on the supplied/detected peak sequence; this tool
measures the error introduced when the supplied sequence is replaced by the
WFDB gqrs/xqrs front-end used by the current pipeline.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parents[1]
ECGCODE = ROOT.parent / "ECGCODE"
if str(ECGCODE) not in sys.path:
    sys.path.insert(0, str(ECGCODE))
from hytrans.datasets.rpeak_detection import detect_rpeaks


def groups(manifest_dir: Path, dataset: str, split: str) -> list[tuple[str, str, list[dict[str, str]]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    with (manifest_dir / f"{dataset}.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if split != "all" and row.get("split") != split:
                continue
            grouped.setdefault((row["record_id"], row["header_path"]), []).append(row)
    result = []
    for (record_id, header), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["event_index"]))
        peaks = np.asarray([int(row["r_peak_sample"]) for row in rows], dtype=np.int64)
        if len(peaks) > 1 and np.any(np.diff(peaks) <= 0):
            raise ValueError(f"non-monotonic reference peaks: {dataset}/{record_id}")
        result.append((record_id, header, rows))
    return result


def match_peaks(reference: np.ndarray, detected: np.ndarray, tolerance: int) -> tuple[int, int, int, np.ndarray]:
    reference = np.asarray(reference, dtype=np.int64)
    detected = np.asarray(detected, dtype=np.int64)
    i = j = tp = 0
    errors: list[int] = []
    while i < len(reference) and j < len(detected):
        delta = int(detected[j] - reference[i])
        if abs(delta) <= tolerance:
            tp += 1; errors.append(abs(delta)); i += 1; j += 1
        elif detected[j] < reference[i] - tolerance:
            j += 1
        else:
            i += 1
    return tp, len(detected) - tp, len(reference) - tp, np.asarray(errors, dtype=np.int64)


def scores(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    ppv = tp / max(tp + fp, 1)
    sensitivity = tp / max(tp + fn, 1)
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "ppv": float(ppv),
            "sensitivity": float(sensitivity), "f1": float(2 * ppv * sensitivity / max(ppv + sensitivity, 1e-12))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["mit_bih_arrhythmia", "mbit", "mit_bih_svdb", "st_petersburg_incart"])
    parser.add_argument("--split", default="test", help="manifest split; use all for every record")
    parser.add_argument("--max-records", type=int, default=0, help="0 means all records")
    parser.add_argument("--tolerance-ms", type=float, default=100.0)
    parser.add_argument("--lead-index", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, object]] = []
    dataset_out: dict[str, object] = {}
    for dataset in args.datasets:
        records = groups(args.manifest_dir, dataset, args.split)
        if args.max_records > 0:
            records = records[:args.max_records]
        total = {"tp": 0, "fp": 0, "fn": 0}
        errors: list[int] = []
        failures = []
        for index, (record_id, header, manifest_rows) in enumerate(records, start=1):
            try:
                record = wfdb.rdrecord(str(Path(header).with_suffix("")))
                signal = record.p_signal if record.p_signal is not None else record.d_signal
                signal = np.asarray(signal, dtype=np.float32)
                fs = int(round(float(record.fs)))
                reference = np.asarray([int(row["r_peak_sample"]) for row in manifest_rows], dtype=np.int64)
                detected = detect_rpeaks(signal, fs, lead_index=args.lead_index)
                tolerance = max(1, round(args.tolerance_ms * fs / 1000.0))
                tp, fp, fn, local_errors = match_peaks(reference, detected, tolerance)
                total["tp"] += tp; total["fp"] += fp; total["fn"] += fn; errors.extend(local_errors.tolist())
                item = {"dataset": dataset, "record_id": record_id, "source_fs": fs, "reference_peaks": len(reference), "detected_peaks": len(detected), "tolerance_samples": tolerance, **scores(tp, fp, fn), "median_abs_error_ms": float(np.median(local_errors) * 1000 / fs) if len(local_errors) else None, "p95_abs_error_ms": float(np.quantile(local_errors, .95) * 1000 / fs) if len(local_errors) else None}
                rows_out.append(item)
            except Exception as exc:  # keep the dataset report complete and explicit
                failures.append({"record_id": record_id, "error": repr(exc)})
            if index % 100 == 0:
                print(json.dumps({"dataset": dataset, "processed": index, "total": len(records)}), flush=True)
        aggregate = scores(**total)
        aggregate.update({"dataset": dataset, "split": args.split, "n_records": len(records), "n_failures": len(failures), "failures": failures,
                          "reference_peaks": int(total["tp"] + total["fn"]), "detected_peaks": int(total["tp"] + total["fp"]),
                          "tolerance_ms": args.tolerance_ms, "median_abs_error_ms": float(np.median(errors) * 1000 / np.mean([r["source_fs"] for r in rows_out if r["dataset"] == dataset])) if errors else None})
        dataset_out[dataset] = aggregate
    with (args.output_dir / "rpeak_per_record.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in rows_out for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows_out)
    payload = {"manifest_dir": str(args.manifest_dir.resolve()), "split": args.split, "detector": "hytrans.datasets.rpeak_detection.detect_rpeaks (WFDB gqrs/xqrs, scipy fallback)", "tolerance_ms": args.tolerance_ms, "datasets": dataset_out}
    (args.output_dir / "rpeak_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
