"""MIT-BIH DS1/DS2 data construction and strict-causal features.

At beat i the waveform ends at R_i + decision_delay; RR descriptors use only
R_1..R_i.  F/Q events remain in the record sequence but are not scored.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import wfdb
from scipy.signal import resample


DS1 = ("101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122", "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230")
DS2 = ("100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234")
AAMI = {"N": 0, "L": 0, "R": 0, "e": 0, "j": 0, "A": 1, "a": 1, "J": 1, "S": 1, "V": 2, "E": 2, "F": -100, "/": -100, "f": -100, "Q": -100}


def build_manifest(raw_root: str | Path, out_dir: str | Path) -> None:
    """Build the conventional public MIT-BIH DS1/DS2 event manifest."""
    raw_root, out_dir = Path(raw_root).resolve(), Path(out_dir).resolve()
    headers = sorted(raw_root.glob("*.hea"), key=lambda p: int(p.stem))
    if len(headers) != 48:
        raise RuntimeError(f"expected 48 MIT-BIH headers in {raw_root}, found {len(headers)}")
    rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for header in headers:
        record_id = header.stem
        if record_id not in DS1 and record_id not in DS2:
            continue
        split = "train" if record_id in DS1 else "test"
        annotation = wfdb.rdann(str(header.with_suffix("")), "atr")
        events = [(int(sample), str(symbol), AAMI[str(symbol)]) for sample, symbol in zip(annotation.sample, annotation.symbol) if str(symbol) in AAMI]
        peaks = [x[0] for x in events]
        if any(right <= left for left, right in zip(peaks, peaks[1:])):
            raise ValueError(f"non-monotonic R peaks in {record_id}")
        first = header.read_text(encoding="utf-8").splitlines()[0].split()
        fs = int(float(first[2]))
        counts = {name: sum(label == value for _, _, label in events) for name, value in (("N", 0), ("S", 1), ("V", 2), ("unscored", -100))}
        summary.append({"record_id": record_id, "split": split, "n_events": len(events), **counts})
        for index, (peak, symbol, label) in enumerate(events):
            rows.append({"record_id": record_id, "split": split, "header_path": str(header), "source_fs": fs,
                         "event_index": index, "r_peak_sample": peak, "raw_symbol": symbol,
                         "label": label, "is_scored": int(label >= 0)})
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, values, fields in ((out_dir / "mit_bih_arrhythmia.csv", rows, list(rows[0])), (out_dir / "record_summary.csv", summary, list(summary[0]))):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(values)
    info = {"protocol": "beatgru_public_ds1ds2_causal", "dataset": "MIT-BIH Arrhythmia", "split_name": "standard inter-patient DS1/DS2",
            "raw_root": str(raw_root), "records_by_split": {"train": len(DS1), "test": len(DS2)},
            "ds1_train": list(DS1), "ds2_test": list(DS2),
            "label_policy": "AAMI N/S/V scored; F/Q history-only", "conditional_on_detected_or_annotated_rpeaks": True}
    (out_dir / "event_manifest.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def direct_causal_rr(peaks: np.ndarray, sample_rate: int) -> np.ndarray:
    """Eight RR features at i from current and past R peaks only."""
    peaks = np.asarray(peaks, dtype=np.int64)
    if peaks.ndim != 1 or (len(peaks) > 1 and np.any(np.diff(peaks) <= 0)):
        raise ValueError("R peaks must be strictly increasing")
    result = np.zeros((len(peaks), 8), dtype=np.float32)
    history: list[float] = []
    for index in range(1, len(peaks)):
        rr = max(float(peaks[index] - peaks[index - 1]) / float(sample_rate), 1e-4)
        history.append(rr); values = np.asarray(history, dtype=np.float64)
        previous = history[-2] if len(history) > 1 else rr
        recent = float(values[-8:].mean())
        older = float(values[-16:-8].mean()) if len(values) > 8 else recent
        result[index] = (np.clip(rr, .1, 5.), np.clip(previous, .1, 5.), np.clip(rr / max(previous, 1e-4), .1, 10.),
                         np.clip(recent, .1, 5.), np.clip(rr - recent, -3., 3.), np.clip(recent - older, -3., 3.),
                         np.log1p(len(history)) / 8., 1.)
    return result


def causal_windows(signal: np.ndarray, peaks: np.ndarray, source_fs: int, target_fs: int,
                   lookback_ms: int, delay_ms: int) -> np.ndarray:
    """Cut every raw window before resampling; no future waveform reaches scipy."""
    signal = np.nan_to_num(np.asarray(signal, dtype=np.float32))
    # Quantize first in the target grid, then map that exact window duration
    # back to the source grid.  This matches the fixed target-window contract
    # (350 ms at 250 Hz is 88 samples, i.e. an effective 352 ms source cut).
    pre_target = round(lookback_ms * target_fs / 1000)
    post_target = round(delay_ms * target_fs / 1000)
    effective_pre_ms = pre_target / target_fs * 1000.0
    effective_post_ms = post_target / target_fs * 1000.0
    pre = round(effective_pre_ms * source_fs / 1000)
    post = round(effective_post_ms * source_fs / 1000)
    width = pre_target + post_target + 1
    out = np.zeros((len(peaks), signal.shape[1], width), dtype=np.float32)
    for index, peak in enumerate(np.asarray(peaks, dtype=np.int64)):
        start, end = int(peak) - pre, int(peak) + post + 1
        if end > len(signal):
            raise ValueError("decision delay exceeds available waveform")
        source_start = max(start, 0); segment = signal[source_start:end]
        if source_start > start:
            padding = np.repeat(segment[:1], source_start - start, axis=0) if len(segment) else np.zeros((source_start - start, signal.shape[1]), dtype=np.float32)
            segment = np.concatenate((padding, segment), axis=0)
        if len(segment) != width:
            segment = resample(segment, width, axis=0).astype(np.float32, copy=False)
        mean, std = segment.mean(axis=0, keepdims=True), segment.std(axis=0, keepdims=True)
        out[index] = ((segment - mean) / np.maximum(std, 1e-6)).T
    return out


def derivative_channels(wave: np.ndarray) -> np.ndarray:
    first = np.diff(wave, axis=-1, prepend=wave[..., :1])
    second = np.diff(first, axis=-1, prepend=first[..., :1])
    return np.concatenate((wave, first, second), axis=1).astype(np.float32, copy=False)


def _preferred_lead(record: wfdb.Record, selector: str = "canonical_first") -> np.ndarray:
    signal = record.p_signal if record.p_signal is not None else record.d_signal
    signal = np.asarray(signal, dtype=np.float32)
    aliases = {"MLII": "II", "MII": "II", "MLIII": "III", "MCL1": "V1"}
    names = [aliases.get(str(x).upper(), str(x).upper()) for x in record.sig_name]
    # Match the original canonical-lead loader: use the first *canonical*
    # available lead, not source-file order (record 102 has V5,V2 and thus
    # uses V2).  ``source_matched_ii`` is a pre-declared source/target
    # acquisition rule: the released MIT-BIH model was trained on MLII, hence
    # an external database with a named lead II must use that signal.  It does
    # not inspect target labels, predictions, or target-wide statistics.
    canonical = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
    known = [(canonical.index(name), index) for index, name in enumerate(names) if name in canonical]
    if selector == "file_second":
        return signal[:, 1:2] if signal.shape[1] > 1 else signal[:, :1]
    if selector == "source_matched_precordial":
        index = next((names.index(name) for name in ("V1", "V2", "V3", "V4", "V5", "V6") if name in names),
                     1 if signal.shape[1] > 1 else 0)
        return signal[:, index:index + 1]
    if selector == "source_matched_pair":
        # Source/INCART: named II plus a precordial lead.  SVDB has only
        # ECG1/ECG2, so its deterministic file order is used.  This rule is
        # declared before evaluation and never observes labels or statistics.
        primary = names.index("II") if "II" in names else 0
        secondary = next((names.index(name) for name in ("V1", "V2", "V3", "V4", "V5", "V6")
                          if name in names and names.index(name) != primary), None)
        if secondary is None:
            secondary = next((i for i in range(signal.shape[1]) if i != primary), primary)
        return signal[:, [primary, secondary]]
    if selector in {"lead_ii", "source_matched_ii"} and "II" in names:
        index = names.index("II")
    elif selector == "canonical_first":
        index = min(known)[1] if known else 0
    else:
        if selector not in {"canonical_first", "lead_ii", "source_matched_ii", "source_matched_pair", "file_second", "source_matched_precordial"}:
            raise ValueError("invalid lead selector")
        index = min(known)[1] if known else 0
    return signal[:, index:index + 1]


@dataclass
class Record:
    record_id: str
    wave: np.ndarray
    rr: np.ndarray
    labels: np.ndarray
    score: np.ndarray


def load_records(manifest_dir: str | Path, split: str, target_fs: int = 250,
                 lookback_ms: int = 350, delay_ms: int = 200,
                 dataset: str = "mit_bih_arrhythmia", lead_selector: str = "canonical_first",
                 record_ids: Iterable[str] | None = None) -> list[Record]:
    """Load records chronologically.  This function never reorders event rows."""
    root = Path(manifest_dir)
    wanted = None if record_ids is None else {str(value).strip() for value in record_ids if str(value).strip()}
    with (root / f"{dataset}.csv").open(encoding="utf-8", newline="") as handle:
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in csv.DictReader(handle):
            if (split == "all" or row["split"] == split or
                    (split == "train_plus_val" and row["split"] == "train")):
                if wanted is None or row["record_id"] in wanted:
                    grouped.setdefault(row["record_id"], []).append(row)
    result: list[Record] = []
    # MIT-BIH/SVDB IDs are numeric whereas INCART uses I01...I75.  Sort for
    # deterministic record order without assuming a numeric filename.
    for record_id, rows in sorted(grouped.items(), key=lambda pair: pair[0]):
        # The compact BEATGRU manifest stores an explicit event_index; the
        # audited strict-online external manifest intentionally stores only
        # chronological CSV rows.  In the latter case CSV row order is the
        # event order and is verified again by monotonic R peaks below.
        event_ids = [int(row["event_index"]) if row.get("event_index", "") != "" else i
                     for i, row in enumerate(rows)]
        peaks = np.asarray([int(row["r_peak_sample"]) for row in rows], dtype=np.int64)
        if event_ids != list(range(len(rows))) or (len(peaks) > 1 and np.any(np.diff(peaks) <= 0)):
            raise ValueError(f"manifest sequence is not chronological: {record_id}")
        header = Path(rows[0]["header_path"])
        wf_record = wfdb.rdrecord(str(header.with_suffix("")))
        signal = _preferred_lead(wf_record, lead_selector)
        source_fs = int(round(float(wf_record.fs)))
        # ``delay_ms`` is converted to samples on the original source grid.
        valid = peaks + round(delay_ms * source_fs / 1000) < len(signal)
        peaks = peaks[valid]
        # Accept both this package's compact manifest and the project's prior
        # audited manifest, so existing DS1/DS2 data can be reused verbatim.
        all_labels = np.asarray([
            int(row["label"]) if row.get("label", "") != "" else
            {"N": 0, "S": 1, "V": 2}.get(row.get("aami_label", ""), -100)
            for row in rows
        ], dtype=np.int64)
        score = np.asarray([
            int(row["is_scored"]) == 1 if row.get("is_scored", "") != "" else label >= 0
            for row, label in zip(rows, all_labels)
        ], dtype=bool)[valid]
        labels = all_labels[valid]
        result.append(Record(record_id, derivative_channels(causal_windows(signal, peaks, source_fs, target_fs, lookback_ms, delay_ms)),
                             direct_causal_rr(peaks, source_fs), labels, score))
    if not result:
        raise ValueError(f"no records for split={split}" + ("" if wanted is None else f", requested_records={len(wanted)}"))
    return result


def tensors(record: Record, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (torch.from_numpy(record.wave).to(device), torch.from_numpy(record.rr).to(device),
            torch.from_numpy(record.labels).to(device), torch.from_numpy(record.score).to(device))
