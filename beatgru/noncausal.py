"""Non-causal control variants NC1 / NC2 of the locked BEATGRU pipeline.

These variants exist to price the strict-causal constraint on the paper's own
architecture: the backbone, loss, budget and evaluation are held fixed, and
*only the information set* changes.

NC1
    The 8-dimensional causal RR descriptor becomes 16-dimensional by appending
    features that require the **next** R peak: the post-RR ``R_{i+1} - R_i``,
    pre/post ratios, look-ahead means, and the corresponding differences.  The
    waveform window is untouched, so NC1 is causal in morphology but not in
    rhythm.

NC2
    NC1 plus two further relaxations of the same kind: the waveform and the RR
    descriptor are standardised with the **whole record's** mean and standard
    deviation instead of the current window's, and the history decoder becomes
    **bidirectional**, so a beat's decision can depend on later beats.

Nothing in this module is imported by the locked pipeline.  ``beatgru/data.py``,
``models.py`` and ``train.py`` are byte-identical to the archived run; the
control only *adds* this file and the matching tool.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import wfdb
from scipy.signal import resample
from torch import nn

from .data import Record, _preferred_lead, derivative_channels, direct_causal_rr

POST_RR_DIM = 8
TOTAL_RR_DIM = 8 + POST_RR_DIM


# --------------------------------------------------------------------------- #
# NC1: RR descriptor that reads one beat into the future
# --------------------------------------------------------------------------- #
def extended_rr(peaks: np.ndarray, sample_rate: int) -> np.ndarray:
    """``direct_causal_rr`` widened with post-RR / look-ahead features.

    The first eight columns are bit-identical to ``direct_causal_rr``; the
    extra eight need ``R_{i+1}`` and therefore could not be computed online at
    ``R_i + 200 ms``.  The final beat of a record has no successor; there the
    post-RR slot is filled with the current interval so the row stays defined,
    and the affected beat is reported in the run metadata.
    """
    peaks = np.asarray(peaks, dtype=np.int64)
    causal = direct_causal_rr(peaks, sample_rate)
    intervals = np.zeros(len(peaks), dtype=np.float64)
    if len(peaks) > 1:
        intervals[1:] = np.maximum(np.diff(peaks).astype(np.float64) / float(sample_rate), 1e-4)
    post = np.empty(len(peaks), dtype=np.float64)
    post[:-1] = intervals[1:]
    post[-1] = intervals[-1]
    previous = np.empty(len(peaks), dtype=np.float64)
    previous[0] = intervals[0] if len(peaks) else 1e-4
    if len(peaks) > 1:
        previous[1:] = intervals[:-1]
    width = np.minimum(4, np.maximum(len(peaks) - 1, 1))
    look_ahead = np.empty(len(peaks), dtype=np.float64)
    for index in range(len(peaks)):
        window = intervals[index + 1: index + 1 + width]
        look_ahead[index] = float(window.mean()) if window.size else post[index]
    recent = causal[:, 3].astype(np.float64)
    extra = np.stack((
        np.clip(post, .1, 5.),
        np.clip(post / np.maximum(intervals, 1e-4), .1, 10.),
        np.clip(post / np.maximum(previous, 1e-4), .1, 10.),
        np.clip(post - intervals, -3., 3.),
        np.clip(post - recent, -3., 3.),
        np.clip(look_ahead, .1, 5.),
        np.clip(look_ahead / np.maximum(recent, 1e-4), .1, 10.),
        np.ones(len(peaks), dtype=np.float64),
    ), axis=1)
    return np.concatenate((causal, extra), axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# NC2: whole-record statistics
# --------------------------------------------------------------------------- #
def record_standardize(rr: np.ndarray) -> np.ndarray:
    """Standardise every RR column with that record's own mean and sd."""
    rr = np.asarray(rr, dtype=np.float32)
    if len(rr) == 0:
        return rr
    mean = rr.mean(axis=0, keepdims=True)
    std = rr.std(axis=0, keepdims=True)
    return ((rr - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def causal_windows_recordnorm(signal: np.ndarray, peaks: np.ndarray, source_fs: int, target_fs: int,
                              lookback_ms: int, delay_ms: int) -> np.ndarray:
    """``causal_windows`` with whole-record instead of per-window z-scoring."""
    signal = np.nan_to_num(np.asarray(signal, dtype=np.float32))
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    normalised = ((signal - mean) / np.maximum(std, 1e-6)).astype(np.float32)
    pre_target = round(lookback_ms * target_fs / 1000)
    post_target = round(delay_ms * target_fs / 1000)
    pre = round(pre_target / target_fs * 1000.0 * source_fs / 1000)
    post = round(post_target / target_fs * 1000.0 * source_fs / 1000)
    width = pre_target + post_target + 1
    out = np.zeros((len(peaks), normalised.shape[1], width), dtype=np.float32)
    for index, peak in enumerate(np.asarray(peaks, dtype=np.int64)):
        start, end = int(peak) - pre, int(peak) + post + 1
        if end > len(normalised):
            raise ValueError("decision delay exceeds available waveform")
        source_start = max(start, 0)
        segment = normalised[source_start:end]
        if source_start > start:
            padding = (np.repeat(segment[:1], source_start - start, axis=0) if len(segment)
                       else np.zeros((source_start - start, normalised.shape[1]), dtype=np.float32))
            segment = np.concatenate((padding, segment), axis=0)
        if len(segment) != width:
            segment = resample(segment, width, axis=0).astype(np.float32, copy=False)
        out[index] = segment.T
    return out


# --------------------------------------------------------------------------- #
# loader
# --------------------------------------------------------------------------- #
def load_records_noncausal(manifest_dir: str | Path, split: str, target_fs: int = 250,
                           lookback_ms: int = 350, delay_ms: int = 200,
                           dataset: str = "mit_bih_arrhythmia", lead_selector: str = "canonical_first",
                           record_ids: Iterable[str] | None = None, variant: str = "nc1") -> list[Record]:
    """Same record order, window geometry and label policy as ``load_records``.

    ``variant`` selects the information set: ``nc1`` widens the RR descriptor,
    ``nc2`` additionally uses record-level standardisation.
    """
    if variant not in {"nc1", "nc2"}:
        raise ValueError("variant must be nc1 or nc2")
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
    for record_id, rows in sorted(grouped.items(), key=lambda pair: pair[0]):
        peaks = np.asarray([int(row["r_peak_sample"]) for row in rows], dtype=np.int64)
        if len(peaks) > 1 and np.any(np.diff(peaks) <= 0):
            raise ValueError(f"manifest sequence is not chronological: {record_id}")
        header = Path(rows[0]["header_path"])
        wf_record = wfdb.rdrecord(str(header.with_suffix("")))
        signal = _preferred_lead(wf_record, lead_selector)
        source_fs = int(round(float(wf_record.fs)))
        valid = peaks + round(delay_ms * source_fs / 1000) < len(signal)
        peaks = peaks[valid]
        all_labels = np.asarray([
            int(row["label"]) if row.get("label", "") != "" else
            {"N": 0, "S": 1, "V": 2}.get(row.get("aami_label", ""), -100)
            for row in rows], dtype=np.int64)
        score = np.asarray([
            int(row["is_scored"]) == 1 if row.get("is_scored", "") != "" else label >= 0
            for row, label in zip(rows, all_labels)], dtype=bool)[valid]
        labels = all_labels[valid]
        cutter = causal_windows_recordnorm if variant == "nc2" else None
        if cutter is None:
            from .data import causal_windows
            wave = derivative_channels(causal_windows(signal, peaks, source_fs, target_fs, lookback_ms, delay_ms))
        else:
            wave = derivative_channels(cutter(signal, peaks, source_fs, target_fs, lookback_ms, delay_ms))
        rr = extended_rr(peaks, source_fs)
        if variant == "nc2":
            rr = record_standardize(rr)
        result.append(Record(record_id, wave, rr, labels, score))
    if not result:
        raise ValueError(f"no records for split={split}")
    return result


# --------------------------------------------------------------------------- #
# NC2 decoder: bidirectional over the record
# --------------------------------------------------------------------------- #
class BidirectionalLogitGRU(nn.Module):
    """Bidirectional version of ``LogitGRU``; identical head, wider state.

    Every other design choice of the locked decoder is preserved: the zero
    initialised bounded residual ``base_logits + a_max * delta(h)``, the same
    token construction, the same optimiser settings in the training tool.
    The recurrence direction is the only change, so any difference is
    attributable to the information set and not to capacity conventions.
    """
    def __init__(self, rr_dim: int = TOTAL_RR_DIM, hidden: int = 128,
                 max_correction: float = 0.75, n_classes: int = 3) -> None:
        super().__init__()
        self.max_correction = float(max_correction)
        self.input = nn.Sequential(nn.Linear(n_classes + rr_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.delta = nn.Linear(2 * hidden, n_classes)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward_chunk(self, base_logits: torch.Tensor, rr: torch.Tensor,
                      state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        token = self.input(torch.cat((base_logits, rr), dim=1))
        history, new_state = self.gru(token.unsqueeze(0), state)
        return base_logits + self.max_correction * self.delta(history.squeeze(0)), new_state


def decoder_for(variant: str, hidden: int, max_correction: float):
    from .models import LogitGRU
    return BidirectionalLogitGRU if variant == "nc2" else LogitGRU
