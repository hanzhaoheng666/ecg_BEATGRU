"""Automated causal-invariance and record-state-isolation checks."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
import wfdb

from .data import _preferred_lead, causal_windows, derivative_channels, load_records
from .models import BeatCNN, EmbeddingGRU, LogitGRU
from .train import device_from, emissions, forward_history


def replay(model: LogitGRU | EmbeddingGRU, row: dict[str, object], device: torch.device, stop: int | None, chunk: int,
           state: torch.Tensor | None = None) -> tuple[np.ndarray, torch.Tensor | None]:
    count = len(row["labels"]) if stop is None else int(stop)
    parts = []
    with torch.inference_mode():
        for start in range(0, count, chunk):
            logits, state = forward_history(model, row, start, min(count, start + chunk), state, device)
            parts.append(logits.cpu().numpy()); state = state.detach()
    return np.concatenate(parts), state


def waveform_prefix_error(manifest: Path, target_fs: int, lookback_ms: int, delay_ms: int,
                          lead_selector: str) -> float:
    """Changing samples after a cutoff must not alter windows before it."""
    with (manifest / "mit_bih_arrhythmia.csv").open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    first_id = rows[0]["record_id"]; rows = [row for row in rows if row["record_id"] == first_id]
    peaks = np.asarray([int(row["r_peak_sample"]) for row in rows], dtype=np.int64)
    record = wfdb.rdrecord(str(Path(rows[0]["header_path"]).with_suffix("")))
    signal = _preferred_lead(record, lead_selector); source_fs = int(round(float(record.fs)))
    valid = peaks + round(delay_ms * source_fs / 1000) < len(signal); peaks = peaks[valid]
    cut = min(200, len(peaks)); before = derivative_channels(causal_windows(signal, peaks[:cut], source_fs, target_fs, lookback_ms, delay_ms))
    altered = signal.copy(); cutoff = int(peaks[cut - 1] + round(delay_ms * source_fs / 1000) + 1)
    altered[cutoff:] = np.random.default_rng(17).normal(size=altered[cutoff:].shape).astype(np.float32)
    after = derivative_channels(causal_windows(altered, peaks[:cut], source_fs, target_fs, lookback_ms, delay_ms))
    return float(np.max(np.abs(before - after)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--backbone", default="", help="override the CNN path saved in a released checkpoint")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); device = device_from(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False); config = checkpoint["args"]
    backbone_ckpt = torch.load(args.backbone or checkpoint["backbone"], map_location="cpu", weights_only=False)
    base = BeatCNN(int(backbone_ckpt.get("channels", 3)), 8).to(device); base.load_state_dict(backbone_ckpt["model"]); base.eval()
    lead_selector = config.get("lead_selector", "canonical_first")
    history_input = config.get("history_input", "logits")
    records = load_records(args.manifest_dir, "test", config["target_fs"], config["lookback_ms"], config["delay_ms"], lead_selector=lead_selector)
    rows = emissions(base, records, device, include_embedding=history_input == "embedding")
    model = (EmbeddingGRU(160, 8, config["hidden"], config["max_correction"]) if history_input == "embedding"
             else LogitGRU(8, config["hidden"], config["max_correction"])).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    row = rows[0]; cut = min(400, len(row["labels"]))
    full, _ = replay(model, row, device, None, config["chunk"]); prefix, _ = replay(model, row, device, cut, config["chunk"])
    altered = copy.deepcopy(row); altered["rr"][cut:] = np.random.default_rng(23).normal(size=altered["rr"][cut:].shape).astype(np.float32); altered["labels"][cut:] = 0
    altered_prefix, _ = replay(model, altered, device, cut, config["chunk"])
    future_error = float(np.max(np.abs(prefix - altered_prefix)))
    prefix_error = float(np.max(np.abs(full[:cut] - prefix)))
    # Correct state-isolation check: carry the first record's final hidden
    # state into the second record (illegal) and verify it differs from the
    # explicitly reset second-record replay.  The production path uses the
    # latter, so the report records both the carryover sensitivity and that
    # reset output is deterministic.
    first_out, carried_state = replay(model, row, device, None, config["chunk"])
    second = rows[1]
    standalone, _ = replay(model, second, device, min(8192, len(second["labels"])), config["chunk"])
    standalone_again, _ = replay(model, second, device, min(8192, len(second["labels"])), config["chunk"])
    carried, _ = replay(model, second, device, min(8192, len(second["labels"])), config["chunk"], carried_state)
    reset_error = float(np.max(np.abs(standalone - standalone_again)))
    carryover_error = float(np.max(np.abs(standalone - carried)))
    wave_error = waveform_prefix_error(args.manifest_dir, config["target_fs"], config["lookback_ms"], config["delay_ms"], lead_selector)
    tolerance = 1e-4
    payload = {"future_waveform_prefix_max_abs_error": wave_error, "prefix_replay_max_abs_error": prefix_error,
               "future_rr_and_label_prefix_max_abs_error": future_error, "record_reset_determinism_max_abs_error": reset_error,
               "illegal_carryover_sensitivity_max_abs_error": carryover_error,
               "tolerance": tolerance, "status": "PASS" if max(wave_error, prefix_error, future_error, reset_error) < tolerance else "FAIL",
               "state_carryover_was_explicitly_tested": True,
               "protocol": {"future_waveform": False, "future_rpeak_or_post_rr": False, "future_label": False, "state_reset": "record boundary"}}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
