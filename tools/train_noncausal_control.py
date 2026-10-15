#!/usr/bin/env python3
"""Train, evaluate and audit the non-causal control variants NC1 / NC2.

The locked pipeline (``beatgru.train``, ``tools/run_history_ablation.py``) is
reproduced here with a single substitution: the information set.  Architecture,
loss, optimiser, epoch budget, data split, threshold policy and evaluation code
are unchanged, so the difference between the control and the locked M3 run is
attributable to the extra future information and nothing else.

Subcommands
    train-cnn   frozen-emission backbone, same recipe as the locked M1
    train-gru   history decoder; NC1 uses ``LogitGRU``, NC2 the bidirectional one
    evaluate    one frozen pass over a target split
    audit       four data-level causal tests, run on the control on purpose
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import wfdb
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beatgru.data import Record, _preferred_lead, causal_windows, derivative_channels
from beatgru.metrics import metric
from beatgru.models import BeatCNN, LogitGRU
from beatgru.noncausal import (TOTAL_RR_DIM, BidirectionalLogitGRU, causal_windows_recordnorm,
                               extended_rr, load_records_noncausal, record_standardize)
from beatgru.train import device_from, emissions, record_ids_from_file, seed_all, weights

VARIANTS = ("nc1", "nc2")


def build_decoder(variant: str, hidden: int, max_correction: float, device: torch.device) -> nn.Module:
    if variant == "nc2":
        return BidirectionalLogitGRU(TOTAL_RR_DIM, hidden, max_correction).to(device)
    return LogitGRU(TOTAL_RR_DIM, hidden, max_correction).to(device)


def cut(variant: str, signal: np.ndarray, peaks: np.ndarray, source_fs: int, target_fs: int,
        lookback_ms: int, delay_ms: int) -> np.ndarray:
    if variant == "nc2":
        return derivative_channels(causal_windows_recordnorm(signal, peaks, source_fs, target_fs,
                                                             lookback_ms, delay_ms))
    return derivative_channels(causal_windows(signal, peaks, source_fs, target_fs, lookback_ms, delay_ms))


def describe_rr(variant: str, peaks: np.ndarray, source_fs: int) -> np.ndarray:
    rr = extended_rr(peaks, source_fs)
    return record_standardize(rr) if variant == "nc2" else rr


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def run_cnn(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    device = device_from(args.device)
    selected = record_ids_from_file(args.record_ids_file)
    rows = load_records_noncausal(args.manifest_dir, "train_plus_val", args.target_fs, args.lookback_ms,
                                  args.delay_ms, lead_selector=args.lead_selector, record_ids=selected,
                                  variant=args.variant)
    x = np.concatenate([row.wave[row.score] for row in rows])
    rr = np.concatenate([row.rr[row.score] for row in rows])
    y = np.concatenate([row.labels[row.score] for row in rows])
    counts, class_weights = weights(rows, args.power)
    model = BeatCNN(int(x.shape[1]), TOTAL_RR_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(rr), torch.from_numpy(y)),
                        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for xb, rb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb.to(device), rb.to(device)), yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})
        print(json.dumps({**history[-1], "variant": args.variant, "stage": "cnn"}), flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "channels": int(x.shape[1]), "rr_dim": TOTAL_RR_DIM,
                "variant": args.variant, "class_counts": counts.tolist(), "args": vars(args),
                "protocol": {"non_causal_control": args.variant, "future_waveform": False,
                             "future_rpeak_or_post_rr": True, "future_label": False,
                             "record_level_statistics": args.variant == "nc2",
                             "bidirectional_history": args.variant == "nc2"}},
               out / "derivative_cnn.pt")
    (out / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def run_gru(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    device = device_from(args.device)
    backbone_ckpt = torch.load(args.backbone, map_location="cpu", weights_only=False)
    base = BeatCNN(int(backbone_ckpt.get("channels", 3)), TOTAL_RR_DIM).to(device)
    base.load_state_dict(backbone_ckpt["model"])
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad = False
    selected = record_ids_from_file(args.record_ids_file)
    records = load_records_noncausal(args.manifest_dir, "train_plus_val", args.target_fs, args.lookback_ms,
                                     args.delay_ms, lead_selector=args.lead_selector, record_ids=selected,
                                     variant=args.variant)
    rows = emissions(base, records, device)
    pseudo = [Record(str(row["record_id"]), np.empty((0,)), row["rr"], row["labels"], row["score"]) for row in rows]
    counts, class_weights = weights(pseudo, args.power)
    model = build_decoder(args.variant, args.hidden, args.max_correction, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); order = list(range(len(rows))); random.Random(args.seed + epoch).shuffle(order); losses = []
        for index in order:
            row = rows[index]; state = None; terms = []
            labels = torch.from_numpy(row["labels"]).to(device); keep = torch.from_numpy(row["score"]).to(device)
            optimizer.zero_grad(set_to_none=True)
            n_chunks = max(1, (len(row["labels"]) + args.chunk - 1) // args.chunk)
            for start in range(0, len(row["labels"]), args.chunk):
                logits = torch.from_numpy(row["logits"][start:start + args.chunk]).to(device)
                rr = torch.from_numpy(row["rr"][start:start + args.chunk]).to(device)
                predicted, state = model.forward_chunk(logits, rr, state)
                active = keep[start:start + args.chunk]
                if bool(active.any()):
                    terms.append(loss_fn(predicted[active], labels[start:start + args.chunk][active]) / n_chunks)
                if state is not None:
                    state = state.detach()
            if terms:
                loss = torch.stack(terms).sum()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})
        print(json.dumps({**history[-1], "variant": args.variant, "stage": "gru"}), flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "backbone": str(Path(args.backbone).resolve()),
                "variant": args.variant, "rr_dim": TOTAL_RR_DIM, "class_counts": counts.tolist(),
                "args": vars(args),
                "protocol": {"non_causal_control": args.variant,
                             "input": f"frozen CNN logits + {TOTAL_RR_DIM}-D RR + "
                                      f"{'bidirectional' if args.variant == 'nc2' else 'unidirectional'} GRU",
                             "future_waveform": False, "future_rpeak_or_post_rr": True, "future_label": False,
                             "record_level_statistics": args.variant == "nc2",
                             "bidirectional_history": args.variant == "nc2",
                             "state_reset": "record boundary"}},
               out / "history.pt")
    (out / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def run_evaluate(args: argparse.Namespace) -> None:
    device = device_from(args.device)
    backbone_ckpt = torch.load(args.backbone, map_location="cpu", weights_only=False)
    base = BeatCNN(int(backbone_ckpt.get("channels", 3)), TOTAL_RR_DIM).to(device)
    base.load_state_dict(backbone_ckpt["model"]); base.eval()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = ckpt["args"]
    hidden, correction = int(saved["hidden"]), float(saved["max_correction"])
    model = build_decoder(args.variant, hidden, correction, device)
    model.load_state_dict(ckpt["model"]); model.eval()
    selected = record_ids_from_file(args.record_ids_file)
    records = load_records_noncausal(args.manifest_dir, args.target_split, args.target_fs, args.lookback_ms,
                                     args.delay_ms, dataset=args.dataset, lead_selector=args.lead_selector,
                                     record_ids=selected, variant=args.variant)
    chunk = 1 << 30 if args.variant == "nc2" else args.chunk
    truth, predicted, logits_all, ids, offsets = [], [], [], [], [0]
    with torch.inference_mode():
        for record in records:
            z = base(torch.from_numpy(record.wave).to(device), torch.from_numpy(record.rr).to(device))
            parts = []; state = None
            for start in range(0, len(record.labels), chunk):
                stop = start + chunk
                part, state = model.forward_chunk(z[start:stop], torch.from_numpy(record.rr[start:stop]).to(device), state)
                parts.append(part.cpu().numpy())
                if state is not None:
                    state = state.detach()
            out = np.concatenate(parts)
            keep = record.score
            truth.append(record.labels[keep]); predicted.append(out.argmax(1)[keep]); logits_all.append(out[keep])
            ids.append(str(record.record_id)); offsets.append(offsets[-1] + int(keep.sum()))
    y = np.concatenate(truth); p = np.concatenate(predicted); z = np.concatenate(logits_all)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "test_predictions.npz", logits=z, labels=y, predictions=p,
                        record_ids=np.asarray(ids), record_offsets=np.asarray(offsets, dtype=np.int64))
    payload = {"metrics": metric(y, p), "n_target_records": len(records), "target_dataset": args.dataset,
               "target_split": args.target_split, "lead_selector": args.lead_selector,
               "record_ids": ids, "variant": args.variant, "protocol": ckpt["protocol"],
               "record_metrics": {rid: metric(y[offsets[i]:offsets[i + 1]], p[offsets[i]:offsets[i + 1]])
                                  for i, rid in enumerate(ids)}}
    (out_dir / "test_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))


# --------------------------------------------------------------------------- #
# causal audit of the control, at the data-construction level
# --------------------------------------------------------------------------- #
def raw_record(manifest_dir: Path, dataset: str, split: str, delay_ms: int, lead_selector: str) -> dict:
    import csv
    with (manifest_dir / f"{dataset}.csv").open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if split == "all" or row["split"] == split]
    record_id = rows[0]["record_id"]
    rows = [row for row in rows if row["record_id"] == record_id]
    peaks = np.asarray([int(row["r_peak_sample"]) for row in rows], dtype=np.int64)
    wf_record = wfdb.rdrecord(str(Path(rows[0]["header_path"]).with_suffix("")))
    signal = _preferred_lead(wf_record, lead_selector)
    source_fs = int(round(float(wf_record.fs)))
    valid = peaks + round(delay_ms * source_fs / 1000) < len(signal)
    return {"record_id": record_id, "signal": signal, "peaks": peaks[valid], "source_fs": source_fs}


def features(variant: str, sample: dict, peaks: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    wave = cut(variant, sample["signal"], peaks, sample["source_fs"], args.target_fs, args.lookback_ms, args.delay_ms)
    return wave, describe_rr(variant, peaks, sample["source_fs"])


def decoder_outputs(backbone: BeatCNN, decoder: nn.Module, wave: np.ndarray, rr: np.ndarray,
                    device: torch.device, stop: int | None = None) -> np.ndarray:
    """CNN logits followed by the history decoder, optionally scoped to a prefix."""
    limit = len(rr) if stop is None else int(stop)
    with torch.inference_mode():
        logits = backbone(torch.from_numpy(wave[:limit]).to(device), torch.from_numpy(rr[:limit]).to(device))
        predicted, _ = decoder.forward_chunk(logits, torch.from_numpy(rr[:limit]).to(device), None)
    return predicted.cpu().numpy()


def run_audit(args: argparse.Namespace) -> None:
    device = device_from(args.device)
    backbone_ckpt = torch.load(args.backbone, map_location="cpu", weights_only=False)
    base = BeatCNN(int(backbone_ckpt.get("channels", 3)), TOTAL_RR_DIM).to(device)
    base.load_state_dict(backbone_ckpt["model"]); base.eval()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = ckpt["args"]
    decoder = build_decoder(args.variant, int(saved["hidden"]), float(saved["max_correction"]), device)
    decoder.load_state_dict(ckpt["model"]); decoder.eval()

    sample = raw_record(Path(args.manifest_dir), args.dataset, args.split, args.delay_ms, args.lead_selector)
    peaks = sample["peaks"]; fs = sample["source_fs"]
    usable = max(2, min(400, len(peaks) - 5))
    wave, rr = features(args.variant, sample, peaks, args)
    reference = decoder_outputs(base, decoder, wave, rr, device)
    tolerance = 1e-4

    # 1. future waveform: corrupt everything after the decision instant of beat `usable`
    cutoff = int(peaks[usable - 1]) + round(args.delay_ms * fs / 1000)
    altered = sample["signal"].copy()
    altered[cutoff + 1:] = np.random.default_rng(17).normal(size=altered[cutoff + 1:].shape).astype(np.float32)
    changed = dict(sample); changed["signal"] = altered
    changed_wave, changed_rr = features(args.variant, changed, peaks, args)
    waveform_error = float(np.max(np.abs(decoder_outputs(base, decoder, changed_wave, changed_rr, device)[:usable]
                                          - reference[:usable])))

    # 2. future R peak: move the next peak by +40 ms
    shifted = peaks.copy()
    shifted[min(usable, len(peaks) - 1)] += int(round(0.040 * fs))
    shifted_wave, shifted_rr = features(args.variant, sample, shifted, args)
    peak_error = float(np.max(np.abs(decoder_outputs(base, decoder, shifted_wave, shifted_rr, device)[:usable]
                                     - reference[:usable])))

    # 3. prefix replay: rebuild every feature from the prefix alone
    prefix = peaks[:usable]
    prefix_wave, prefix_rr = features(args.variant, sample, prefix, args)
    prefix_error = float(np.max(np.abs(decoder_outputs(base, decoder, prefix_wave, prefix_rr, device)[:usable]
                                       - reference[:usable])))

    # 4. record scope of the decoder only: same feature matrix, run on the prefix
    scope_error = float(np.max(np.abs(decoder_outputs(base, decoder, wave, rr, device, stop=usable)[:usable]
                                      - reference[:usable])))

    payload = {
        "variant": args.variant,
        "record_id": sample["record_id"],
        "manipulated_beats": usable,
        "tolerance": tolerance,
        "future_waveform_prefix_max_abs_error": waveform_error,
        "future_rpeak_max_abs_error": peak_error,
        "prefix_replay_max_abs_error": prefix_error,
        "record_scope_max_abs_error": scope_error,
        "fails_future_waveform_test": waveform_error >= tolerance,
        "fails_future_rpeak_test": peak_error >= tolerance,
        "fails_prefix_replay_test": prefix_error >= tolerance,
        "fails_record_scope_test": scope_error >= tolerance,
        "status": "FAIL" if max(waveform_error, peak_error, prefix_error, scope_error) >= tolerance else "PASS",
        "expected": "FAIL: the control is deliberately non-causal",
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--variant", choices=VARIANTS, required=True)
        p.add_argument("--manifest-dir", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--device", default="cuda")
        p.add_argument("--target-fs", type=int, default=250)
        p.add_argument("--lookback-ms", type=int, default=350)
        p.add_argument("--delay-ms", type=int, default=200)
        p.add_argument("--record-ids-file", default="")
        p.add_argument("--lead-selector", default="canonical_first")

    cnn = sub.add_parser("train-cnn"); common(cnn)
    cnn.add_argument("--epochs", type=int, default=15); cnn.add_argument("--batch-size", type=int, default=512)
    cnn.add_argument("--lr", type=float, default=5e-4); cnn.add_argument("--weight-decay", type=float, default=1e-4)
    cnn.add_argument("--power", type=float, default=.6)

    gru = sub.add_parser("train-gru"); common(gru)
    gru.add_argument("--backbone", required=True); gru.add_argument("--epochs", type=int, default=15)
    gru.add_argument("--lr", type=float, default=3e-4); gru.add_argument("--weight-decay", type=float, default=1e-4)
    gru.add_argument("--power", type=float, default=.4); gru.add_argument("--hidden", type=int, default=128)
    gru.add_argument("--max-correction", type=float, default=.75); gru.add_argument("--chunk", type=int, default=8192)

    evaluation = sub.add_parser("evaluate"); common(evaluation)
    evaluation.add_argument("--checkpoint", required=True); evaluation.add_argument("--backbone", required=True)
    evaluation.add_argument("--dataset", default="mit_bih_arrhythmia")
    evaluation.add_argument("--target-split", default="test", choices=("train", "val", "test", "all"))
    evaluation.add_argument("--chunk", type=int, default=8192)

    audit = sub.add_parser("audit"); common(audit)
    audit.add_argument("--checkpoint", required=True); audit.add_argument("--backbone", required=True)
    audit.add_argument("--dataset", default="mit_bih_arrhythmia")
    audit.add_argument("--split", default="test")

    args = parser.parse_args()
    if args.command == "train-cnn":
        run_cnn(args)
    elif args.command == "train-gru":
        run_gru(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    else:
        run_audit(args)


if __name__ == "__main__":
    main()
