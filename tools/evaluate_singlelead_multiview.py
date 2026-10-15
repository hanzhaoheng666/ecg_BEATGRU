#!/usr/bin/env python3
"""Frozen single-lead BEATGRU evaluated through two independent lead views.

This is intentionally an inference-only adapter: the source CNN and GRU have
three waveform channels and are not retrained on a second lead.  Each view has
its own causal GRU state, reset at the same record boundaries; logits are then
averaged at the current decision time.  No target label or target statistic is
used in fusion.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Support both ``python -m ...`` and direct execution from scripts/.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beatgru.data import load_records
from beatgru.metrics import metric
from beatgru.models import BeatCNN, EmbeddingGRU, LogitGRU
from beatgru.train import device_from, emissions, evaluate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backbone", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", default="mit_bih_arrhythmia")
    p.add_argument("--target-split", choices=("train", "val", "test", "all"), default="test")
    p.add_argument("--first-selector", default="source_matched_ii")
    p.add_argument("--second-selector", default="source_matched_precordial")
    p.add_argument("--record-ids-file", default="")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["args"]; device = device_from(args.device)
    selected = None if not args.record_ids_file else [x.strip() for x in Path(args.record_ids_file).read_text().splitlines() if x.strip()]
    backbone = torch.load(args.backbone or ckpt["backbone"], map_location="cpu", weights_only=False)
    if int(backbone.get("channels", 3)) != 3:
        raise ValueError("this tool is for a frozen single-lead (3-channel) checkpoint")
    base = BeatCNN(3, 8).to(device); base.load_state_dict(backbone["model"]); base.eval()
    history_input = cfg.get("history_input", "logits")
    model_type = EmbeddingGRU if history_input == "embedding" else LogitGRU
    def history() -> LogitGRU | EmbeddingGRU:
        model = (EmbeddingGRU(160, 8, cfg["hidden"], cfg["max_correction"])
                 if model_type is EmbeddingGRU else LogitGRU(8, cfg["hidden"], cfg["max_correction"])).to(device)
        model.load_state_dict(ckpt["model"]); return model
    common = dict(manifest_dir=args.manifest_dir, split=args.target_split, target_fs=cfg["target_fs"],
                  lookback_ms=cfg["lookback_ms"], delay_ms=cfg["delay_ms"], dataset=args.dataset, record_ids=selected)
    first = load_records(lead_selector=args.first_selector, **common)
    second = load_records(lead_selector=args.second_selector, **common)
    if [(r.record_id, len(r.labels)) for r in first] != [(r.record_id, len(r.labels)) for r in second]:
        raise ValueError("views do not have identical chronological event sequences")
    if any(not (np.array_equal(a.labels, b.labels) and np.array_equal(a.score, b.score) and np.array_equal(a.rr, b.rr))
           for a, b in zip(first, second)):
        raise ValueError("views disagree on labels, scoring mask, or causal RR")
    rows_a = emissions(base, first, device, include_embedding=history_input == "embedding")
    rows_b = emissions(base, second, device, include_embedding=history_input == "embedding")
    _, logits_a, labels, _ = evaluate(history(), rows_a, device, cfg["chunk"])
    _, logits_b, labels_b, _ = evaluate(history(), rows_b, device, cfg["chunk"])
    if not np.array_equal(labels, labels_b):
        raise ValueError("scored labels differ after view inference")
    logits = (logits_a + logits_b) / 2.0
    predictions = logits.argmax(1); values = metric(labels, predictions)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "test_predictions.npz", logits=logits, labels=labels, predictions=predictions,
                        first_logits=logits_a, second_logits=logits_b)
    payload = {"metrics": values, "n_target_records": len(first), "target_dataset": args.dataset,
               "target_split": args.target_split, "fusion": "arithmetic_mean_of_two_frozen_single-lead_logit_streams",
               "first_selector": args.first_selector, "second_selector": args.second_selector,
               "protocol": {**ckpt["protocol"], "single_lead_source_training": True,
                            "independent_per_view_causal_states": True, "target_label_or_statistic_used_for_fusion": False}}
    (out / "test_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
