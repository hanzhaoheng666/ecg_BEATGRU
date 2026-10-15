#!/usr/bin/env python3
"""Per-lead frozen sensitivity of the locked model on INCART.

The pre-declared lead rule for the locked result is: the source domain is
MIT-BIH MLII, therefore an external database that names a lead II is read from
that lead.  This tool does **not** change that rule and it does **not** select a
lead.  It re-runs the frozen checkpoint once per available lead and reports the
whole grid, so the reader can see how much of the INCART result is the lead
choice and how much is the model.  Selecting the best row here would be target
domain tuning and is explicitly not done; the declared rule stays II.

No training, no threshold tuning, no re-selection.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from beatgru.data import causal_windows, derivative_channels, direct_causal_rr
from beatgru.metrics import metric
from beatgru.models import BeatCNN, LogitGRU
from rpeak_jitter_sensitivity import cut_windows

DECLARED_RULE = "source_matched_ii (MIT-BIH MLII -> INCART lead II), fixed before any target label was read"


def load_grouped(manifest_dir: Path, dataset: str, split: str, delay_ms: int) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with (manifest_dir / f"{dataset}.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if split != "all" and row["split"] != split:
                continue
            grouped.setdefault(row["record_id"], []).append(row)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--dataset", default="st_petersburg_incart")
    parser.add_argument("--split", default="all")
    parser.add_argument("--backbone", required=True, help="locked M1/M3 CNN checkpoint")
    parser.add_argument("--history", required=True, help="locked M3 history decoder checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-fs", type=int, default=250)
    parser.add_argument("--lookback-ms", type=int, default=350)
    parser.add_argument("--delay-ms", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-records", type=int, default=0, help="debug subset; 0 = every record")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    backbone_ckpt = torch.load(args.backbone, map_location="cpu", weights_only=False)
    base = BeatCNN(int(backbone_ckpt.get("channels", 3)), 8).to(device)
    base.load_state_dict(backbone_ckpt["model"]); base.eval()
    history_ckpt = torch.load(args.history, map_location="cpu", weights_only=False)
    saved = history_ckpt["args"]
    history = LogitGRU(8, int(saved["hidden"]), float(saved["max_correction"])).to(device)
    history.load_state_dict(history_ckpt["model"]); history.eval()

    grouped = load_grouped(Path(args.manifest_dir), args.dataset, args.split, args.delay_ms)
    record_ids = sorted(grouped, key=lambda value: value)
    if args.max_records:
        record_ids = record_ids[: args.max_records]

    lead_names: list[str] | None = None
    caches: list[dict[str, object]] = []
    for record_id in record_ids:
        rows = grouped[record_id]
        peaks = np.asarray([int(row["r_peak_sample"]) for row in rows], dtype=np.int64)
        labels = np.asarray([int(row["label"]) if row.get("label", "") != "" else
                             {"N": 0, "S": 1, "V": 2}.get(row.get("aami_label", ""), -100)
                             for row in rows], dtype=np.int64)
        scored = np.asarray([int(row["is_scored"]) == 1 if row.get("is_scored", "") != "" else label >= 0
                             for row, label in zip(rows, labels)], dtype=bool)
        record = wfdb.rdrecord(str(Path(rows[0]["header_path"]).with_suffix("")))
        signal = record.p_signal if record.p_signal is not None else record.d_signal
        signal = np.asarray(signal, dtype=np.float32)
        names = [str(x) for x in record.sig_name]
        source_fs = int(round(float(record.fs)))
        valid = peaks + round(args.delay_ms * source_fs / 1000) < signal.shape[0]
        peaks, labels, scored = peaks[valid], labels[valid], scored[valid]
        if lead_names is None:
            lead_names = names
        elif names != lead_names:
            raise ValueError(f"{record_id}: lead order differs from the first record")
        caches.append({"record_id": record_id, "signal": signal, "peaks": peaks, "labels": labels,
                       "score": scored, "source_fs": source_fs})

    # The vectorised cutter must reproduce the locked contract exactly; the
    # strict contract cannot be used directly because it raises whenever a
    # window would pass the end of a record, which happens for a few INCART
    # records at 257 Hz.
    probe = caches[0]
    reference = derivative_channels(causal_windows(probe["signal"][:, :1], probe["peaks"][:-1], probe["source_fs"],
                                                   args.target_fs, args.lookback_ms, args.delay_ms))
    candidate = derivative_channels(cut_windows(probe["signal"][:, :1], probe["peaks"][:-1], probe["source_fs"],
                                                args.target_fs, args.lookback_ms, args.delay_ms))
    if not np.array_equal(reference, candidate):
        raise ValueError("vectorised cutter diverged from the locked contract on INCART")

    report = {"dataset": args.dataset, "split": args.split, "n_records": len(caches),
              "declared_lead_rule": DECLARED_RULE,
              "sensitivity_only": "the whole grid is reported; no lead was selected on target data",
              "leads": {}}
    with torch.inference_mode():
        for lead_index, lead in enumerate(lead_names or []):
            truth, predicted_m1, predicted_m3 = [], [], []
            for item in caches:
                wave = derivative_channels(cut_windows(item["signal"][:, lead_index:lead_index + 1], item["peaks"],
                                                       item["source_fs"], args.target_fs,
                                                       args.lookback_ms, args.delay_ms))
                rr = direct_causal_rr(item["peaks"], item["source_fs"])
                logits = base(torch.from_numpy(wave).to(device), torch.from_numpy(rr).to(device))
                corrected, _ = history.forward_chunk(logits, torch.from_numpy(rr).to(device), None)
                keep = item["score"]
                truth.append(item["labels"][keep])
                predicted_m1.append(logits.argmax(1).cpu().numpy()[keep])
                predicted_m3.append(corrected.argmax(1).cpu().numpy()[keep])
            y = np.concatenate(truth)
            report["leads"][lead] = {
                "lead_index": lead_index,
                "M1": {key: value for key, value in metric(y, np.concatenate(predicted_m1)).items() if key != "confusion_matrix"},
                "M3": {key: value for key, value in metric(y, np.concatenate(predicted_m3)).items() if key != "confusion_matrix"},
            }
            print(json.dumps({"lead": lead,
                              "M1_macro_f1": round(report["leads"][lead]["M1"]["macro_f1"], 6),
                              "M3_macro_f1": round(report["leads"][lead]["M3"]["macro_f1"], 6)}), flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "incart_per_lead.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# INCART 逐导联冻结敏感性（零训练，M1 与 M3）", "",
             f"- 预先固定的导联规则：`{DECLARED_RULE}`",
             "- 下表是全部导联的完整网格，仅作敏感性；**没有**依据目标域指标挑选导联",
             f"- 记录数 {len(caches)}，计分心拍 {int(sum(item['score'].sum() for item in caches))}", "",
             "| 导联 | M1 Macro-F1 | M1 S-F1 | M1 V-F1 | M3 Macro-F1 | M3 S-F1 | M3 V-F1 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for lead, payload in report["leads"].items():
        a, b = payload["M1"], payload["M3"]
        marker = "  ← 预声明规则" if lead == "II" else ""
        lines.append(f"| {lead}{marker} | {a['macro_f1']:.6f} | {a['S_f1']:.6f} | {a['V_f1']:.6f} | "
                     f"{b['macro_f1']:.6f} | {b['S_f1']:.6f} | {b['V_f1']:.6f} |")
    (out / "incart_per_lead.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
