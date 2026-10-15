#!/usr/bin/env python3
"""Add record provenance to a per-beat prediction file without touching it.

Early locked runs stored only ``labels/logits/predictions``.  The record-cluster
bootstrap needs to know which record every beat belongs to.  This tool
reconstructs the manifest-derived record sequence, verifies that it reproduces
the stored beat count exactly, and writes a *separate* file carrying
``record_ids``/``record_offsets``.  The original artefact is never modified.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def record_layout(manifest_dir: Path, dataset: str, split: str, delay_ms: int) -> tuple[list[str], list[int]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with (manifest_dir / f"{dataset}.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if split != "all" and row["split"] != split:
                continue
            grouped.setdefault(row["record_id"], []).append(row)
    ids: list[str] = []
    offsets = [0]
    for record_id, rows in sorted(grouped.items(), key=lambda item: item[0]):
        header = Path(rows[0]["header_path"])
        info = wfdb.rdheader(str(header.with_suffix("")))
        source_fs = int(round(float(info.fs)))
        sig_len = int(info.sig_len)
        kept = 0
        for row in rows:
            peak = int(row["r_peak_sample"])
            if peak + round(delay_ms * source_fs / 1000) >= sig_len:
                continue
            scored = (int(row["is_scored"]) == 1 if row.get("is_scored", "") != ""
                      else {"N": 0, "S": 1, "V": 2}.get(row.get("aami_label", ""), -100) >= 0)
            if scored:
                kept += 1
        if kept:
            ids.append(record_id)
            offsets.append(offsets[-1] + kept)
    return ids, offsets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True, choices=("train", "val", "test", "all"))
    parser.add_argument("--delay-ms", type=int, default=200)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source = np.load(args.npz, allow_pickle=False)
    ids, offsets = record_layout(Path(args.manifest_dir), args.dataset, args.split, args.delay_ms)
    n_beats = len(source["labels"])
    if offsets[-1] != n_beats:
        raise ValueError(f"reconstructed layout has {offsets[-1]} beats but the file stores {n_beats}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, labels=source["labels"], predictions=source["predictions"],
                        logits=source["logits"],
                        record_ids=np.asarray(ids), record_offsets=np.asarray(offsets, dtype=np.int64))
    print(json.dumps({"source": str(Path(args.npz).resolve()), "written": str(out.resolve()),
                      "n_records": len(ids), "n_beats": int(n_beats)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
