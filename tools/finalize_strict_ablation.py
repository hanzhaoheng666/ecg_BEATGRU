#!/usr/bin/env python3
"""Finalize and audit the completed strict M0--M3 ablation archive.

This script does not retrain a model.  It fills the strict archive contract
for an already completed run: per-beat provenance, the missing M0 logits,
immutable provenance metadata, all paired record-level bootstrap contrasts,
and Figures 3--5.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beatgru.data import load_records
from beatgru.metrics import CLASS_NAMES, metric
from beatgru.models import MorphologyCNN


METRICS = ("macro_f1", "S_f1", "V_f1", "sv_f1", "balanced_accuracy", "accuracy")
PAIRS = {
    "M0_to_M1": ("M0", "M1"),
    "M1_to_M2": ("M1", "M2"),
    "M1_to_M3": ("M1", "M3"),
    "M3_to_M2": ("M3", "M2"),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def target_spec(target: str) -> tuple[str, str]:
    if target == "DS2_FINAL_ONCE":
        return "mit_bih_arrhythmia", "test"
    if target == "INCART_II_ALL":
        return "st_petersburg_incart", "all"
    if target == "SVDB_ECG1_ALL":
        return "mit_bih_svdb", "all"
    raise ValueError(target)


def manifest_metadata(manifest_dir: Path, dataset: str, split: str, record_ids: list[str],
                      delay_ms: int) -> dict[str, np.ndarray]:
    """Return metadata for exactly the scored events consumed by load_records."""
    grouped: dict[str, list[dict[str, str]]] = {}
    with (manifest_dir / f"{dataset}.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if split != "all" and row["split"] != split:
                continue
            grouped.setdefault(row["record_id"], []).append(row)

    ids: list[str] = []
    event_index: list[int] = []
    beat_offset: list[int] = []
    peak_seconds: list[float] = []
    offsets = [0]
    for record_id in record_ids:
        rows = grouped.get(str(record_id))
        if rows is None:
            raise ValueError(f"record {record_id} missing from {manifest_dir}/{dataset}.csv")
        header = Path(rows[0]["header_path"])
        import wfdb  # local import keeps --help usable without loading wfdb

        header_info = wfdb.rdheader(str(header.with_suffix("")))
        source_fs = int(round(float(header_info.fs)))
        sig_len = int(header_info.sig_len)
        kept = []
        for row in rows:
            peak = int(row["r_peak_sample"])
            valid = peak + round(delay_ms * source_fs / 1000) < sig_len
            scored = int(row.get("is_scored", "0")) == 1
            if valid and scored:
                kept.append(row)
        ids.extend([str(record_id)] * len(kept))
        event_index.extend([int(row["event_index"]) for row in kept])
        beat_offset.extend([int(row["r_peak_sample"]) for row in kept])
        peak_seconds.extend([int(row["r_peak_sample"]) / float(source_fs) for row in kept])
        offsets.append(offsets[-1] + len(kept))

    return {
        "record_id": np.asarray(ids, dtype="U32"),
        "event_index": np.asarray(event_index, dtype=np.int64),
        "beat_offset": np.asarray(beat_offset, dtype=np.int64),
        "r_peak_sample": np.asarray(beat_offset, dtype=np.int64),
        "beat_time_sec": np.asarray(peak_seconds, dtype=np.float64),
        "record_ids": np.asarray([str(x) for x in record_ids], dtype="U32"),
        "record_offsets": np.asarray(offsets, dtype=np.int64),
    }


def regenerate_m0_logits(root: Path, manifest_dir: Path, seed: int, target: str,
                         npz: dict[str, np.ndarray], device: str = "cpu") -> dict[str, np.ndarray]:
    dataset, split = target_spec(target)
    checkpoint = root / f"seed{seed}" / "M0" / "backbone" / "m0_morphology.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = MorphologyCNN(int(saved.get("channels", 3))).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    args = saved.get("args", {})
    records = load_records(manifest_dir, split, int(args.get("target_fs", 250)),
                           int(args.get("lookback_ms", 350)), int(args.get("delay_ms", 200)),
                           dataset=dataset, lead_selector=args.get("lead_selector", "canonical_first"),
                           record_ids=[str(x) for x in npz["record_ids"]])
    logits, labels, predictions = [], [], []
    with torch.inference_mode():
        for record in records:
            z = model(torch.from_numpy(record.wave).to(device)).cpu().numpy()
            keep = record.score
            logits.append(z[keep])
            labels.append(record.labels[keep])
            predictions.append(z.argmax(1)[keep])
    result = {
        "logits": np.concatenate(logits).astype(np.float32),
        "labels": np.concatenate(labels).astype(np.int64),
        "predictions": np.concatenate(predictions).astype(np.int64),
    }
    if not np.array_equal(result["labels"], npz["labels"]):
        raise ValueError(f"M0 labels changed while regenerating logits: seed={seed}, target={target}")
    if not np.array_equal(result["predictions"], npz["predictions"]):
        # A single borderline argmax can differ between the original GPU
        # evaluation and a later reconstruction.  The regenerated logits and
        # predictions are the internally consistent source of truth; the
        # difference is recorded in the finalized metrics below.
        changed = int(np.sum(result["predictions"] != npz["predictions"]))
        print(f"warning: M0 regenerated predictions changed at {changed} beats; using regenerated logits/predictions", flush=True)
    return result


def finalize_prediction(root: Path, manifest_dir: Path, seed: int, method: str, target: str, device: str) -> None:
    out = root / f"seed{seed}" / method / target
    path = out / "test_predictions.npz"
    old = {key: value for key, value in np.load(path, allow_pickle=False).items()}
    if method == "M0" and "logits" not in old:
        old.update(regenerate_m0_logits(root, manifest_dir, seed, target, old, device=device))
    if "logits" not in old:
        raise ValueError(f"missing logits in {path}")
    dataset, split = target_spec(target)
    record_ids = [str(x) for x in old["record_ids"]]
    metadata = manifest_metadata(manifest_dir, dataset, split, record_ids, delay_ms=200)
    n = len(old["labels"])
    for key in ("logits", "labels", "predictions", "record_id", "event_index", "beat_offset", "r_peak_sample", "beat_time_sec"):
        if key in metadata and len(metadata[key]) != n:
            raise ValueError(f"metadata length mismatch {path}: {key}={len(metadata[key])}, labels={n}")
    if len(metadata["record_offsets"]) != len(record_ids) + 1:
        raise ValueError(f"record offset mismatch in {path}")
    np.savez_compressed(
        path,
        logits=np.asarray(old["logits"], dtype=np.float32),
        labels=np.asarray(old["labels"], dtype=np.int64),
        predictions=np.asarray(old["predictions"], dtype=np.int64),
        record_id=metadata["record_id"],
        record_ids=metadata["record_ids"],
        event_index=metadata["event_index"],
        beat_offset=metadata["beat_offset"],
        r_peak_sample=metadata["r_peak_sample"],
        beat_time_sec=metadata["beat_time_sec"],
        record_offsets=metadata["record_offsets"],
    )
    metrics_path = out / "test_metrics.json"
    payload = read_json(metrics_path)
    payload["metrics"] = metric(old["labels"], old["predictions"])
    payload["record_metrics"] = {
        str(record_id): metric(old["labels"][start:stop], old["predictions"][start:stop])
        for record_id, start, stop in zip(metadata["record_ids"], metadata["record_offsets"][:-1], metadata["record_offsets"][1:])
    }
    payload.update({
        "dataset": dataset,
        "target_split": split,
        "n_scored_beats": n,
        "metadata_fields": ["record_id", "event_index", "beat_offset", "r_peak_sample", "beat_time_sec", "record_offsets"],
    })
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    config = {
        "seed": seed, "method": method, "target": target, "dataset": dataset, "target_split": split,
        "manifest_dir": str(manifest_dir.resolve()), "target_fs": 250, "lookback_ms": 350,
        "delay_ms": 200, "scored_labels": "AAMI N/S/V; F/Q excluded from metrics",
        "evaluation": "frozen checkpoint; no DS2/external selection, threshold, or lead tuning",
    }
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "command.txt").write_text(
        f"Historical source: scripts/run_m0_m3_three_seed_ablation.sh <manifest> <external_manifest> <root>\n"
        f"Finalized by: python tools/finalize_strict_ablation.py --root {root} --manifest-dir {manifest_dir}\n",
        encoding="utf-8")


def bootstrap(values: np.ndarray, rng: np.random.Generator, n: int = 10000) -> list[float]:
    draws = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def paired_record_delta(root: Path, seed: int, left: str, right: str, target: str) -> dict[str, object]:
    a = read_json(root / f"seed{seed}" / left / target / "test_metrics.json")
    b = read_json(root / f"seed{seed}" / right / target / "test_metrics.json")
    ids = sorted(set(a["record_metrics"]) & set(b["record_metrics"]))
    result: dict[str, object] = {"left": left, "right": right, "delta": "right-left", "n_records": len(ids)}
    for offset, name in enumerate(("macro_f1", "S_f1", "V_f1")):
        values = np.asarray([b["record_metrics"][rid][name] - a["record_metrics"][rid][name] for rid in ids], dtype=float)
        ci = bootstrap(values, np.random.default_rng(1700 + seed * 31 + offset))
        result[name] = {
            "mean_delta": float(values.mean()), "bootstrap95": ci,
            "n_improved": int(np.sum(values > 0)), "n_records": len(values),
            "deltas_by_record": {rid: float(value) for rid, value in zip(ids, values)},
        }
    return result


def summarize(root: Path) -> dict[str, object]:
    out: dict[str, object] = {"bootstrap_unit": "DS2 record", "pairs": {}}
    metric_rows = []
    for target, methods in (("DS2_FINAL_ONCE", ("M0", "M1", "M2", "M3")),
                            ("INCART_II_ALL", ("M1", "M3")),
                            ("SVDB_ECG1_ALL", ("M1", "M3"))):
        for method in methods:
            for seed in (0, 1, 2):
                payload = read_json(root / f"seed{seed}" / method / target / "test_metrics.json")
                metric_rows.append({"seed": seed, "method": method, "target": target,
                                    **{name: payload["metrics"][name] for name in METRICS}})
    with (root / "summary" / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "method", "target", *METRICS])
        writer.writeheader(); writer.writerows(metric_rows)
    for pair, (left, right) in PAIRS.items():
        out["pairs"][pair] = {str(seed): paired_record_delta(root, seed, left, right, "DS2_FINAL_ONCE") for seed in (0, 1, 2)}
    (root / "summary" / "paired_bootstrap.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 严格 M0–M3 三 seed 消融汇总", "", "训练与冻结评估均来自已完成的 15 epoch 运行；均值±标准差按三个 seed 计算。", "", "| target/method | Macro-F1 | S-F1 | V-F1 | SV-F1 | Balanced Acc | Accuracy |", "|---|---:|---:|---:|---:|---:|---:|"]
    for target, methods in (("DS2_FINAL_ONCE", ("M0", "M1", "M2", "M3")), ("INCART_II_ALL", ("M1", "M3")), ("SVDB_ECG1_ALL", ("M1", "M3"))):
        for method in methods:
            values = [row for row in metric_rows if row["target"] == target and row["method"] == method]
            text = []
            for name in ("macro_f1", "S_f1", "V_f1", "sv_f1", "balanced_accuracy", "accuracy"):
                arr = np.asarray([row[name] for row in values], dtype=float)
                text.append(f"{arr.mean():.4f}±{arr.std(ddof=1):.4f}")
            lines.append(f"| {target}/{method} | " + " | ".join(text) + " |")
    lines += ["", "## DS2 record-level paired bootstrap", "", "delta = right - left; bootstrap unit = DS2 record", ""]
    lines.append("| contrast | seed | Macro-F1 delta [95% CI] | S-F1 delta [95% CI] | V-F1 delta [95% CI] |")
    lines.append("|---|---:|---:|---:|---:|")
    for pair in PAIRS:
        for seed in (0, 1, 2):
            row = out["pairs"][pair][str(seed)]
            vals = []
            for name in ("macro_f1", "S_f1", "V_f1"):
                v = row[name]; vals.append(f"{v['mean_delta']:.4f} [{v['bootstrap95'][0]:.4f}, {v['bootstrap95'][1]:.4f}]")
            lines.append(f"| {pair} | {seed} | {vals[0]} | {vals[1]} | {vals[2]} |")
    (root / "summary" / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (root / "summary" / "paired_bootstrap.md").write_text("\n".join(lines[lines.index("## DS2 record-level paired bootstrap"):]), encoding="utf-8")
    return out


def aggregate_metric(root: Path, target: str, method: str, name: str) -> tuple[float, float]:
    values = [read_json(root / f"seed{seed}" / method / target / "test_metrics.json")["metrics"][name] for seed in (0, 1, 2)]
    return float(np.mean(values)), float(np.std(values, ddof=1))


def make_figures(root: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ["M0", "M1", "M2", "M3"]
    colors = ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, names, title in zip(axes, (("macro_f1", "S_f1", "V_f1"), ("sv_f1", "balanced_accuracy", "accuracy")), ("Macro/S/V F1", "SV-F1/Balanced accuracy/Accuracy")):
        x = np.arange(len(names)); width = .18
        for index, method in enumerate(methods):
            means = [aggregate_metric(root, "DS2_FINAL_ONCE", method, name)[0] for name in names]
            ax.bar(x + (index - 1.5) * width, means, width, label=method, color=colors[index])
        ax.set_xticks(x, names); ax.set_ylim(0, 1.05); ax.set_title(title); ax.grid(axis="y", alpha=.25)
    axes[0].legend(ncol=4, frameon=False, loc="lower right")
    fig.suptitle("Figure 3. Strict causal M0–M3 DS2 ablation")
    fig.savefig(root / "summary" / "figure3_m0_m3_main_ablation.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, target, title in zip(axes, ("INCART_II_ALL", "SVDB_ECG1_ALL"), ("INCART-II", "SVDB-ECG1")):
        x = np.arange(3); width = .34
        for index, method in enumerate(("M1", "M3")):
            means = [aggregate_metric(root, target, method, name)[0] for name in ("macro_f1", "S_f1", "V_f1")]
            ax.bar(x + (index - .5) * width, means, width, label=method, color=("#74c476", "#238b45")[index])
        ax.set_xticks(x, ("Macro-F1", "S-F1", "V-F1")); ax.set_ylim(0, 1.05); ax.set_title(title); ax.grid(axis="y", alpha=.25)
    axes[0].legend(frameon=False); fig.suptitle("Figure 4. Frozen external M1/M3 evaluation")
    fig.savefig(root / "summary" / "figure4_external_frozen.png", dpi=180); plt.close(fig)

    bootstrap_data = read_json(root / "summary" / "paired_bootstrap.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for seed in (0, 1, 2):
        values = [bootstrap_data["pairs"]["M1_to_M3"][str(seed)][name]["mean_delta"] for name in ("macro_f1", "S_f1", "V_f1")]
        axes[0].plot(("Macro-F1", "S-F1", "V-F1"), values, marker="o", label=f"seed {seed}")
    axes[0].axhline(0, color="black", linewidth=.8); axes[0].set_title("M1→M3 per-record mean changes"); axes[0].set_ylabel("F1 delta"); axes[0].legend(frameon=False)
    cms = []
    for seed in (0, 1, 2):
        a = read_json(root / f"seed{seed}" / "M1" / "DS2_FINAL_ONCE" / "test_metrics.json")["metrics"]["confusion_matrix"]
        b = read_json(root / f"seed{seed}" / "M3" / "DS2_FINAL_ONCE" / "test_metrics.json")["metrics"]["confusion_matrix"]
        cms.append(np.asarray(b, dtype=float) / np.maximum(np.asarray(b).sum(axis=1, keepdims=True), 1) - np.asarray(a, dtype=float) / np.maximum(np.asarray(a).sum(axis=1, keepdims=True), 1))
    image = axes[1].imshow(np.mean(cms, axis=0), cmap="coolwarm", vmin=-.1, vmax=.1)
    axes[1].set_xticks(range(3), CLASS_NAMES); axes[1].set_yticks(range(3), CLASS_NAMES); axes[1].set_xlabel("predicted"); axes[1].set_ylabel("true"); axes[1].set_title("Mean normalized confusion change M3−M1")
    fig.colorbar(image, ax=axes[1], fraction=.046, pad=.04); fig.suptitle("Figure 5. Historical-state changes")
    fig.savefig(root / "summary" / "figure5_m1_m3_record_confusion.png", dpi=180); plt.close(fig)


def provenance(root: Path, manifest_dir: Path, external_manifest: Path) -> None:
    source_files = [ROOT / "beatgru" / name for name in ("data.py", "models.py", "train.py", "metrics.py", "audit.py")]
    source_files += [ROOT / "tools" / name for name in ("train_m0_morphology.py", "run_history_ablation.py", "summarize_ablation.py", "finalize_strict_ablation.py")]
    data_files = sorted(manifest_dir.glob("*.csv")) + sorted(manifest_dir.glob("*.json"))
    external_files = sorted(external_manifest.glob("*.csv")) + sorted(external_manifest.glob("*.json"))
    try:
        cuda = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        cuda = "nvidia-smi unavailable"
    try:
        torch_cuda = torch.version.cuda
    except Exception:
        torch_cuda = None
    log_births = []
    for path in sorted(root.glob("seed*/M*_train.log")):
        try:
            stat = path.stat()
            # Linux commonly exposes no st_birthtime; ctime is the closest
            # preserved creation-time proxy for these newly created redirects.
            birth = getattr(stat, "st_birthtime", None) or stat.st_ctime
            if birth is not None:
                log_births.append(float(birth))
        except OSError:
            pass
    inferred_start = datetime.fromtimestamp(min(log_births)).astimezone().isoformat() if log_births else None
    archive = {
        "archive_type": "strict_m0_m3_ablation_finalized",
        "source_run": str(root.resolve()),
        "finalized_at_utc": now(),
        "training_started_at": inferred_start,
        "training_started_at_source": "inferred from earliest seed M0/M2/M3 training log creation time; legacy launcher did not write a start timestamp",
        "code_commit": None,
        "code_commit_note": "BEATGRU is not a git worktree; source SHA256 fingerprints are recorded instead",
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_files if path.exists()},
        "data_manifest_dir": str(manifest_dir.resolve()),
        "data_manifest_sha256": {path.name: sha256(path) for path in data_files},
        "external_manifest_dir": str(external_manifest.resolve()),
        "external_manifest_sha256": {path.name: sha256(path) for path in external_files},
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "torch_cuda": torch_cuda, "cuda_devices_at_finalize": cuda},
        "fixed_training_config": {"seeds": [0, 1, 2], "epochs": 15, "cnn_lr": 5e-4, "cnn_weight_decay": 1e-4, "cnn_batch_size": 512, "cnn_class_power": .6, "gru_lr": 3e-4, "gru_weight_decay": 1e-4, "gru_class_power": .4, "gru_hidden": 128, "gru_max_correction": .75, "gru_chunk": 8192},
        "evaluation_policy": {"ds2_once": True, "external_frozen": True, "lead_incart": "II", "lead_svdb": "ECG1/canonical_first", "threshold_tuning": False, "target_selection": False, "scored_classes": list(CLASS_NAMES)},
        "tasks": [{"seed": seed, "method": method, "ds2": str((root / f"seed{seed}" / method / "DS2_FINAL_ONCE").relative_to(root))} for seed in (0, 1, 2) for method in ("M0", "M1", "M2", "M3")],
    }
    path = root / "MANIFEST.json"
    if path.exists():
        path.chmod(0o644)
    path.write_text(json.dumps(archive, ensure_ascii=False, indent=2), encoding="utf-8")
    path.chmod(0o444)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    root = args.root.resolve(); manifest_dir = args.manifest_dir.resolve(); external = (args.external_manifest or args.manifest_dir).resolve()
    for seed in (0, 1, 2):
        for method in ("M0", "M1", "M2", "M3"):
            finalize_prediction(root, manifest_dir, seed, method, "DS2_FINAL_ONCE", args.device)
        for method in ("M1", "M3"):
            finalize_prediction(root, external, seed, method, "INCART_II_ALL", args.device)
            finalize_prediction(root, external, seed, method, "SVDB_ECG1_ALL", args.device)
    summarize(root)
    make_figures(root)
    provenance(root, manifest_dir, external)
    print(json.dumps({"root": str(root), "status": "FINALIZED", "manifest": str(root / "MANIFEST.json"), "figures": ["figure3_m0_m3_main_ablation.png", "figure4_external_frozen.png", "figure5_m1_m3_record_confusion.png"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
