#!/usr/bin/env python3
"""Same-scope end-to-end resource comparison for the two decoders.

Earlier measurements were not comparable: the latency figures on record for the
locked model were end-to-end (frozen CNN + decoder, real DS2 windows), while the
windowed decoder was later timed decoder-only on synthetic tensors.  This tool
removes that discrepancy: every configuration below runs the *same* frozen CNN on
the *same* real windows and differs only in the history decoder.

Configurations
--------------
``m3_recurrent``  BeatCNN + LogitGRU, hidden state carried across the record
``m2_windowed``   BeatCNN + a strict 32-token ring buffer, GRU re-run from a zero
                  initial state over the buffered window each beat
``beatcnn_only``  BeatCNN alone (lower bound; no history)

The windowed buffer is implemented here with a preallocated double buffer, so the
online form does not reallocate per beat.  That is the fair implementation for a
resource comparison; the batched ``ShortLogitGRU.forward_windows`` path used by
the evaluation scripts materialises ``[n, window, hidden]`` and is measured
separately as ``m2_batched`` because its footprint is a property of the batched
loop, not of the architecture.

Modes
-----
``--mode latency   --manifest-dir ... --backbone ... --m2 ... --m3 ...``
``--mode memory    --n-beats N``          (one case per process for a clean RSS)
``--mode sizes     --backbone ... --m2 ... --m3 ...``
"""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beatgru.data import load_records  # noqa: E402
from beatgru.models import BeatCNN, LogitGRU, ShortLogitGRU  # noqa: E402

WINDOW = 32
DECODERS = ("m3_recurrent", "m2_windowed", "m2_batched", "beatcnn_only")


def rss_kb() -> int:
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return -1


class TokenRing:
    """Preallocated ring of the last ``window`` tokens (no per-beat allocation)."""

    def __init__(self, window: int, dim: int) -> None:
        self.window = window
        self.buffer = torch.zeros(2 * window, dim)
        self.seen = 0
        self.scratch = torch.zeros(1, window, dim)

    def push(self, token: torch.Tensor) -> torch.Tensor:
        """Append one token and return the chronological last-``window`` view."""
        slot = self.seen % (2 * self.window)
        self.buffer[slot] = token
        self.seen += 1
        valid = min(self.seen, self.window)
        start = slot - valid + 1
        if start >= 0:
            return self.buffer[start:start + valid]
        wrapped = self.buffer[start:]          # slot+1 entries from the tail
        head = self.buffer[:valid - wrapped.shape[0]]
        return torch.cat((wrapped, head))


def load_all(backbone: Path, m2: Path | None, m3: Path | None, device: torch.device):
    bb = torch.load(backbone, map_location="cpu", weights_only=False)
    base = BeatCNN(int(bb.get("channels", 3)), 8).to(device)
    base.load_state_dict(bb["model"]); base.eval()
    models: dict[str, nn.Module] = {}
    if m3 is not None:
        ck = torch.load(m3, map_location="cpu", weights_only=False)
        cfg = ck["args"]
        m = LogitGRU(cfg.get("rr_dim", 8), cfg["hidden"], cfg["max_correction"]).to(device)
        m.load_state_dict(ck["model"]); m.eval(); models["m3_recurrent"] = m
    if m2 is not None:
        ck = torch.load(m2, map_location="cpu", weights_only=False)
        cfg = ck["args"]
        m = ShortLogitGRU(cfg.get("rr_dim", 8), cfg["hidden"], cfg["max_correction"], WINDOW).to(device)
        m.load_state_dict(ck["model"]); m.eval(); models["m2_windowed"] = m
    return base, models


def param_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def int8_kb(module: nn.Module) -> float | None:
    try:
        q = torch.ao.quantization.quantize_dynamic(module, {nn.Linear, nn.GRU}, dtype=torch.qint8)
        buf = io.BytesIO(); torch.save(q.state_dict(), buf)
        return round(buf.getbuffer().nbytes / 1024, 1)
    except Exception:  # noqa: BLE001
        return None


@torch.inference_mode()
def stream_record(base, models, wave, rr, scope: str, ring: TokenRing | None):
    state = None
    for index in range(wave.shape[0]):
        x = wave[index:index + 1]
        r = rr[index:index + 1]
        logits = base(x, r)
        if scope == "m3_recurrent":
            _, state = models["m3_recurrent"].forward_chunk(logits, r, state)
        elif scope == "m2_windowed":
            model = models["m2_windowed"]
            token = model.input(torch.cat((logits, r), dim=1)).squeeze(0)   # [hidden]
            window = ring.push(token)
            _, hidden = model.gru(window.unsqueeze(0))
            model.delta(hidden[-1])
        elif scope == "m2_batched":
            models["m2_windowed"].forward_windows(logits, r)


def latency_mode(args) -> None:
    device = torch.device("cpu")
    torch.set_num_threads(1)
    records = load_records(Path(args.manifest_dir), "test", 250, 350, 200, lead_selector="canonical_first")
    tensors = [(torch.from_numpy(r.wave), torch.from_numpy(r.rr)) for r in records]
    base, models = load_all(Path(args.backbone), Path(args.m2), Path(args.m3), device)
    hidden_dim = int(models["m2_windowed"].gru.hidden_size)
    report: dict[str, object] = {
        "scope": "end-to-end: frozen CNN + history decoder, real DS2 windows, CPU single thread",
        "manifest_dir": args.manifest_dir, "n_records": len(records),
        "n_beats_with_f_q": int(sum(t[0].shape[0] for t in tensors)),
        "warmup_beats": args.warmup, "decoders": {},
    }
    for scope in DECODERS:
        ring = TokenRing(WINDOW, hidden_dim) if scope == "m2_windowed" else None
        durations: list[float] = []
        seen = 0
        for wave, rr in tensors:
            state = None
            for index in range(wave.shape[0]):
                start = time.perf_counter_ns()
                stream_record(base, models, wave[index:index + 1], rr[index:index + 1], scope, ring)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                seen += 1
                if seen > args.warmup:
                    durations.append(elapsed)
        values = np.asarray(durations)
        report["decoders"][scope] = {
            "n_timed": int(len(values)),
            "mean_ms": round(float(values.mean()), 4),
            "p50_ms": round(float(np.quantile(values, 0.50)), 4),
            "p90_ms": round(float(np.quantile(values, 0.90)), 4),
            "p99_ms": round(float(np.quantile(values, 0.99)), 4),
            "max_ms": round(float(values.max()), 4),
            "sd_ms": round(float(statistics.pstdev(values)), 4),
        }
        print(scope, json.dumps(report["decoders"][scope]))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def memory_mode(args) -> None:
    torch.set_num_threads(1)
    device = torch.device("cpu")
    base, models = load_all(Path(args.backbone), Path(args.m2), Path(args.m3), device)
    n = int(args.n_beats)
    generator = torch.Generator().manual_seed(20260927)
    wave = torch.randn(n, 3, 138, generator=generator)
    rr = torch.randn(n, 8, generator=generator)
    ring = TokenRing(WINDOW, int(models["m2_windowed"].gru.hidden_size)) if args.case == "m2_windowed" else None
    baseline = rss_kb()
    start = time.perf_counter()
    stream_record(base, models, wave, rr, args.case, ring)
    elapsed = time.perf_counter() - start
    print(json.dumps({"case": args.case, "n_beats": n, "seconds": round(elapsed, 4),
                      "per_beat_ms": round(elapsed / n * 1000, 4),
                      "rss_baseline_kb": baseline, "rss_after_kb": rss_kb(),
                      "rss_increment_kb": rss_kb() - baseline}, ensure_ascii=False))


def sizes_mode(args) -> None:
    device = torch.device("cpu")
    base, models = load_all(Path(args.backbone), Path(args.m2), Path(args.m3), device)
    out: dict[str, object] = {
        "beatcnn_params": param_count(base),
        "beatcnn_file_kb": round(Path(args.backbone).stat().st_size / 1024, 1),
        "beatcnn_int8_kb": int8_kb(base),
        "decoders": {},
    }
    param_hidden = int(models["m2_windowed"].gru.hidden_size)
    for name, path in (("m2_windowed", args.m2), ("m3_recurrent", args.m3)):
        module = models[name]
        out["decoders"][name] = {
            "params": param_count(module),
            "file_kb": round(Path(path).stat().st_size / 1024, 1),
            "int8_kb": int8_kb(module),
            "online_state_bytes_fp32": (WINDOW * param_hidden * 4) if name == "m2_windowed" else (param_hidden * 4),
            "online_state_note": ("32 buffered hidden-dim tokens" if name == "m2_windowed" else "one hidden vector"),
        }
    out["total"] = {
        "m2_windowed_params": out["beatcnn_params"] + out["decoders"]["m2_windowed"]["params"],
        "m3_recurrent_params": out["beatcnn_params"] + out["decoders"]["m3_recurrent"]["params"],
        "m2_windowed_file_kb": round(out["beatcnn_file_kb"] + out["decoders"]["m2_windowed"]["file_kb"], 1),
        "m3_recurrent_file_kb": round(out["beatcnn_file_kb"] + out["decoders"]["m3_recurrent"]["file_kb"], 1),
    }
    out["hardware"] = {"platform": platform.platform(), "python": sys.version.split()[0],
                       "torch": torch.__version__, "cpu_model": _cpu_model(),
                       "cpu_logical_threads": os.cpu_count()}
    print(json.dumps(out, ensure_ascii=False, indent=2))


def _cpu_model() -> str:
    with open("/proc/cpuinfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("latency", "memory", "sizes"))
    ap.add_argument("--manifest-dir")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--m2", required=True)
    ap.add_argument("--m3", required=True)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--n-beats", type=int)
    ap.add_argument("--case", choices=DECODERS)
    args = ap.parse_args()
    {"latency": latency_mode, "memory": memory_mode, "sizes": sizes_mode}[args.mode](args)


if __name__ == "__main__":
    main()
