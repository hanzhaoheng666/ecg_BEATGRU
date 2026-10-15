#!/usr/bin/env python3
"""Does the online state grow with record length, for M2 (windowed) and M3 (recurrent)?

Motivation
----------
An earlier write-up proposed "bounded state" as M2's advantage over M3.  That
claim has to be checked, because M3's decoder is a GRU: its carried state is a
fixed-size hidden vector, so at the algorithmic level M3 is *also* O(1) in
record length.  What differs between the two is the length of the input
dependency, not the size of the state.

This tool therefore measures, honestly and separately:

* the analytic per-beat state footprint each decoder must carry online
  (M2: the last ``window`` tokens; M3: one hidden vector);
* peak resident memory while streaming records of increasing length, using each
  decoder's *correct online form*
  - M2: a ring buffer of the last ``window`` tokens, GRU re-run per beat
  - M3: a carried hidden state
* and, as a control, M2 evaluated through ``forward_windows`` on the whole
  record, which materialises an ``[n, window, hidden]`` tensor.  That is a
  property of the batched implementation, not of the architecture, and it is
  reported so the two are not confused.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beatgru.models import LogitGRU, ShortLogitGRU  # noqa: E402

WINDOW = 32


def rss_kb() -> int:
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return -1


def load(backbone: Path, history: Path) -> tuple[LogitGRU, ShortLogitGRU, dict]:
    """Load the same trained weights into both decoder forms.

    ``LogitGRU`` and ``ShortLogitGRU`` share parameter shapes, so one checkpoint
    parameterises both; the difference is entirely in how history is consumed.
    """
    ck = torch.load(history, map_location="cpu", weights_only=False)
    cfg = ck["args"]
    recurrent = LogitGRU(cfg.get("rr_dim", 8), cfg["hidden"], cfg["max_correction"])
    windowed = ShortLogitGRU(cfg.get("rr_dim", 8), cfg["hidden"], cfg["max_correction"], WINDOW)
    recurrent.load_state_dict(ck["model"]); recurrent.eval()
    windowed.load_state_dict(ck["model"]); windowed.eval()
    return recurrent, windowed, cfg


@torch.inference_mode()
def stream_recurrent(model: LogitGRU, logits: torch.Tensor, rr: torch.Tensor) -> int:
    """Carry one hidden state across the whole record (M3's online form)."""
    state = None
    for index in range(logits.shape[0]):
        _, state = model.forward_chunk(logits[index:index + 1], rr[index:index + 1], state)
    return int(logits.shape[0])


@torch.inference_mode()
def stream_ring(model: ShortLogitGRU, logits: torch.Tensor, rr: torch.Tensor) -> int:
    """Keep only the last ``window`` tokens (M2's online form)."""
    tokens = model.input(torch.cat((logits, rr), dim=1))
    buffer: torch.Tensor | None = None
    for index in range(tokens.shape[0]):
        token = tokens[index:index + 1]
        buffer = token if buffer is None else torch.cat((buffer, token), dim=0)[-model.window:]
        model.gru(buffer.unsqueeze(0))
    return int(logits.shape[0])


@torch.inference_mode()
def batched_windows(model: ShortLogitGRU, logits: torch.Tensor, rr: torch.Tensor) -> int:
    """The batched evaluation form: materialises [n, window, hidden]."""
    model.forward_windows(logits, rr)
    return int(logits.shape[0])


def run_case(fn, model, logits: torch.Tensor, rr: torch.Tensor) -> dict[str, float]:
    before = rss_kb()
    start = time.perf_counter()
    beats = fn(model, logits, rr)
    elapsed = time.perf_counter() - start
    after = rss_kb()
    return {"beats": beats, "seconds": round(elapsed, 4),
            "rss_before_kb": before, "rss_after_kb": after,
            "rss_increment_kb": after - before}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True, type=Path, help="locked M3 checkpoint (also parameterises M2)")
    ap.add_argument("--out", type=Path, help="aggregate output (multi-case mode)")
    ap.add_argument("--lengths", default="1000,4000,16000,64000")
    ap.add_argument("--case", choices=("m2_ring", "m3_state", "m2_batched"),
                    help="run a single case in this process and print one JSON line "
                         "(use this from a shell loop so each case starts from a clean RSS)")
    ap.add_argument("--n-beats", type=int, help="record length for --case mode")
    args = ap.parse_args()
    torch.set_num_threads(1)
    recurrent, windowed, cfg = load(Path(args.history), Path(args.history))
    hidden = int(cfg["hidden"]); rr_dim = int(cfg.get("rr_dim", 8)); classes = 3

    analytic = {
        "hidden": hidden,
        "window": WINDOW,
        "m2_online_state_bytes_fp32": WINDOW * (classes + rr_dim) * 4,
        "m3_online_state_bytes_fp32": hidden * 4,
        "m2_batched_window_tensor_bytes_per_beat": WINDOW * hidden * 4,
        "note": ("both decoders hold O(1) state online: the windowed form a 32-token "
                 "buffer, the recurrent form one hidden vector. The batched "
                 "forward_windows path materialises an [n, window, hidden] tensor and is "
                 "O(n), but that is an implementation property, not an architectural one."),
    }

    if args.case:
        n = int(args.n_beats)
        generator = torch.Generator().manual_seed(20260927)
        logits = torch.randn(n, classes, generator=generator)
        rr = torch.randn(n, rr_dim, generator=generator)
        fn, model = {"m2_ring": (stream_ring, windowed),
                     "m3_state": (stream_recurrent, recurrent),
                     "m2_batched": (batched_windows, windowed)}[args.case]
        baseline = rss_kb()
        case = run_case(fn, model, logits, rr)
        case.update({"case": args.case, "n_beats": n, "rss_baseline_before_case_kb": baseline})
        print(json.dumps(case, ensure_ascii=False))
        return

    rows = []
    generator = torch.Generator().manual_seed(20260927)
    for length in [int(x) for x in args.lengths.split(",")]:
        logits = torch.randn(length, classes, generator=generator)
        rr = torch.randn(length, rr_dim, generator=generator)
        case = {"n_beats": length}
        case["m2_ring_buffer"] = run_case(stream_ring, windowed, logits, rr)
        case["m3_carried_state"] = run_case(stream_recurrent, recurrent, logits, rr)
        case["m2_batched_windows"] = run_case(batched_windows, windowed, logits, rr)
        rows.append(case)
        print(json.dumps(case, ensure_ascii=False))

    payload = {
        "analytic_state": analytic,
        "cases": rows,
        "interpretation": ("a flat online increment means memory does not grow with record "
                           "length; growth in m2_batched_windows is an artefact of "
                           "materialising the window tensor"),
    }
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "state_boundedness.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
