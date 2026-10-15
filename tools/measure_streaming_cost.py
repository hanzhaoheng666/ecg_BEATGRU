#!/usr/bin/env python3
"""Measure the deployed cost of the locked BEATGRU model.

This tool performs no training and changes no model.  It loads the frozen
DS1 checkpoints, replays the DS2 records in two inference modes, and reports

* per-beat wall-clock latency (p50 / p99, not just the mean) for
  - BeatCNN alone (no recursion) and
  - the complete BEATGRU (frozen CNN + causal history decoder),
  both in streaming (one beat at a time, hidden state carried) and in the
  whole-record batched mode used by the evaluation scripts;
* parameter count and on-disk model size in fp32 and (optionally) int8;
* process resident-memory measurements for a streaming pass.

The 200 ms in the paper is the *decision-delay constant* of the causal window
(``R_i + 200 ms``), i.e. a time budget, not a measured compute cost.  What is
measured here is the compute actually spent after the window has been filled.
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

from beatgru.data import load_records
from beatgru.models import BeatCNN, LogitGRU


def rss_kb() -> int:
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return -1


def hardware() -> dict[str, object]:
    threads = os.cpu_count()
    model_name = "unknown"
    with open("/proc/cpuinfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("model name"):
                model_name = line.split(":", 1)[1].strip()
                break
    total_kb = 0
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemTotal:"):
                total_kb = int(line.split()[1])
                break
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cpu_model": model_name,
        "cpu_logical_threads": threads,
        "mem_total_gib": round(total_kb / 1024 / 1024, 2),
    }


def gpu_name(device: torch.device) -> str:
    if device.type != "cuda":
        return "none"
    return torch.cuda.get_device_name(device)


def checkpoint_size_kb(path: Path) -> float:
    return round(path.stat().st_size / 1024, 1)


def param_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def int8_dynamic_size_kb(model: nn.Module) -> float | None:
    """Size of a dynamically quantised copy of the frozen model, or None."""
    try:
        quantised = torch.ao.quantization.quantize_dynamic(model, {nn.Linear, nn.GRU}, dtype=torch.qint8)
    except Exception:  # noqa: BLE001 - optional measurement, never fatal
        return None
    buffer = io.BytesIO()
    try:
        torch.save(quantised.state_dict(), buffer)
    except Exception:  # noqa: BLE001
        return None
    return round(buffer.getbuffer().nbytes / 1024, 1)


def load_models(backbone: Path, history: Path, device: torch.device) -> tuple[BeatCNN, LogitGRU, dict]:
    backbone_ckpt = torch.load(backbone, map_location="cpu", weights_only=False)
    base = BeatCNN(int(backbone_ckpt.get("channels", 3)), 8).to(device)
    base.load_state_dict(backbone_ckpt["model"])
    base.eval()
    history_ckpt = torch.load(history, map_location="cpu", weights_only=False)
    saved = history_ckpt["args"]
    history_model = LogitGRU(8, int(saved["hidden"]), float(saved["max_correction"])).to(device)
    history_model.load_state_dict(history_ckpt["model"])
    history_model.eval()
    return base, history_model, saved


def percentiles(durations_ms: list[float]) -> dict[str, float]:
    values = np.asarray(durations_ms, dtype=np.float64)
    return {
        "n_beats": int(len(values)),
        "mean_ms": round(float(values.mean()), 4),
        "p50_ms": round(float(np.quantile(values, 0.50)), 4),
        "p90_ms": round(float(np.quantile(values, 0.90)), 4),
        "p99_ms": round(float(np.quantile(values, 0.99)), 4),
        "max_ms": round(float(values.max()), 4),
        "sd_ms": round(float(statistics.pstdev(values)), 4),
    }


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def stream_pass(base: BeatCNN, history_model: LogitGRU, tensors: list[tuple[torch.Tensor, torch.Tensor]],
                device: torch.device, warmup: int, model_scope: str) -> list[float]:
    """One beat at a time; the hidden state is carried within a record only."""
    durations: list[float] = []
    seen = 0
    for wave, rr in tensors:
        state = None
        for index in range(wave.shape[0]):
            x = wave[index:index + 1]
            r = rr[index:index + 1]
            sync(device)
            start = time.perf_counter_ns()
            logits = base(x, r)
            if model_scope == "beatgru":
                _, state = history_model.forward_chunk(logits, r, state)
            sync(device)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            seen += 1
            if seen > warmup:
                durations.append(elapsed)
    return durations


@torch.inference_mode()
def batched_pass(base: BeatCNN, history_model: LogitGRU, tensors: list[tuple[torch.Tensor, torch.Tensor]],
                 device: torch.device, warmup: int, model_scope: str, chunk: int) -> list[float]:
    """One whole record (or a fixed chunk of it) per forward call."""
    durations: list[float] = []
    seen = 0
    for wave, rr in tensors:
        state = None
        for start in range(0, wave.shape[0], chunk):
            stop = min(start + chunk, wave.shape[0])
            x = wave[start:stop]
            r = rr[start:stop]
            sync(device)
            t0 = time.perf_counter_ns()
            logits = base(x, r)
            if model_scope == "beatgru":
                _, state = history_model.forward_chunk(logits, r, state)
            sync(device)
            elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
            n = stop - start
            seen += n
            if seen > warmup:
                durations.append(elapsed_ms / n)
            if state is not None:
                state = state.detach()
    return durations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup", type=int, default=500, help="beats discarded before timing")
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--threads-multi", type=int, default=8,
                        help="thread count for the multi-threaded CPU configuration; large host "
                             "thread counts add scheduling overhead that no edge deployment sees")
    parser.add_argument("--timing-also-gpu", action="store_true", default=True)
    parser.add_argument("--quantize-int8", action="store_true",
                        help="also time a dynamically quantised (int8) copy on CPU single-thread")
    args = parser.parse_args()

    manifest = Path(args.manifest_dir)
    backbone_path = Path(args.backbone)
    history_path = Path(args.history)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records = load_records(manifest, "test", 250, 350, 200, lead_selector="canonical_first")
    n_beats = sum(len(record.labels) for record in records)
    cpu = torch.device("cpu")
    baseline_rss = rss_kb()

    report: dict[str, object] = {
        "hardware": hardware(),
        "manifest_dir": str(manifest),
        "dataset": "mit_bih_arrhythmia",
        "split": "test",
        "n_records": len(records),
        "n_beats_with_f_q": n_beats,
        "decision_delay_budget_ms": 200.0,
        "warmup_beats": args.warmup,
        "checkpoints": {"backbone": str(backbone_path), "history": str(history_path)},
        "note": ("latency is measured model compute per beat after the causal window is filled; "
                 "the 200 ms figure is the algorithmic decision-delay budget, not a measured cost"),
    }

    # ---------- model sizes (CPU, no timing) ----------
    base_cpu, history_cpu, saved = load_models(backbone_path, history_path, cpu)
    sizes = {
        "beatcnn_params": param_count(base_cpu),
        "logitgru_params": param_count(history_cpu),
        "beatgru_params_total": param_count(base_cpu) + param_count(history_cpu),
        "backbone_file_kb": checkpoint_size_kb(backbone_path),
        "history_file_kb": checkpoint_size_kb(history_path),
        "beatgru_file_kb": round(checkpoint_size_kb(backbone_path) + checkpoint_size_kb(history_path), 1),
        "beatcnn_int8_kb": int8_dynamic_size_kb(base_cpu),
        "beatgru_int8_kb": (None if int8_dynamic_size_kb(history_cpu) is None else
                            round(int8_dynamic_size_kb(base_cpu) + int8_dynamic_size_kb(history_cpu), 1)),
    }
    report["model_size"] = sizes

    # ---------- latency ----------
    tensors_cpu = [(torch.from_numpy(record.wave), torch.from_numpy(record.rr)) for record in records]
    latency: dict[str, object] = {}

    for label, threads in (("cpu_1_thread", 1), ("cpu_%d_threads" % args.threads_multi, args.threads_multi)):
        torch.set_num_threads(int(threads))
        for scope in ("beatcnn", "beatgru"):
            durations = stream_pass(base_cpu, history_cpu, tensors_cpu, cpu, args.warmup, scope)
            latency[f"streaming_{label}_{scope}"] = percentiles(durations)
            durations = batched_pass(base_cpu, history_cpu, tensors_cpu, cpu, args.warmup, scope, args.chunk)
            latency[f"batched_{label}_{scope}"] = percentiles(durations)
    torch.set_num_threads(os.cpu_count())
    report["cpu_threads_configuration"] = {"single_thread": 1, "multi_thread": args.threads_multi,
                                           "host_logical_threads": os.cpu_count()}

    if args.timing_also_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        base_gpu, history_gpu, _ = load_models(backbone_path, history_path, device)
        tensors_gpu = [(wave.to(device), rr.to(device)) for wave, rr in tensors_cpu]
        for scope in ("beatcnn", "beatgru"):
            durations = stream_pass(base_gpu, history_gpu, tensors_gpu, device, args.warmup, scope)
            latency[f"streaming_gpu_{scope}"] = percentiles(durations)
            durations = batched_pass(base_gpu, history_gpu, tensors_gpu, device, args.warmup, scope, args.chunk)
            latency[f"batched_gpu_{scope}"] = percentiles(durations)
        report["gpu"] = gpu_name(device)
        del base_gpu, history_gpu, tensors_gpu
        torch.cuda.empty_cache()
    else:
        report["gpu"] = "none"
    if args.quantize_int8:
        torch.set_num_threads(1)
        try:
            base_i8 = torch.ao.quantization.quantize_dynamic(base_cpu, {nn.Linear, nn.GRU}, dtype=torch.qint8)
            history_i8 = torch.ao.quantization.quantize_dynamic(history_cpu, {nn.Linear, nn.GRU}, dtype=torch.qint8)
            for scope, first, second in (("beatcnn", base_i8, None), ("beatgru", base_i8, history_i8)):
                durations = stream_pass(first, second, tensors_cpu, cpu, args.warmup, scope)
                latency[f"streaming_cpu_1_thread_int8_{scope}"] = percentiles(durations)
        except Exception as error:  # noqa: BLE001 - optional measurement
            report["int8_latency_error"] = f"{type(error).__name__}: {error}"
    report["latency"] = latency

    # ---------- resident memory during a streaming pass ----------
    torch.set_num_threads(1)
    after_models = rss_kb()
    tensors_loaded = [(torch.from_numpy(record.wave), torch.from_numpy(record.rr)) for record in records]
    after_records = rss_kb()
    peak = after_records
    with torch.inference_mode():
        for wave, rr in tensors_loaded:
            state = None
            for index in range(wave.shape[0]):
                x = wave[index:index + 1]
                r = rr[index:index + 1]
                logits = base_cpu(x, r)
                _, state = history_cpu.forward_chunk(logits, r, state)
            peak = max(peak, rss_kb())
    report["memory_rss_kb"] = {
        "baseline_after_torch_import": baseline_rss,
        "after_loading_frozen_models": after_models,
        "after_loading_ds2_windows_for_evaluation": after_records,
        "peak_during_streaming_pass": peak,
        "model_only_increment_kb": after_models - baseline_rss,
        "streaming_peak_increment_over_models_kb": peak - after_models,
        "note": ("the evaluation harness materialises all DS2 windows before the pass; a true "
                 "edge deployment streams one 550 ms window (138x3 float32 = 1.6 KiB) at a time"),
    }

    (out / "latency_memory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 实测延迟与内存（锁定 BEATGRU，DS2 全量回放）", "",
             f"- 硬件：{report['hardware']['cpu_model']}，逻辑线程 {report['hardware']['cpu_logical_threads']}，内存 {report['hardware']['mem_total_gib']} GiB",
             f"- GPU：{report['gpu']}", f"- 预热丢弃 {args.warmup} 拍；DS2 共 {len(records)} 条记录 / {n_beats} 个事件",
             f"- 决策延迟预算（算法常量）：200 ms；下表为模型实际计算耗时", "",
             "| 模式 | 配置 | p50 (ms) | p99 (ms) | mean (ms) |", "|---|---|---:|---:|---:|"]
    multi = f"cpu_{args.threads_multi}_threads"
    order = [
        ("streaming_cpu_1_thread_beatcnn", "流式逐拍", "仅 BeatCNN / CPU 单线程"),
        ("streaming_cpu_1_thread_beatgru", "流式逐拍", "完整 BEATGRU / CPU 单线程"),
        (f"streaming_{multi}_beatcnn", "流式逐拍", f"仅 BeatCNN / CPU {args.threads_multi} 线程"),
        (f"streaming_{multi}_beatgru", "流式逐拍", f"完整 BEATGRU / CPU {args.threads_multi} 线程"),
        ("batched_cpu_1_thread_beatcnn", "整段批处理", "仅 BeatCNN / CPU 单线程"),
        ("batched_cpu_1_thread_beatgru", "整段批处理", "完整 BEATGRU / CPU 单线程"),
        ("streaming_gpu_beatcnn", "流式逐拍", "仅 BeatCNN / GPU"),
        ("streaming_gpu_beatgru", "流式逐拍", "完整 BEATGRU / GPU"),
        ("batched_gpu_beatcnn", "整段批处理", "仅 BeatCNN / GPU"),
        ("batched_gpu_beatgru", "整段批处理", "完整 BEATGRU / GPU"),
    ]
    for key, mode, label in order:
        if key not in latency:
            continue
        row = latency[key]
        lines.append(f"| {mode} | {label} | {row['p50_ms']:.4f} | {row['p99_ms']:.4f} | {row['mean_ms']:.4f} |")
    lines += ["", "## 模型体积与参数量", "",
              "| 模型 | 参数量 | fp32 文件 (KB) | int8 动态量化 (KB) |", "|---|---:|---:|---:|",
              f"| BeatCNN | {sizes['beatcnn_params']:,} | {sizes['backbone_file_kb']} | {sizes['beatcnn_int8_kb']} |",
              f"| LogitGRU | {sizes['logitgru_params']:,} | {sizes['history_file_kb']} | "
              f"{None if sizes['beatgru_int8_kb'] is None or sizes['beatcnn_int8_kb'] is None else round(sizes['beatgru_int8_kb'] - sizes['beatcnn_int8_kb'], 1)} |",
              f"| 完整 BEATGRU | {sizes['beatgru_params_total']:,} | {sizes['beatgru_file_kb']} | {sizes['beatgru_int8_kb']} |",
              "", "## 常驻内存", ""]
    for key, value in report["memory_rss_kb"].items():
        lines.append(f"- `{key}`: {value}")
    (out / "latency_memory.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
