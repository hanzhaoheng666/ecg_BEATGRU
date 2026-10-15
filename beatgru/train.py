"""Train/evaluate the BEATGRU emission CNN and causal history decoder."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import Record, load_records
from .metrics import metric
from .models import BeatCNN, EmbeddingGRU, LogitGRU


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def device_from(name: str) -> torch.device:
    return torch.device(name if name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))


def record_ids_from_file(path: str) -> list[str] | None:
    """Read a predeclared record subset for DS1 group cross-validation."""
    if not path:
        return None
    values = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    if not values or len(set(values)) != len(values):
        raise ValueError(f"record-id file must be nonempty and unique: {path}")
    return values


def weights(rows: list[Record], power: float) -> tuple[np.ndarray, torch.Tensor]:
    labels = np.concatenate([row.labels[row.score] for row in rows])
    counts = np.bincount(labels, minlength=3).astype(float)
    value = np.power(counts.max() / np.maximum(counts, 1), power); value /= value.mean()
    return counts, torch.tensor(value, dtype=torch.float32)


def initialize_from_icentia(model: BeatCNN, checkpoint: str) -> list[str]:
    """Transfer only the label-free morphology conv stem from Icentia SSL.

    SSL saw one raw physical lead.  Its first kernel is therefore copied to
    the raw channel of the raw/d1/d2 target encoder; derivative channels stay
    at their normal random initialization and are trained on DS1.  Later
    convolution blocks have identical shapes and are copied directly.
    """
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    encoder = source.get("encoder", source.get("model", source))
    if any(str(k).startswith("conv.") for k in encoder):
        encoder = {str(k)[5:]: v for k, v in encoder.items() if str(k).startswith("conv.")}
    own = model.state_dict(); loaded=[]
    for key, value in encoder.items():
        target = f"conv.{key}"
        if target not in own:
            continue
        if own[target].shape == value.shape:
            own[target].copy_(value); loaded.append(target)
        elif key == "0.weight":
            if value.ndim != 3 or own[target].shape[0] != value.shape[0] or own[target].shape[2] != value.shape[2]:
                continue
            own[target].zero_(); own[target][:, :1] = value
            loaded.append(target + "[raw-only]")
    if len(loaded) < 8:
        raise ValueError(f"too few Icentia conv tensors matched {checkpoint}: {loaded}")
    model.load_state_dict(own); return loaded


def simulate_low_bandwidth(waveform: torch.Tensor, probability: float, source_rate: int) -> torch.Tensor:
    """Training-only, causal-window bandwidth augmentation.

    `waveform` already contains only R_i-350ms..R_i+200ms.  The raw channel is
    area-downsampled to simulate an acquisition rate, linearly restored to the
    fixed 250-Hz input grid, and derivative channels are rebuilt.  No target
    data or samples outside the legal decision window are used.
    """
    if probability <= 0:
        return waveform
    if waveform.ndim != 3 or waveform.shape[1] != 3 or source_rate >= 250:
        raise ValueError("bandwidth augmentation expects [B,3,T] and source_rate < 250")
    chosen = torch.rand(waveform.shape[0], device=waveform.device) < probability
    if not bool(chosen.any()):
        return waveform
    raw = waveform[chosen, :1]
    low_length = max(2, round(raw.shape[-1] * source_rate / 250.0))
    low = F.interpolate(raw, size=low_length, mode="area")
    restored = F.interpolate(low, size=raw.shape[-1], mode="linear", align_corners=False)
    first = torch.diff(restored, dim=-1, prepend=restored[..., :1])
    second = torch.diff(first, dim=-1, prepend=first[..., :1])
    result = waveform.clone(); result[chosen] = torch.cat((restored, first, second), dim=1)
    return result


def augment_lead_order(waveform: torch.Tensor, probability: float) -> torch.Tensor:
    """Swap the two leads while preserving raw/d1/d2 channel semantics."""
    if probability <= 0:
        return waveform
    if waveform.ndim != 3 or waveform.shape[1] != 6:
        raise ValueError("lead-order augmentation expects [B,6,T]")
    chosen = torch.rand(waveform.shape[0], device=waveform.device) < probability
    if not bool(chosen.any()):
        return waveform
    result = waveform.clone()
    # derivative_channels is [raw lead1, raw lead2, d1 lead1, d1 lead2,
    # d2 lead1, d2 lead2], not lead-major triplets.
    result[chosen] = waveform[chosen][:, [1, 0, 3, 2, 5, 4]]
    return result


def emissions(model: BeatCNN, rows: list[Record], device: torch.device, batch: int = 2048,
              include_embedding: bool = False) -> list[dict[str, object]]:
    model.eval(); output = []
    with torch.inference_mode():
        for row in rows:
            chunks = []
            features = []
            for i in range(0, len(row.labels), batch):
                wave = torch.from_numpy(row.wave[i:i + batch]).to(device)
                rr = torch.from_numpy(row.rr[i:i + batch]).to(device)
                encoded = model.encode(wave, rr)
                chunks.append(model.head(encoded).cpu().numpy())
                if include_embedding:
                    features.append(encoded.cpu().numpy())
            item = {"record_id": row.record_id, "logits": np.concatenate(chunks).astype(np.float32),
                    "rr": row.rr, "labels": row.labels, "score": row.score}
            if include_embedding:
                item["embedding"] = np.concatenate(features).astype(np.float32)
            output.append(item)
    return output


def forward_history(model: LogitGRU | EmbeddingGRU, row: dict[str, object], start: int,
                    stop: int, state: torch.Tensor | None, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.from_numpy(row["logits"][start:stop]).to(device)
    rr = torch.from_numpy(row["rr"][start:stop]).to(device)
    if isinstance(model, EmbeddingGRU):
        return model.forward_chunk(logits, rr, torch.from_numpy(row["embedding"][start:stop]).to(device), state)
    return model.forward_chunk(logits, rr, state)


def evaluate(model: LogitGRU | EmbeddingGRU, rows: list[dict[str, object]], device: torch.device, chunk: int) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    model.eval(); all_y = []; all_p = []; all_logits = []
    with torch.inference_mode():
        for row in rows:
            state = None; parts = []
            for start in range(0, len(row["labels"]), chunk):
                stop = start + chunk
                logits, state = forward_history(model, row, start, stop, state, device)
                parts.append(logits.cpu().numpy()); state = state.detach()
            logits = np.concatenate(parts); keep = row["score"]
            all_logits.append(logits[keep]); all_y.append(row["labels"][keep]); all_p.append(logits.argmax(1)[keep])
    return metric(np.concatenate(all_y), np.concatenate(all_p)), np.concatenate(all_logits), np.concatenate(all_y), np.concatenate(all_p)


def run_cnn(args: argparse.Namespace) -> None:
    selected = record_ids_from_file(args.record_ids_file)
    rows = load_records(args.manifest_dir, "train_plus_val", args.target_fs, args.lookback_ms, args.delay_ms,
                        lead_selector=args.lead_selector)
    if selected is not None:
        rows = load_records(args.manifest_dir, "train_plus_val", args.target_fs, args.lookback_ms, args.delay_ms,
                            lead_selector=args.lead_selector, record_ids=selected)
    x = np.concatenate([row.wave[row.score] for row in rows]); rr = np.concatenate([row.rr[row.score] for row in rows]); y = np.concatenate([row.labels[row.score] for row in rows])
    counts, class_weights = weights(rows, args.power); device = device_from(args.device)
    channels = int(x.shape[1])
    model = BeatCNN(channels, 8).to(device)
    loaded_encoder = initialize_from_icentia(model, args.pretrained_encoder) if args.pretrained_encoder else []
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(rr), torch.from_numpy(y)), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for xb, rb, yb in loader:
            xb = xb.to(device)
            xb = simulate_low_bandwidth(xb, args.bandwidth_augment_prob, args.bandwidth_augment_rate)
            xb = augment_lead_order(xb, args.lead_swap_prob)
            optimizer.zero_grad(set_to_none=True); loss = loss_fn(model(xb, rb.to(device)), yb.to(device)); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0); optimizer.step(); losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses))}); print(json.dumps(history[-1]), flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "channels": channels, "class_counts": counts.tolist(), "args": vars(args), "pretrained_encoder": str(Path(args.pretrained_encoder).resolve()) if args.pretrained_encoder else None, "loaded_encoder_keys": loaded_encoder,
                "protocol": {"strict_Ri_plus_delay_online": True, "feature_mode": "raw_d1_d2", "future_waveform": False, "future_rpeak_or_post_rr": False, "future_label": False}}, out / "derivative_cnn.pt")
    (out / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def run_gru(args: argparse.Namespace) -> None:
    device = device_from(args.device); backbone_ckpt = torch.load(args.backbone, map_location="cpu", weights_only=False)
    base = BeatCNN(int(backbone_ckpt.get("channels", 3)), 8).to(device); base.load_state_dict(backbone_ckpt["model"]); base.eval()
    for parameter in base.parameters(): parameter.requires_grad = False
    selected = record_ids_from_file(args.record_ids_file)
    rows = emissions(base, load_records(args.manifest_dir, "train_plus_val", args.target_fs, args.lookback_ms, args.delay_ms,
                                        lead_selector=args.lead_selector, record_ids=selected), device,
                     include_embedding=args.history_input == "embedding")
    pseudo = [Record(str(row["record_id"]), np.empty((0,)), row["rr"], row["labels"], row["score"]) for row in rows]
    counts, class_weights = weights(pseudo, args.power)
    model_type = EmbeddingGRU if args.history_input == "embedding" else LogitGRU
    model = model_type(8, args.hidden, args.max_correction).to(device) if model_type is LogitGRU else model_type(160, 8, args.hidden, args.max_correction).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device)); history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); order = list(range(len(rows))); random.Random(args.seed + epoch).shuffle(order); losses = []
        for index in order:
            row = rows[index]; state = None; terms = []; labels = torch.from_numpy(row["labels"]).to(device); keep = torch.from_numpy(row["score"]).to(device)
            optimizer.zero_grad(set_to_none=True)
            n_chunks = max(1, (len(row["labels"]) + args.chunk - 1) // args.chunk)
            for start in range(0, len(row["labels"]), args.chunk):
                logits, state = forward_history(model, row, start, start + args.chunk, state, device)
                active = keep[start:start + args.chunk]
                if bool(active.any()): terms.append(loss_fn(logits[active], labels[start:start + args.chunk][active]) / n_chunks)
                state = state.detach()
            if terms:
                loss = torch.stack(terms).sum(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 3.0); optimizer.step(); losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses))}); print(json.dumps(history[-1]), flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "backbone": str(Path(args.backbone).resolve()), "class_counts": counts.tolist(), "args": vars(args),
                "protocol": {"strict_Ri_plus_delay_online": True, "input": f"frozen CNN {args.history_input} + causal RR + GRU", "state_reset": "record boundary", "future_waveform": False, "future_rpeak_or_post_rr": False, "future_label": False}}, out / "beatgru.pt")
    (out / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def run_evaluate(args: argparse.Namespace) -> None:
    device = device_from(args.device); ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False); saved = ckpt["args"]
    keys = ("target_fs", "lookback_ms", "delay_ms", "hidden", "max_correction", "chunk")
    mismatch = {key: (saved[key], getattr(args, key)) for key in keys if saved[key] != getattr(args, key)}
    if mismatch: raise ValueError(f"checkpoint/config mismatch: {mismatch}")
    backbone_path = args.backbone or ckpt["backbone"]
    backbone = torch.load(backbone_path, map_location="cpu", weights_only=False)
    base = BeatCNN(int(backbone.get("channels", 3)), 8).to(device); base.load_state_dict(backbone["model"])
    selected = record_ids_from_file(args.record_ids_file)
    history_input = saved.get("history_input", "logits")
    rows = emissions(base, load_records(args.manifest_dir, args.target_split, args.target_fs, args.lookback_ms, args.delay_ms,
                                        dataset=args.dataset, lead_selector=args.lead_selector, record_ids=selected), device,
                     include_embedding=history_input == "embedding")
    model = (EmbeddingGRU(160, 8, args.hidden, args.max_correction) if history_input == "embedding"
             else LogitGRU(8, args.hidden, args.max_correction)).to(device)
    model.load_state_dict(ckpt["model"])
    values, logits, labels, predictions = evaluate(model, rows, device, args.chunk)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "test_predictions.npz", logits=logits, labels=labels, predictions=predictions)
    protocol = dict(ckpt["protocol"]); protocol["test_loaded"] = True
    payload = {"metrics": values, "n_target_records": len(rows), "target_dataset": args.dataset,
               "target_split": args.target_split, "lead_selector": args.lead_selector,
               "record_ids_file": str(Path(args.record_ids_file).resolve()) if args.record_ids_file else None,
               "protocol": protocol}
    (out / "test_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(); sub = root.add_subparsers(dest="command", required=True)
    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--manifest-dir", required=True); p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="cuda")
        p.add_argument("--target-fs", type=int, default=250); p.add_argument("--lookback-ms", type=int, default=350); p.add_argument("--delay-ms", type=int, default=200); p.add_argument("--record-ids-file", default="", help="predeclared record IDs for source-only group CV"); p.add_argument("--lead-selector", choices=("canonical_first", "lead_ii", "source_matched_ii", "source_matched_pair"), default="canonical_first")
    cnn = sub.add_parser("train-cnn"); common(cnn); cnn.add_argument("--epochs", type=int, default=15); cnn.add_argument("--batch-size", type=int, default=512); cnn.add_argument("--lr", type=float, default=5e-4); cnn.add_argument("--weight-decay", type=float, default=1e-4); cnn.add_argument("--power", type=float, default=.6); cnn.add_argument("--bandwidth-augment-prob", type=float, default=0.0); cnn.add_argument("--bandwidth-augment-rate", type=int, default=128); cnn.add_argument("--lead-swap-prob", type=float, default=0.0); cnn.add_argument("--pretrained-encoder", default="", help="label-free Icentia SSL checkpoint; conv stem only")
    gru = sub.add_parser("train-gru"); common(gru); gru.add_argument("--backbone", required=True); gru.add_argument("--epochs", type=int, default=15); gru.add_argument("--lr", type=float, default=3e-4); gru.add_argument("--weight-decay", type=float, default=1e-4); gru.add_argument("--power", type=float, default=.4); gru.add_argument("--hidden", type=int, default=128); gru.add_argument("--max-correction", type=float, default=.75); gru.add_argument("--chunk", type=int, default=8192); gru.add_argument("--history-input", choices=("logits", "embedding"), default="logits")
    evaluate_p = sub.add_parser("evaluate"); common(evaluate_p); evaluate_p.add_argument("--checkpoint", required=True); evaluate_p.add_argument("--backbone", default="", help="override the CNN path saved in a released checkpoint"); evaluate_p.add_argument("--dataset", default="mit_bih_arrhythmia", help="CSV stem in manifest-dir; e.g. mit_bih_svdb or st_petersburg_incart"); evaluate_p.add_argument("--target-split", choices=("train", "val", "test", "all"), default="test", help="all evaluates every target record without fitting on it"); evaluate_p.add_argument("--hidden", type=int, default=128); evaluate_p.add_argument("--max-correction", type=float, default=.75); evaluate_p.add_argument("--chunk", type=int, default=8192)
    return root


def main() -> None:
    args = parser().parse_args(); seed_all(args.seed)
    if args.command == "train-cnn": run_cnn(args)
    elif args.command == "train-gru": run_gru(args)
    else: run_evaluate(args)


if __name__ == "__main__": main()
