#!/usr/bin/env python3
"""Independent strict-causal N/S/V/F CNN+GRU experiment line.

This file deliberately leaves the published N/S/V pipeline untouched.  It
scores F (fusion beats) as a fourth class while retaining Q and unsupported
symbols as chronology-only, unscored events.
"""
from __future__ import annotations

import argparse, csv, json, random, sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beatgru.data import Record, _preferred_lead, causal_windows, derivative_channels, direct_causal_rr
from beatgru.models import BeatCNN, LogitGRU

NAMES = ("N", "S", "V", "F")
LABELS = {"N": 0, "S": 1, "V": 2, "F": 3}


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, object]:
    cm = np.zeros((4, 4), dtype=np.int64)
    for actual, pred in zip(y, p):
        if 0 <= actual < 4 and 0 <= pred < 4: cm[actual, pred] += 1
    precision = np.divide(np.diag(cm), cm.sum(0), out=np.zeros(4), where=cm.sum(0) > 0)
    recall = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(4), where=cm.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(4), where=(precision + recall) > 0)
    return {"n_samples": int(len(y)), "accuracy": float(np.mean(y == p)), "balanced_accuracy": float(recall.mean()),
            "macro_f1": float(f1.mean()), "sv_f1": float(f1[1:3].mean()),
            **{f"{name}_precision": float(precision[i]) for i, name in enumerate(NAMES)},
            **{f"{name}_recall": float(recall[i]) for i, name in enumerate(NAMES)},
            **{f"{name}_f1": float(f1[i]) for i, name in enumerate(NAMES)}, "confusion_matrix": cm.tolist()}


def load4(manifest: Path, split: str, target_fs: int, lookback_ms: int, delay_ms: int) -> list[Record]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with (manifest / "mit_bih_arrhythmia.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if split == "all" or row["split"] == split or (split == "train_plus_val" and row["split"] == "train"):
                grouped.setdefault(row["record_id"], []).append(row)
    records: list[Record] = []
    import wfdb
    for record_id, rows in sorted(grouped.items()):
        peaks = np.asarray([int(row["r_peak_sample"]) for row in rows], dtype=np.int64)
        event_ids = [int(row.get("event_index", index)) for index, row in enumerate(rows)]
        if event_ids != list(range(len(rows))) or (len(peaks) > 1 and np.any(np.diff(peaks) <= 0)):
            raise ValueError(f"non-chronological manifest: {record_id}")
        wf = wfdb.rdrecord(str(Path(rows[0]["header_path"]).with_suffix("")))
        signal = _preferred_lead(wf, "canonical_first"); fs = int(round(float(wf.fs)))
        valid = peaks + round(delay_ms * fs / 1000) < len(signal)
        peaks = peaks[valid]
        labels = np.asarray([LABELS.get(row.get("aami_label", ""), -100) for row in rows], dtype=np.int64)[valid]
        score = (labels >= 0)
        records.append(Record(record_id, derivative_channels(causal_windows(signal, peaks, fs, target_fs, lookback_ms, delay_ms)),
                              direct_causal_rr(peaks, fs), labels, score))
    if not records: raise ValueError(f"no records for split={split}")
    return records


def class_weights(rows: list[Record], power: float) -> tuple[np.ndarray, torch.Tensor]:
    y = np.concatenate([row.labels[row.score] for row in rows]); counts = np.bincount(y, minlength=4).astype(float)
    value = np.power(counts.max() / np.maximum(counts, 1), power); value /= value.mean()
    return counts, torch.tensor(value, dtype=torch.float32)


def emit(model: BeatCNN, rows: list[Record], device: torch.device) -> list[dict[str, object]]:
    output = []; model.eval()
    with torch.inference_mode():
        for row in rows:
            chunks = []
            for start in range(0, len(row.labels), 2048):
                chunks.append(model(torch.from_numpy(row.wave[start:start + 2048]).to(device), torch.from_numpy(row.rr[start:start + 2048]).to(device)).cpu().numpy())
            output.append({"record_id": row.record_id, "logits": np.concatenate(chunks).astype(np.float32), "rr": row.rr, "labels": row.labels, "score": row.score})
    return output


def train_cnn(a: argparse.Namespace) -> None:
    seed_all(a.seed); device = torch.device(a.device); rows = load4(a.manifest_dir, "train_plus_val", a.target_fs, a.lookback_ms, a.delay_ms)
    x = np.concatenate([r.wave[r.score] for r in rows]); rr = np.concatenate([r.rr[r.score] for r in rows]); y = np.concatenate([r.labels[r.score] for r in rows]); counts, w = class_weights(rows, a.power)
    model = BeatCNN(x.shape[1], 8, n_classes=4).to(device); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay); loss_fn = nn.CrossEntropyLoss(weight=w.to(device))
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(rr), torch.from_numpy(y)), batch_size=a.batch_size, shuffle=True, pin_memory=True); history=[]
    for epoch in range(1, a.epochs + 1):
        model.train(); losses=[]
        for xb, rb, yb in loader:
            opt.zero_grad(set_to_none=True); loss=loss_fn(model(xb.to(device), rb.to(device)), yb.to(device)); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 3.0); opt.step(); losses.append(float(loss.detach().cpu()))
        history.append({"epoch":epoch,"loss":float(np.mean(losses))}); print(json.dumps(history[-1]), flush=True)
    out=Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model":model.state_dict(),"channels":int(x.shape[1]),"n_classes":4,"class_names":NAMES,"class_counts":counts.tolist(),"args":vars(a),"protocol":{"strict_Ri_plus_delay_online":True,"classes":"AAMI N/S/V/F; Q/unsupported history-only","future_waveform":False,"future_rpeak_or_post_rr":False,"future_label":False}},out/"fourclass_cnn.pt")
    (out/"train_history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")


def train_gru(a: argparse.Namespace) -> None:
    seed_all(a.seed); device=torch.device(a.device); ck=torch.load(a.backbone,map_location="cpu",weights_only=False); base=BeatCNN(int(ck["channels"]),8,n_classes=4).to(device); base.load_state_dict(ck["model"]); base.eval()
    rows=emit(base,load4(a.manifest_dir,"train_plus_val",a.target_fs,a.lookback_ms,a.delay_ms),device); pseudo=[Record(str(r["record_id"]),np.empty((0,)),r["rr"],r["labels"],r["score"]) for r in rows]; counts,w=class_weights(pseudo,a.power)
    model=LogitGRU(8,a.hidden,a.max_correction,n_classes=4).to(device); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay); loss_fn=nn.CrossEntropyLoss(weight=w.to(device)); history=[]
    for epoch in range(1,a.epochs+1):
        model.train(); order=list(range(len(rows))); random.Random(a.seed+epoch).shuffle(order); losses=[]
        for index in order:
            row=rows[index]; state=None; terms=[]; labels=torch.from_numpy(row["labels"]).to(device); keep=torch.from_numpy(row["score"]).to(device); opt.zero_grad(set_to_none=True); n=max(1,(len(labels)+a.chunk-1)//a.chunk)
            for start in range(0,len(labels),a.chunk):
                logits,state=model.forward_chunk(torch.from_numpy(row["logits"][start:start+a.chunk]).to(device),torch.from_numpy(row["rr"][start:start+a.chunk]).to(device),state); active=keep[start:start+a.chunk]
                if bool(active.any()): terms.append(loss_fn(logits[active],labels[start:start+a.chunk][active])/n)
                state=state.detach()
            if terms:
                loss=torch.stack(terms).sum(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),3.0); opt.step(); losses.append(float(loss.detach().cpu()))
        history.append({"epoch":epoch,"loss":float(np.mean(losses))}); print(json.dumps(history[-1]),flush=True)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);torch.save({"model":model.state_dict(),"backbone":str(Path(a.backbone).resolve()),"n_classes":4,"class_names":NAMES,"class_counts":counts.tolist(),"args":vars(a),"protocol":{"strict_Ri_plus_delay_online":True,"classes":"AAMI N/S/V/F; Q/unsupported history-only","state_reset":"record boundary","future_waveform":False,"future_rpeak_or_post_rr":False,"future_label":False}},out/"fourclass_gru.pt");(out/"train_history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")


def evaluate(a: argparse.Namespace) -> None:
    device=torch.device(a.device); ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False); is_gru=a.mode=="m3"; base_path=ck["backbone"] if is_gru else a.checkpoint; b=torch.load(base_path,map_location="cpu",weights_only=False); base=BeatCNN(int(b["channels"]),8,n_classes=4).to(device);base.load_state_dict(b["model"]);base.eval(); model=None
    if is_gru: model=LogitGRU(8,ck["args"]["hidden"],ck["args"]["max_correction"],n_classes=4).to(device);model.load_state_dict(ck["model"]);model.eval()
    rows=emit(base,load4(a.manifest_dir,"test",a.target_fs,a.lookback_ms,a.delay_ms),device); pred=[]
    with torch.inference_mode():
        for row in rows:
            if model is None: z=row["logits"]
            else:
                state=None; parts=[]
                for start in range(0,len(row["labels"]),a.chunk):
                    part,state=model.forward_chunk(torch.from_numpy(row["logits"][start:start+a.chunk]).to(device),torch.from_numpy(row["rr"][start:start+a.chunk]).to(device),state);parts.append(part.cpu().numpy());state=state.detach()
                z=np.concatenate(parts)
            keep=row["score"];pred.append({"record_id":row["record_id"],"logits":z[keep],"labels":row["labels"][keep],"predictions":z.argmax(1)[keep]})
    y=np.concatenate([x["labels"] for x in pred]);p=np.concatenate([x["predictions"] for x in pred]);z=np.concatenate([x["logits"] for x in pred]);offs=np.cumsum([0]+[len(x["labels"]) for x in pred]);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/"test_predictions.npz",logits=z,labels=y,predictions=p,record_ids=np.asarray([x["record_id"] for x in pred]),record_offsets=offs);payload={"metrics":metrics(y,p),"record_metrics":{x["record_id"]:metrics(x["labels"],x["predictions"]) for x in pred},"record_ids":[x["record_id"] for x in pred],"n_target_records":len(pred),"mode":a.mode,"protocol":ck["protocol"]};(out/"test_metrics.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(payload,ensure_ascii=False,indent=2))


def main() -> None:
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="command",required=True)
    for name in ("train-cnn","train-gru"):
        q=s.add_parser(name);q.add_argument("--manifest-dir",type=Path,required=True);q.add_argument("--out",required=True);q.add_argument("--seed",type=int,required=True);q.add_argument("--device",default="cuda");q.add_argument("--target-fs",type=int,default=250);q.add_argument("--lookback-ms",type=int,default=350);q.add_argument("--delay-ms",type=int,default=200);q.add_argument("--epochs",type=int,default=15);q.add_argument("--lr",type=float,default=5e-4 if name=="train-cnn" else 3e-4);q.add_argument("--weight-decay",type=float,default=1e-4);q.add_argument("--power",type=float,default=.6 if name=="train-cnn" else .4);q.add_argument("--batch-size",type=int,default=512);q.add_argument("--hidden",type=int,default=128);q.add_argument("--max-correction",type=float,default=.75);q.add_argument("--chunk",type=int,default=8192)
        if name=="train-gru": q.add_argument("--backbone",required=True)
    q=s.add_parser("evaluate");q.add_argument("--mode",choices=("m1","m3"),required=True);q.add_argument("--manifest-dir",type=Path,required=True);q.add_argument("--checkpoint",required=True);q.add_argument("--out",required=True);q.add_argument("--device",default="cuda");q.add_argument("--target-fs",type=int,default=250);q.add_argument("--lookback-ms",type=int,default=350);q.add_argument("--delay-ms",type=int,default=200);q.add_argument("--chunk",type=int,default=8192)
    a=p.parse_args(); {"train-cnn":train_cnn,"train-gru":train_gru,"evaluate":evaluate}[a.command](a)

if __name__ == "__main__": main()
