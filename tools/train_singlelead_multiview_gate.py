#!/usr/bin/env python3
"""DS1-only training/evaluation for a bounded frozen two-view fusion gate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beatgru.data import load_records
from beatgru.metrics import metric
from beatgru.models import BeatCNN, CausalViewGate, EmbeddingGRU, LogitGRU
from beatgru.train import device_from, emissions, evaluate, seed_all


def ids(path: str) -> list[str] | None:
    return [x.strip() for x in Path(path).read_text().splitlines() if x.strip()] if path else None


def frozen_streams(args: argparse.Namespace, model: LogitGRU | EmbeddingGRU, base: BeatCNN,
                   selected: list[str] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    common = dict(manifest_dir=args.manifest_dir, split=args.target_split, target_fs=args.target_fs,
                  lookback_ms=args.lookback_ms, delay_ms=args.delay_ms, dataset=args.dataset, record_ids=selected)
    first = load_records(lead_selector=args.first_selector, **common)
    second = load_records(lead_selector=args.second_selector, **common)
    if [(r.record_id, len(r.labels)) for r in first] != [(r.record_id, len(r.labels)) for r in second]:
        raise ValueError("two views have different record/event sequences")
    if any(not (np.array_equal(a.labels, b.labels) and np.array_equal(a.score, b.score) and np.array_equal(a.rr, b.rr))
           for a, b in zip(first, second)):
        raise ValueError("two views disagree on labels, scoring, or causal RR")
    device = next(base.parameters()).device
    hinput = args.history_input == "embedding"
    rows_a, rows_b = emissions(base, first, device, include_embedding=hinput), emissions(base, second, device, include_embedding=hinput)
    _, a, labels, _ = evaluate(model, rows_a, device, args.chunk)
    _, b, labels_b, _ = evaluate(model, rows_b, device, args.chunk)
    if not np.array_equal(labels, labels_b):
        raise ValueError("two views disagree after score masking")
    rr = np.concatenate([row.rr[row.score] for row in first]).astype(np.float32)
    return a, b, rr, labels


def load_frozen(args: argparse.Namespace) -> tuple[BeatCNN, LogitGRU | EmbeddingGRU]:
    device = device_from(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone = torch.load(args.backbone or ckpt["backbone"], map_location="cpu", weights_only=False)
    if int(backbone.get("channels", 3)) != 3:
        raise ValueError("gate requires a single-lead checkpoint")
    base = BeatCNN(3, 8).to(device); base.load_state_dict(backbone["model"]); base.eval()
    args.target_fs, args.lookback_ms, args.delay_ms = (ckpt["args"][k] for k in ("target_fs", "lookback_ms", "delay_ms"))
    args.chunk = ckpt["args"]["chunk"]; args.history_input = ckpt["args"].get("history_input", "logits")
    model = (EmbeddingGRU(160, 8, ckpt["args"]["hidden"], ckpt["args"]["max_correction"])
             if args.history_input == "embedding" else LogitGRU(8, ckpt["args"]["hidden"], ckpt["args"]["max_correction"])).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    return base, model


def train(args: argparse.Namespace) -> None:
    seed_all(args.seed); base, history = load_frozen(args)
    a, b, rr, labels = frozen_streams(args, history, base, ids(args.record_ids_file))
    counts = np.bincount(labels, minlength=3).astype(float)
    w = np.power(counts.max() / np.maximum(counts, 1), args.power); w /= w.mean()
    device = next(base.parameters()).device; gate = CausalViewGate(8, args.hidden, args.max_correction).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(a), torch.from_numpy(b), torch.from_numpy(rr), torch.from_numpy(labels)), batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=args.weight_decay); loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=device))
    hist=[]
    for epoch in range(1,args.epochs+1):
        gate.train(); losses=[]
        for aa,bb,rrb,y in loader:
            opt.zero_grad(set_to_none=True); loss=loss_fn(gate(aa.to(device),bb.to(device),rrb.to(device)),y.to(device)); loss.backward(); nn.utils.clip_grad_norm_(gate.parameters(),3.0); opt.step(); losses.append(float(loss.detach().cpu()))
        hist.append({'epoch':epoch,'loss':float(np.mean(losses))}); print(json.dumps(hist[-1]),flush=True)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    torch.save({'model':gate.state_dict(),'checkpoint':str(Path(args.checkpoint).resolve()),'backbone':str(Path(args.backbone or '').resolve()) if args.backbone else '', 'args':vars(args),
                'protocol':{'frozen_single_lead_source_model':True,'gate_inputs':'current two-view logits + causal RR','future_waveform':False,'future_rpeak_or_post_rr':False,'future_label':False,'target_label_or_statistic_used_for_fusion':False}},out/'view_gate.pt')
    (out/'train_history.json').write_text(json.dumps(hist,indent=2),encoding='utf-8')


def evaluate_gate(args: argparse.Namespace) -> None:
    gate_ckpt=torch.load(args.gate,map_location='cpu',weights_only=False); saved=gate_ckpt['args']
    for k in ('target_fs','lookback_ms','delay_ms','hidden','max_correction'):
        if getattr(args,k)!=saved[k]: raise ValueError(f'gate configuration mismatch: {k}')
    args.checkpoint=gate_ckpt['checkpoint']; args.backbone=gate_ckpt['backbone']; args.history_input='logits'
    base,history=load_frozen(args); a,b,rr,labels=frozen_streams(args,history,base,ids(args.record_ids_file))
    gate=CausalViewGate(8,args.hidden,args.max_correction).to(next(base.parameters()).device); gate.load_state_dict(gate_ckpt['model']); gate.eval()
    with torch.inference_mode(): logits=gate(torch.from_numpy(a).to(next(base.parameters()).device),torch.from_numpy(b).to(next(base.parameters()).device),torch.from_numpy(rr).to(next(base.parameters()).device)).cpu().numpy()
    pred=logits.argmax(1); out=Path(args.out); out.mkdir(parents=True,exist_ok=True); np.savez_compressed(out/'test_predictions.npz',logits=logits,labels=labels,predictions=pred)
    payload={'metrics':metric(labels,pred),'target_dataset':args.dataset,'target_split':args.target_split,'fusion':'DS1-trained bounded current-logit + causal-RR gate','first_selector':args.first_selector,'second_selector':args.second_selector,'protocol':gate_ckpt['protocol']}
    (out/'test_metrics.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(payload,ensure_ascii=False,indent=2))


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest='mode',required=True)
    def common(q: argparse.ArgumentParser) -> None:
        q.add_argument('--manifest-dir',required=True);q.add_argument('--out',required=True);q.add_argument('--dataset',default='mit_bih_arrhythmia');q.add_argument('--target-split',default='train',choices=('train','val','test','all'));q.add_argument('--record-ids-file',default='');q.add_argument('--first-selector',default='source_matched_ii');q.add_argument('--second-selector',default='source_matched_precordial');q.add_argument('--device',default='cuda');q.add_argument('--target-fs',type=int,default=250);q.add_argument('--lookback-ms',type=int,default=350);q.add_argument('--delay-ms',type=int,default=200);q.add_argument('--hidden',type=int,default=32);q.add_argument('--max-correction',type=float,default=.25)
    t=s.add_parser('train');common(t);t.add_argument('--checkpoint',required=True);t.add_argument('--backbone',default='');t.add_argument('--seed',type=int,default=0);t.add_argument('--epochs',type=int,default=20);t.add_argument('--batch-size',type=int,default=1024);t.add_argument('--lr',type=float,default=3e-4);t.add_argument('--weight-decay',type=float,default=1e-4);t.add_argument('--power',type=float,default=.4)
    e=s.add_parser('evaluate');common(e);e.add_argument('--gate',required=True)
    return p

if __name__=='__main__':
    a=parser().parse_args(); train(a) if a.mode=='train' else evaluate_gate(a)
