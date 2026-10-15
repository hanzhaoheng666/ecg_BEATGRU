#!/usr/bin/env python3
"""Train/evaluate M1 no-recursive-history, M2 sliding-32, and M3 full-history."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
# Support direct execution from the tools directory.
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from torch import nn
from beatgru.data import Record, load_records
from beatgru.metrics import metric
from beatgru.models import BeatCNN, LogitGRU, ShortLogitGRU
from beatgru.train import device_from, emissions, forward_history, record_ids_from_file, seed_all, weights

def load_backbone(path, device):
    ck=torch.load(path,map_location='cpu',weights_only=False); model=BeatCNN(int(ck.get('channels',3)),8).to(device); model.load_state_dict(ck['model']); model.eval(); return model,ck

def get_rows(a, backbone, split, selected, dataset='mit_bih_arrhythmia', lead='canonical_first'):
    rec=load_records(a.manifest_dir,split,a.target_fs,a.lookback_ms,a.delay_ms,dataset=dataset,lead_selector=lead,record_ids=selected)
    return rec, emissions(backbone,rec,next(backbone.parameters()).device)

def train_history(a):
    seed_all(a.seed); device=device_from(a.device); backbone,bck=load_backbone(a.backbone,device); selected=record_ids_from_file(a.record_ids_file); rec,rows=get_rows(a,backbone,'train_plus_val',selected)
    pseudo=[Record(r['record_id'],np.empty((0,)),r['rr'],r['labels'],r['score']) for r in rows]; counts,w=weights(pseudo,a.power)
    model=ShortLogitGRU(8,a.hidden,a.max_correction,32).to(device) if a.mode=='short32' else LogitGRU(8,a.hidden,a.max_correction).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay); loss_fn=nn.CrossEntropyLoss(weight=w.to(device)); hist=[]
    for epoch in range(1,a.epochs+1):
        model.train(); losses=[]; order=list(range(len(rows))); rng=np.random.default_rng(a.seed+epoch); rng.shuffle(order)
        for j in order:
            row=rows[j]; labels=torch.from_numpy(row['labels']).to(device); keep=torch.from_numpy(row['score']).to(device); opt.zero_grad(set_to_none=True)
            if a.mode=='short32':
                logits=model.forward_windows(torch.from_numpy(row['logits']).to(device),torch.from_numpy(row['rr']).to(device)); active=keep; loss=loss_fn(logits[active],labels[active])
            else:
                state=None; terms=[]; n=max(1,(len(row['labels'])+a.chunk-1)//a.chunk)
                for start in range(0,len(row['labels']),a.chunk):
                    logits,state=model.forward_chunk(torch.from_numpy(row['logits'][start:start+a.chunk]).to(device),torch.from_numpy(row['rr'][start:start+a.chunk]).to(device),state); active=keep[start:start+a.chunk]
                    if bool(active.any()): terms.append(loss_fn(logits[active],labels[start:start+a.chunk][active])/n)
                    state=state.detach()
                if not terms: continue
                loss=torch.stack(terms).sum()
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),3.0); opt.step(); losses.append(float(loss.detach().cpu()))
        hist.append({'epoch':epoch,'loss':float(np.mean(losses))}); print(json.dumps(hist[-1]),flush=True)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'backbone':str(Path(a.backbone).resolve()),'args':vars(a),'mode':a.mode,'class_counts':counts.tolist(),'protocol':{'strict_Ri_plus_delay_online':True,'history_mode':a.mode,'state_reset':'record boundary','future_waveform':False,'future_rpeak_or_post_rr':False,'future_label':False}},out/'history.pt'); (out/'train_history.json').write_text(json.dumps(hist,indent=2))

def predict_rows(mode, model, rows, device, chunk):
    out=[]
    with torch.inference_mode():
        for row in rows:
            if mode=='m1': logits=row['logits']
            elif mode=='short32': logits=model.forward_windows(torch.from_numpy(row['logits']).to(device),torch.from_numpy(row['rr']).to(device)).cpu().numpy()
            else:
                parts=[]; state=None
                for start in range(0,len(row['labels']),chunk):
                    z,state=model.forward_chunk(torch.from_numpy(row['logits'][start:start+chunk]).to(device),torch.from_numpy(row['rr'][start:start+chunk]).to(device),state); parts.append(z.cpu().numpy()); state=state.detach()
                logits=np.concatenate(parts)
            keep=row['score']; out.append({'record_id':row['record_id'],'logits':logits[keep],'labels':row['labels'][keep],'predictions':logits.argmax(1)[keep]})
    return out

def write_results(a, mode, rows, pred_rows, out, protocol):
    y=np.concatenate([r['labels'] for r in pred_rows]); p=np.concatenate([r['predictions'] for r in pred_rows]); z=np.concatenate([r['logits'] for r in pred_rows]); offsets=np.cumsum([0]+[len(r['labels']) for r in pred_rows]);
    out.mkdir(parents=True,exist_ok=True); np.savez_compressed(out/'test_predictions.npz',logits=z,labels=y,predictions=p,record_ids=np.asarray([r['record_id'] for r in pred_rows]),record_offsets=offsets)
    record_metrics={r['record_id']:metric(r['labels'],r['predictions']) for r in pred_rows}; payload={'metrics':metric(y,p),'record_metrics':record_metrics,'record_ids':[r['record_id'] for r in pred_rows],'n_target_records':len(pred_rows),'mode':mode,'protocol':protocol}; (out/'test_metrics.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)); return payload

def evaluate_cmd(a):
    device=device_from(a.device); selected=record_ids_from_file(a.record_ids_file)
    if a.mode=='m1':
        backbone,bck=load_backbone(a.backbone,device); rows=get_rows(a,backbone,a.target_split,selected,a.dataset,a.lead_selector)[1]; protocol={'strict_Ri_plus_delay_online':True,'history_mode':'m1_no_recursive_history','future_waveform':False,'future_rpeak_or_post_rr':False,'future_label':False}
    else:
        ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False); backbone,bck=load_backbone(ck['backbone'],device); rows=get_rows(a,backbone,a.target_split,selected,a.dataset,a.lead_selector)[1]; model=ShortLogitGRU(8,a.hidden,a.max_correction,32).to(device) if a.mode=='short32' else LogitGRU(8,a.hidden,a.max_correction).to(device); model.load_state_dict(ck['model']); model.eval(); protocol=ck['protocol']
    payload=write_results(a,a.mode,rows,predict_rows(a.mode,model if a.mode!='m1' else None,rows,device,a.chunk),Path(a.out),protocol); print(json.dumps(payload,ensure_ascii=False,indent=2))

def main():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest='command',required=True)
    t=s.add_parser('train'); t.add_argument('--mode',choices=('short32','full'),required=True); t.add_argument('--manifest-dir',required=True);t.add_argument('--backbone',required=True);t.add_argument('--out',required=True);t.add_argument('--record-ids-file',default='');t.add_argument('--seed',type=int,default=0);t.add_argument('--device',default='cuda');t.add_argument('--epochs',type=int,default=15);t.add_argument('--lr',type=float,default=3e-4);t.add_argument('--weight-decay',type=float,default=1e-4);t.add_argument('--power',type=float,default=.4);t.add_argument('--hidden',type=int,default=128);t.add_argument('--max-correction',type=float,default=.75);t.add_argument('--chunk',type=int,default=8192);t.add_argument('--target-fs',type=int,default=250);t.add_argument('--lookback-ms',type=int,default=350);t.add_argument('--delay-ms',type=int,default=200)
    e=s.add_parser('evaluate'); e.add_argument('--mode',choices=('m1','short32','full'),required=True);e.add_argument('--manifest-dir',required=True);e.add_argument('--backbone',default='');e.add_argument('--checkpoint',default='');e.add_argument('--out',required=True);e.add_argument('--dataset',default='mit_bih_arrhythmia');e.add_argument('--target-split',default='test',choices=('train','val','test','all'));e.add_argument('--record-ids-file',default='');e.add_argument('--device',default='cuda');e.add_argument('--hidden',type=int,default=128);e.add_argument('--max-correction',type=float,default=.75);e.add_argument('--chunk',type=int,default=8192);e.add_argument('--target-fs',type=int,default=250);e.add_argument('--lookback-ms',type=int,default=350);e.add_argument('--delay-ms',type=int,default=200);e.add_argument('--lead-selector',default='canonical_first')
    a=p.parse_args(); train_history(a) if a.command=='train' else evaluate_cmd(a)
if __name__=='__main__': main()
