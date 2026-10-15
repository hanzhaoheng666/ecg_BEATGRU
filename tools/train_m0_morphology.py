#!/usr/bin/env python3
"""Train the paired M0 raw+d1+d2 morphology-only CNN baseline."""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
# Support direct ``python tools/train_m0_morphology.py`` execution.
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from beatgru.data import load_records
from beatgru.models import MorphologyCNN
from beatgru.metrics import metric
from beatgru.train import device_from, record_ids_from_file, seed_all, weights

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=('train','evaluate'),default='train'); p.add_argument('--manifest-dir',required=True); p.add_argument('--out',required=True); p.add_argument('--checkpoint',default=''); p.add_argument('--dataset',default='mit_bih_arrhythmia'); p.add_argument('--target-split',default='test',choices=('train','val','test','all')); p.add_argument('--seed',type=int,default=0); p.add_argument('--epochs',type=int,default=15); p.add_argument('--batch-size',type=int,default=512); p.add_argument('--lr',type=float,default=5e-4); p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--power',type=float,default=.6); p.add_argument('--device',default='cuda'); p.add_argument('--record-ids-file',default=''); p.add_argument('--target-fs',type=int,default=250); p.add_argument('--lookback-ms',type=int,default=350); p.add_argument('--delay-ms',type=int,default=200); p.add_argument('--lead-selector',default='canonical_first')
    a=p.parse_args(); seed_all(a.seed); selected=record_ids_from_file(a.record_ids_file)
    if a.mode == 'evaluate':
        ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False); device=device_from(a.device); model=MorphologyCNN(int(ck.get('channels',3))).to(device); model.load_state_dict(ck['model']); model.eval(); rows=load_records(a.manifest_dir,a.target_split,a.target_fs,a.lookback_ms,a.delay_ms,dataset=a.dataset,lead_selector=a.lead_selector,record_ids=selected); ys=[];ps=[]; rec={}
        with torch.inference_mode():
            for row in rows:
                logits=model(torch.from_numpy(row.wave).to(device)).cpu().numpy(); keep=row.score; yy=row.labels[keep]; pp=logits.argmax(1)[keep]; ys.append(yy);ps.append(pp); rec[row.record_id]=metric(yy,pp)
        y=np.concatenate(ys); pred=np.concatenate(ps); out=Path(a.out);out.mkdir(parents=True,exist_ok=True); np.savez_compressed(out/'test_predictions.npz',labels=y,predictions=pred,record_ids=np.asarray(list(rec)),record_offsets=np.cumsum([0]+[int(v['n_samples']) for v in rec.values()])); payload={'metrics':metric(y,pred),'record_metrics':rec,'mode':'m0','protocol':ck['protocol']}; (out/'test_metrics.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)); print(json.dumps(payload,ensure_ascii=False,indent=2)); return
    rows=load_records(a.manifest_dir,'train_plus_val',a.target_fs,a.lookback_ms,a.delay_ms,record_ids=selected)
    x=np.concatenate([r.wave[r.score] for r in rows]); y=np.concatenate([r.labels[r.score] for r in rows]); counts,w=weights(rows,a.power); device=device_from(a.device); model=MorphologyCNN(x.shape[1]).to(device)
    loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=a.batch_size,shuffle=True,pin_memory=True); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay); loss_fn=nn.CrossEntropyLoss(weight=w.to(device)); history=[]
    for epoch in range(1,a.epochs+1):
        model.train(); losses=[]
        for xb,yb in loader:
            opt.zero_grad(set_to_none=True); loss=loss_fn(model(xb.to(device)),yb.to(device)); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),3.0); opt.step(); losses.append(float(loss.detach().cpu()))
        history.append({'epoch':epoch,'loss':float(np.mean(losses))}); print(json.dumps(history[-1]),flush=True)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'channels':int(x.shape[1]),'class_counts':counts.tolist(),'args':vars(a),'protocol':{'strict_Ri_plus_delay_online':True,'input':'current raw+d1+d2 morphology only','future_waveform':False,'future_rpeak_or_post_rr':False,'future_label':False}},out/'m0_morphology.pt'); (out/'train_history.json').write_text(json.dumps(history,indent=2))
if __name__=='__main__': main()
