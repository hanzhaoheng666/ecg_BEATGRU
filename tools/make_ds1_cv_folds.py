#!/usr/bin/env python3
"""Create deterministic record-disjoint DS1 folds without loading DS2."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--manifest-dir',required=True); p.add_argument('--out',required=True); p.add_argument('--folds',type=int,default=6); a=p.parse_args()
    counts={}
    with (Path(a.manifest_dir)/'mit_bih_arrhythmia.csv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            if r['split']!='train' or int(r['is_scored'])!=1: continue
            y=int(r.get('label', {'N':0,'S':1,'V':2}.get(r.get('aami_label',''),-1)))
            if 0<=y<3: counts.setdefault(r['record_id'],np.zeros(3,dtype=int))[y]+=1
    if len(counts)<a.folds: raise ValueError('too few DS1 records')
    total=np.sum(list(counts.values()),axis=0).astype(float); target=total/a.folds; current=np.zeros((a.folds,3)); groups=[[] for _ in range(a.folds)]
    for rid in sorted(counts,key=lambda k:(-counts[k][1],-counts[k][2],-counts[k][0],k)):
        options=[]
        for fold in range(a.folds):
            probe=current.copy(); probe[fold]+=counts[rid]
            options.append((float(np.square((probe-target)/np.maximum(target,1)).sum()),len(groups[fold]),fold))
        fold=min(options)[2]; groups[fold].append(rid); current[fold]+=counts[rid]
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); all_ids=set(counts); payload={'protocol':'DS1-only record-disjoint six-fold validation','folds':[]}
    for fold,hold in enumerate(groups):
        hold=sorted(hold); train=sorted(all_ids-set(hold))
        (out/f'fold{fold}_train.txt').write_text('\n'.join(train)+'\n',encoding='utf-8'); (out/f'fold{fold}_val.txt').write_text('\n'.join(hold)+'\n',encoding='utf-8')
        payload['folds'].append({'fold':fold,'train_records':train,'val_records':hold,'val_counts':current[fold].astype(int).tolist()})
    (out/'folds.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
