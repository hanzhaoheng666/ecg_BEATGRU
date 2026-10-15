#!/usr/bin/env python3
"""Select exactly one predeclared candidate from DS1 validation folds."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--candidates',required=True); p.add_argument('--runs',required=True); p.add_argument('--out',required=True); p.add_argument('--folds',type=int,default=6); a=p.parse_args()
    cfg=json.loads(Path(a.candidates).read_text(encoding='utf-8')); rows=[]
    for c in cfg['candidates']:
        metrics=[]
        for fold in range(a.folds):
            path=Path(a.runs)/c['name']/f'fold{fold}'/'validation'/'test_metrics.json'
            d=json.loads(path.read_text(encoding='utf-8')); metrics.append(d['metrics'])
        mean={key:float(np.mean([m[key] for m in metrics])) for key in ('macro_f1','sv_f1','N_f1','S_f1','V_f1','accuracy')}
        std={key:float(np.std([m[key] for m in metrics],ddof=1)) for key in ('macro_f1','sv_f1','S_f1','V_f1')}
        rows.append({'candidate':c,'mean':mean,'std':std})
    best=max(rows,key=lambda x:(x['mean']['macro_f1'],x['mean']['sv_f1']))
    out={'protocol':'DS1-only six-fold candidate selection; DS2 not loaded', 'selection_metric':cfg['selection_metric'], 'candidates':rows,'selected':best}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
