#!/usr/bin/env python3
"""Summarize M0-M3 per-seed/per-record metrics and paired bootstrap deltas."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

METRICS=('macro_f1','sv_f1','balanced_accuracy','accuracy','N_f1','S_f1','V_f1')
def read(root,seed,method,target):
 p=Path(root)/f'seed{seed}'/method/target/'test_metrics.json'; return json.loads(p.read_text())
def mean_rows(rows):
 return {k:{'mean':float(np.mean([r['metrics'][k] for r in rows])),'std':float(np.std([r['metrics'][k] for r in rows],ddof=1))} for k in METRICS}
def bootstrap(x,n=10000,seed=17):
 rng=np.random.default_rng(seed); x=np.asarray(x,float); means=np.mean(x[rng.integers(0,len(x),(n,len(x)))],axis=1); return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]
def paired(a,b):
 ids=sorted(set(a['record_metrics']) & set(b['record_metrics'])); out={}
 for k in METRICS:
  d=np.array([b['record_metrics'][i][k]-a['record_metrics'][i][k] for i in ids]); out[k]={'mean_delta':float(d.mean()),'bootstrap95':bootstrap(d),'n_improved':int(np.sum(d>0)),'n_records':len(d),'deltas_by_record':{i:float(v) for i,v in zip(ids,d)}}
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--baseline-root',default='');p.add_argument('--external',action='store_true');a=p.parse_args(); out=Path(a.out);out.mkdir(parents=True,exist_ok=True); root=Path(a.root); seeds=[0,1,2]; methods=['M0','M1','M2','M3']; targets=['DS2_FINAL_ONCE']
 rows=[]; raw={}
 for target in targets:
  for method in methods:
   vals=[]
   for seed in seeds:
    d=read(root,seed,method,target); raw[f'seed{seed}_{method}_{target}']=d; vals.append(d)
    rows.append({'seed':seed,'method':method,'target':target,**{k:d['metrics'][k] for k in METRICS}})
   raw[f'{method}_{target}_summary']=mean_rows(vals)
 for target,methods_ext in [('INCART_II_ALL',['M1','M3']),('SVDB_ECG1_ALL',['M1','M3'])]:
  for method in methods_ext:
   vals=[]
   for seed in seeds:
    d=read(root,seed,method,target); raw[f'seed{seed}_{method}_{target}']=d; vals.append(d); rows.append({'seed':seed,'method':method,'target':target,**{k:d['metrics'][k] for k in METRICS}})
   raw[f'{method}_{target}_summary']=mean_rows(vals)
 for r in rows:
  pass
 (out/'per_seed_metrics.csv').open('w',newline='').write('')
 with (out/'per_seed_metrics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['seed','method','target',*METRICS]);w.writeheader();w.writerows(rows)
 raw['paired_DS2_M1_to_M3']=[]
 for seed in seeds:
  raw['paired_DS2_M1_to_M3'].append({'seed':seed,**paired(raw[f'seed{seed}_M1_DS2_FINAL_ONCE'],raw[f'seed{seed}_M3_DS2_FINAL_ONCE'])})
 # Pooled three-seed paired record deltas are reported separately from per-seed
 # CIs to avoid treating beats as independent samples.
 raw['paired_DS2_M1_to_M3_record_level_by_seed']={str(seed):paired(raw[f'seed{seed}_M1_DS2_FINAL_ONCE'],raw[f'seed{seed}_M3_DS2_FINAL_ONCE']) for seed in seeds}
 raw['paired_external_M1_to_M3_record_level_by_seed']={}
 for target in ('INCART_II_ALL','SVDB_ECG1_ALL'):
  raw['paired_external_M1_to_M3_record_level_by_seed'][target]={str(seed):paired(raw[f'seed{seed}_M1_{target}'],raw[f'seed{seed}_M3_{target}']) for seed in seeds}
 (out/'summary.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2))
 lines=['# M0–M3 三 seed 严格因果消融汇总','', '| 目标/方法 | Macro-F1 | S-F1 | V-F1 | SV-F1 | Balanced Acc | Accuracy |','|---|---:|---:|---:|---:|---:|---:|']
 for target in ['DS2_FINAL_ONCE','INCART_II_ALL','SVDB_ECG1_ALL']:
  for method in (methods if target=='DS2_FINAL_ONCE' else ['M1','M3']):
   key=f'{method}_{target}_summary';
   if key not in raw: continue
   s=raw[key]; lines.append(f"| {target}/{method} | {s['macro_f1']['mean']:.4f}±{s['macro_f1']['std']:.4f} | {s['S_f1']['mean']:.4f}±{s['S_f1']['std']:.4f} | {s['V_f1']['mean']:.4f}±{s['V_f1']['std']:.4f} | {s['sv_f1']['mean']:.4f}±{s['sv_f1']['std']:.4f} | {s['balanced_accuracy']['mean']:.4f}±{s['balanced_accuracy']['std']:.4f} | {s['accuracy']['mean']:.4f}±{s['accuracy']['std']:.4f} |")
 lines += ['', 'DS2 的 M1→M3 配对记录差值（seed 0 记录级审计；三个 seed 的逐 seed 数据见 summary.json）：']
 for seed in seeds:
  lines.append(f'\nSeed {seed}:')
  for k in ('macro_f1','S_f1','V_f1'):
   q=raw['paired_DS2_M1_to_M3_record_level_by_seed'][str(seed)][k]; lines.append(f"- {k}: 平均差 {q['mean_delta']:.4f}，bootstrap 95% CI [{q['bootstrap95'][0]:.4f}, {q['bootstrap95'][1]:.4f}]，改善记录 {q['n_improved']}/{q['n_records']}")
 (out/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__': main()
