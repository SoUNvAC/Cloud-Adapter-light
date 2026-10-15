"""Frozen-threshold operating summaries; group bootstrap, no evaluation scan."""
import json
from pathlib import Path
import numpy as np
from phase120_metrics import ap,table
SETTINGS=('R_05','R_cal','C100_05','C100_cal')
PAIRS=(('C100_cal','R_cal'),('R_cal','R_05'),('C100_cal','C100_05'))
def interval(values):
    a=np.asarray([v for v in values if v is not None],np.float64)
    out=dict(n=len(a),mean=float(a.mean()) if len(a) else None,CI95=None,U95=None,L95=None)
    if len(a)>=2:
        ix=np.random.default_rng(122015).integers(len(a),size=(10000,len(a)));means=a[ix].mean(1)
        slow=np.array([sum(float(a[j]) for j in row)/len(a) for row in ix]);assert np.max(np.abs(means-slow))<1e-10
        out.update(CI95=np.quantile(means,[.025,.975]).tolist(),U95=float(np.quantile(means,.95)),L95=float(np.quantile(means,.05)))
    return out
def operating(y,s,t):
    y=np.asarray(y,np.uint8);s=np.asarray(s,np.float64);pred=s>=np.float64(t)
    p=int((y==2).sum());n=int((y!=2).sum());tp=int((pred&(y==2)).sum());fp=int((pred&(y!=2)).sum())
    out=dict(positive=p,negative=n,TP=tp,FP=fp,recall=tp/p if p else None,precision=tp/(tp+fp) if tp+fp else None,
        FPR=fp/n if n else None)
    for name,kind in [('Surface',0),('Cloud',1)]:
        nn=int((y==kind).sum());ff=int((pred&(y==kind)).sum())
        out.update({name+'_negative':nn,name+'_FP':ff,name+'_FPR':ff/nn if nn else None})
    return out
def evaluate(groups,pool,thresholds,outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True);per=[];aps={};state={};summaries=[];comparison=[];group_deltas=[]
    for gid,g in sorted(groups.items()):
        y=np.asarray(g['truth'],np.uint8);assert set(np.unique(y))<=set((0,1,2))
        aps[gid]={p:ap(y==2,g[p]) for p in ('R','C')}
        for setting in SETTINGS:
            p='C' if setting.startswith('C100') else 'R';t=thresholds[p] if setting.endswith('_cal') else .5
            row=dict(pool=pool,group_id=gid,products=g['products'],setting=setting,threshold=float(t),AP=aps[gid][p],**operating(y,g[p],t))
            per.append(row);state[gid,setting]=row
    y=np.concatenate([g['truth'] for _,g in sorted(groups.items())]);pooledAP={}
    for p in ('R','C'):
        s=np.concatenate([g[p] for _,g in sorted(groups.items())]);pooledAP[p]=ap(y==2,s);del s
    for setting in SETTINGS:
        p='C' if setting.startswith('C100') else 'R';t=thresholds[p] if setting.endswith('_cal') else .5
        rr=[r for r in per if r['setting']==setting];summary=dict(pool=pool,setting=setting,threshold=float(t),all_groups=len(rr),
            AP_groups=sum(r['AP'] is not None for r in rr),recall_groups=sum(r['recall'] is not None for r in rr),
            negative_groups=sum(r['FPR'] is not None for r in rr),
            macro_AP_pp=float(np.mean([r['AP'] for r in rr if r['AP'] is not None]))*100,
            pooled_AP_pp=pooledAP[p]*100 if pooledAP[p] is not None else None,
            zero_recall_positive_groups=sum(r['recall']==0 for r in rr if r['positive']),
            over_1pct_FP_groups=sum(r['FPR']>.01 for r in rr if r['FPR'] is not None),
            max_group_FPR_pp=max(r['FPR'] for r in rr if r['FPR'] is not None)*100)
        for key in ('recall','precision','FPR','Surface_FPR','Cloud_FPR'):
            ci=interval([r[key]*100 if r[key] is not None else None for r in rr]);summary.update({f'macro_{key}_pp':ci['mean'],f'{key}_groups':ci['n'],
                f'{key}_CI95_pp':ci['CI95'],f'{key}_U95_pp':ci['U95']})
        s=np.concatenate([g[p] for _,g in sorted(groups.items())]);summary.update({'pooled_'+k:v for k,v in operating(y,s,t).items()});del s
        summaries.append(summary)
    for gid in sorted(groups):
        for a,b in PAIRS:
            ra,rb=state[gid,a],state[gid,b];row=dict(pool=pool,group_id=gid,comparison=a+'-'+b)
            for key in ('AP','recall','precision','FPR','Surface_FPR','Cloud_FPR'):
                row[key+'_delta_pp']=(ra[key]-rb[key])*100 if ra[key] is not None and rb[key] is not None else None
            if a.split('_')[0]==b.split('_')[0]:assert row['AP_delta_pp'] in (0,None)
            group_deltas.append(row)
    for a,b in PAIRS:
        name=a+'-'+b;rr=[v for v in group_deltas if v['comparison']==name]
        row=dict(pool=pool,comparison=name,primary=pool=='Catalogue' and a=='C100_cal' and b=='R_cal',resamples=10000,seed=122015)
        for key in ('AP','recall','precision','FPR','Surface_FPR','Cloud_FPR'):
            ci=interval([v[key+'_delta_pp'] for v in rr]);row[key]=ci
            valid=[(v[key+'_delta_pp'],v['group_id']) for v in rr if v[key+'_delta_pp'] is not None]
            row[key+'_worst_group']=min(valid)[1] if valid else None;row[key+'_worst_delta_pp']=min(valid)[0] if valid else None
        comparison.append(row)
    for p in ('R','C100'):
        a,b=[v for v in summaries if v['setting'].startswith(p+'_')]
        assert a['macro_AP_pp']==b['macro_AP_pp'] and a['pooled_AP_pp']==b['pooled_AP_pp']
    table(out/(pool+'_four_workpoints.csv'),summaries);table(out/(pool+'_per_group.csv'),per);table(out/(pool+'_group_differences.csv'),group_deltas)
    with (out/(pool+'_comparisons.json')).open('x',encoding='utf-8') as f:json.dump(dict(rows=comparison,unit='related groups; percentile intervals descriptive'),f,indent=2,allow_nan=False)
    return summaries,comparison
