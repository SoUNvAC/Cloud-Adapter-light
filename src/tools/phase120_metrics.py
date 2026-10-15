"""Fixed related-group AP/operating summaries, independently tied and replayed."""
import csv
import json
from pathlib import Path
import numpy as np
from phase89_cpu_ap import average_precision_score as bins_ap


def ap(y,s):
    y=np.asarray(y,bool);s=np.asarray(s,np.float32)
    assert len(y)==len(s) and np.isfinite(s).all()
    if not y.any() or not (~y).any():return None
    order=np.argsort(-s,kind='stable');sorted_s=s[order];positive=y[order]
    ends=np.flatnonzero(np.r_[sorted_s[1:]!=sorted_s[:-1],True])
    tp=np.cumsum(positive,dtype=np.int64)[ends]
    value=float(np.sum(np.diff(np.r_[0,tp])*tp/(ends+1)/y.sum()))
    assert abs(value-bins_ap(y,s))<1e-10
    return value


def operating(y,s):
    y=np.asarray(y,bool);pred=np.asarray(s)>=.5
    p,n=int(y.sum()),int((~y).sum());tp,fp=int((y&pred).sum()),int((~y&pred).sum())
    return {'positive':p,'negative':n,'TP':tp,'FP':fp,'recall':tp/p if p else None,
            'precision':tp/(tp+fp) if tp+fp else None,'negative_FP_fraction':fp/n if n else None}


def table(path,rows):
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def evaluate(groups,pool):
    """groups: gid -> metadata, truth bool, E/R/N confidence on frozen support."""
    result=[];pooled={p:[] for p in ('E','R','N','C','D')};labels=[]
    for gid,g in sorted(groups.items()):
        y=g['truth'];pos=int(y.sum());neg=len(y)-pos
        row={'pool':pool,'group_id':gid,'products':g['products'],'positive':pos,'negative':neg,
             'sparse_support':bool(pos>0 and (pos<5 or neg<20))}
        labels.append(y)
        for p in pooled:
            row[p+'_AP']=ap(y,g[p]);pooled[p].append(g[p])
            row.update({p+'_'+k:v for k,v in operating(y,g[p]).items()})
        for a,b in (('D','R'),('D','C'),('C','R')):
            row[a+'_minus_'+b+'_AP_pp']=(row[a+'_AP']-row[b+'_AP'])*100 if row[a+'_AP'] is not None else None
        result.append(row)
    summaries=[];truth=np.concatenate(labels)
    for p in pooled:
        s=np.concatenate(pooled[p]);a=ap(truth,s);values=[r[p+'_AP'] for r in result if r[p+'_AP'] is not None]
        recall=[r[p+'_recall'] for r in result if r[p+'_recall'] is not None]
        fp=[r[p+'_negative_FP_fraction'] for r in result if r[p+'_negative_FP_fraction'] is not None]
        summaries.append({'pool':pool,'setting':p,'all_groups':len(result),'AP_groups':len(values),
                          'macro_AP_pp':float(np.mean(values))*100 if values else None,'pooled_AP_pp':a*100 if a is not None else None,
                          'macro_recall_pp':float(np.mean(recall))*100 if recall else None,
                          'macro_negative_FP_pp':float(np.mean(fp))*100 if fp else None,
                          'worst_group_negative_FP_pp':max(fp)*100 if fp else None,
                          **{'pooled_'+k:v for k,v in operating(truth,s).items()}})
        del s
    comparisons=[]
    for a,b in (('D','R'),('D','C'),('C','R')):
        key=a+'_minus_'+b+'_AP_pp';delta=np.array([r[key] for r in result if r[key] is not None],float)
        c={'pool':pool,'comparison':a+'-'+b,'n_related_groups':len(delta),
           'mean_delta_AP_pp':float(delta.mean()) if len(delta) else None,'CI95_pp':None,
           'improved_gt1pp':int((delta>1).sum()),'stable_within1pp':int((np.abs(delta)<=1).sum()),
           'declined_lt_minus1pp':int((delta< -1).sum()),'resamples':10000,'seed':120015,
           'no_significance_or_minimum_effect_pass_gate':True}
        if len(delta)>=2:
            idx=np.random.default_rng(120015).integers(len(delta),size=(10000,len(delta)))
            draws=delta[idx].mean(1);scalar=np.array([sum(float(delta[i]) for i in row)/len(delta) for row in idx])
            assert np.max(np.abs(draws-scalar))<1e-10
            c['CI95_pp']=np.quantile(draws,[.025,.975]).tolist()
        comparisons.append(c)
    return result,summaries,comparisons


def deliver(directory,pool,groups):
    out=Path(directory);out.mkdir(exist_ok=True,parents=True)
    per,summary,comparison=evaluate(groups,pool)
    table(out/(pool+'_per_group.csv'),per);table(out/(pool+'_summary.csv'),summary)
    with (out/(pool+'_comparisons.json')).open('x',encoding='utf-8') as f:
        json.dump({'rows':comparison,'AP_group_unit':'related group; products combined first'},f,indent=2,allow_nan=False);f.write('\n')
    return per,summary,comparison
