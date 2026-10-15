"""Seven policies; related groups, never training seeds, are resampling units."""
import json
from pathlib import Path
import numpy as np
from phase120_metrics import ap,operating,table

POLICIES=('E','R','N','C12001','D12001','C12002','D12002')
PAIRS=tuple((a,b) for seed in (12001,12002) for a,b in
            ((f'C{seed}','R'),(f'D{seed}',f'C{seed}'),(f'D{seed}','R')))

def contrast(pool,name,delta):
    d=np.asarray(delta,float)
    result=dict(pool=pool,comparison=name,n_related_groups=len(d),mean_delta_AP_pp=float(d.mean()) if len(d) else None,
        CI95_pp=None,improved_gt1pp=int((d>1).sum()),stable_within1pp=int((np.abs(d)<=1).sum()),
        declined_lt_minus1pp=int((d< -1).sum()),resamples=10000,seed=121015,
        conditional_on_two_observed_seeds='two_seed' in name,no_significance_pass_gate=True)
    if len(d)>=2:
        idx=np.random.default_rng(121015).integers(len(d),size=(10000,len(d)))
        draws=d[idx].mean(1)
        independent=np.array([sum(float(d[i]) for i in row)/len(d) for row in idx])
        assert np.max(np.abs(draws-independent))<1e-10
        result['CI95_pp']=np.quantile(draws,[.025,.975]).tolist()
    return result

def oracle(groups,policy,pool):
    """Float32 negative ties locate a boundary; float64 nextafter excludes it.

    Binary search is over the ordered positive float32 bit lattice, not an
    approximate probability grid. Feasibility uses integer group FP counts.
    Independent direct float64 comparisons certify both adjacent boundaries.
    """
    rows=sorted(groups.items());neg=[]
    for _,g in rows:
        s=np.asarray(g[policy],np.float32);y=np.asarray(g['truth'],bool)
        assert np.isfinite(s).all() and ((s>=0)&(s<=1)).all()
        neg.append(np.sort(s[~y]))
    den=np.array([len(v) for v in neg],np.int64);supported=den>0
    if not supported.any():return dict(pool=pool,setting=policy,threshold=None,status='no_negative_support'),[]
    def bits(v):return int(np.asarray(np.float32(v)).view(np.uint32))
    def value(v):return np.asarray(np.uint32(v)).view(np.float32).item()
    lo=min(bits(v[0]) for v in neg if len(v));hi=max(bits(v[-1]) for v in neg if len(v))
    def counts(b,side):return np.array([len(v)-np.searchsorted(v,np.float32(b),side=side) for v in neg],np.int64)
    def rate(fp):return float(np.mean(fp[supported]/den[supported]))
    while lo<hi:
        mid=(lo+hi)//2
        if rate(counts(value(mid),'right'))<=.01:hi=mid
        else:lo=mid+1
    boundary=float(value(lo));threshold=float(np.nextafter(np.float64(boundary),np.inf))
    fp=counts(boundary,'right');lower=counts(boundary,'left')
    assert rate(fp)<=.01 and rate(lower)>.01 and np.nextafter(threshold,-np.inf)==boundary
    detail=[];recalls=[];ties=[]
    for i,(gid,g) in enumerate(rows):
        s=np.asarray(g[policy],np.float64);y=np.asarray(g['truth'],bool);pred=s>=threshold
        actual=int((pred&~y).sum());unsafe=int(((s>=boundary)&~y).sum())
        assert actual==int(fp[i]) and unsafe==int(lower[i])
        p=int(y.sum());tp=int((pred&y).sum());recall=tp/p if p else None
        if p:recalls.append(recall)
        ties.append(int(((s==boundary)&~y).sum()))
        detail.append(dict(pool=pool,setting=policy,group_id=gid,positive=p,negative=int(den[i]),TP=tp,FP=actual,
            recall=recall,negative_FP_fraction=actual/int(den[i]) if den[i] else None))
    result=dict(pool=pool,setting=policy,status='oracle_reference_dependent_not_deployable',budget=.01,
        threshold=threshold,unsafe_negative_boundary=boundary,macro_recall_pp=float(np.mean(recalls))*100 if recalls else None,
        macro_negative_FP_pp=rate(fp)*100,max_group_negative_FP_pp=float(np.max(fp[supported]/den[supported]))*100,
        zero_recall_positive_groups=sum(v==0 for v in recalls),negative_groups=int(supported.sum()),AP_groups=len(recalls),
        lower_boundary_macro_negative_FP_pp=rate(lower)*100,atomic_negative_tie_counts=ties,
        rule='nextafter(first unsafe negative tie,+inf); float64 score >= threshold',
        independent_direct_counts_passed=True,mean_constraint_is_not_per_group_safety=True)
    return result,detail

def evaluate(groups,pool):
    per=[]
    for gid,g in sorted(groups.items()):
        y=g['truth'];p=int(y.sum());n=len(y)-p
        row=dict(pool=pool,group_id=gid,products=g['products'],positive=p,negative=n,sparse_support=bool(p and (p<5 or n<20)))
        for policy in POLICIES:
            row[policy+'_AP']=ap(y,g[policy])
            row.update({policy+'_'+k:v for k,v in operating(y,g[policy]).items()})
        for a,b in PAIRS:
            row[a+'_minus_'+b+'_AP_pp']=(row[a+'_AP']-row[b+'_AP'])*100 if row[a+'_AP'] is not None else None
        for name in ('C-R','D-C','D-R'):
            a,b=name.split('-');keys=[f'{a}{s}_minus_{b if b=="R" else b+str(s)}_AP_pp' for s in (12001,12002)]
            row['two_seed_'+name+'_AP_pp']=float(np.mean([row[k] for k in keys])) if row[keys[0]] is not None else None
        per.append(row)
    summary=[];truth=np.concatenate([g['truth'] for _,g in sorted(groups.items())])
    for policy in POLICIES:
        scores=np.concatenate([g[policy] for _,g in sorted(groups.items())]);pa=ap(truth,scores)
        vals=[r[policy+'_AP'] for r in per if r[policy+'_AP'] is not None]
        rr=[r[policy+'_recall'] for r in per if r[policy+'_recall'] is not None]
        fp=[r[policy+'_negative_FP_fraction'] for r in per if r[policy+'_negative_FP_fraction'] is not None]
        summary.append(dict(pool=pool,setting=policy,all_groups=len(per),AP_groups=len(vals),
            macro_AP_pp=float(np.mean(vals))*100 if vals else None,pooled_AP_pp=pa*100 if pa is not None else None,
            macro_recall_pp=float(np.mean(rr))*100 if rr else None,macro_negative_FP_pp=float(np.mean(fp))*100 if fp else None,
            worst_group_negative_FP_pp=max(fp)*100 if fp else None,zero_recall_positive_groups=sum(v==0 for v in rr),
            **{'pooled_'+k:v for k,v in operating(truth,scores).items()}))
        del scores
    comparisons=[]
    for a,b in PAIRS:
        key=a+'_minus_'+b+'_AP_pp'
        c=contrast(pool,a+'-'+b,[r[key] for r in per if r[key] is not None])
        rr=[(r[a+'_recall']-r[b+'_recall'])*100 for r in per if r[a+'_recall'] is not None]
        c.update(worst_delta_AP_pp=min((r[key] for r in per if r[key] is not None),default=None),
                 worst_delta_recall_pp=min(rr,default=None))
        comparisons.append(c)
    for name in ('C-R','D-C','D-R'):
        key='two_seed_'+name+'_AP_pp'
        comparisons.append(contrast(pool,'two_seed_'+name,[r[key] for r in per if r[key] is not None]))
    return per,summary,comparisons

def deliver(directory,pool,groups):
    out=Path(directory);out.mkdir(exist_ok=True,parents=True)
    per,summary,comparison=evaluate(groups,pool)
    table(out/(pool+'_per_group.csv'),per);table(out/(pool+'_summary.csv'),summary)
    with (out/(pool+'_comparisons.json')).open('x',encoding='utf-8') as f:
        json.dump(dict(rows=comparison,unit='related groups; within-group seed average before resampling'),f,indent=2,allow_nan=False)
    ors=[];details=[]
    for policy in POLICIES:
        r,d=oracle(groups,policy,pool);ors.append(r);details.extend(d)
    table(out/(pool+'_oracle.csv'),ors);table(out/(pool+'_oracle_per_group.csv'),details)
    contrasts=[]
    for seed in (12001,12002):
        for a,b in ((f'C{seed}','R'),(f'D{seed}',f'C{seed}')):
            av=next(r for r in ors if r['setting']==a);bv=next(r for r in ors if r['setting']==b)
            contrasts.append(dict(pool=pool,comparison=a+'-'+b,oracle_delta_macro_recall_pp=av['macro_recall_pp']-bv['macro_recall_pp'],
                not_deployable=True))
    with (out/(pool+'_oracle_comparisons.json')).open('x',encoding='utf-8') as f:json.dump(contrasts,f,indent=2)
    return per,summary,comparison
