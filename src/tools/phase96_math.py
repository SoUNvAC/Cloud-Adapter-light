"""Fixed Phase96 tail and marginal diagnostics, without fitting or threshold search."""
import numpy as np

QUANTILES=(.01,.05,.5,.95,.99)
BANDS=('B04','B03','B02','B08','B11','B12','Source_logit')


def describe(x):
    a=np.asarray(x).ravel(); finite=np.isfinite(a); values=a[finite]
    out=dict(count=len(a),finite_count=int(finite.sum()),finite_fraction=float(finite.mean()) if len(a) else None,
             zero_count=int((a==0).sum()),zero_fraction=float((a==0).mean()) if len(a) else None)
    if len(values):
        out.update(min=float(values.min()),max=float(values.max()),**{f'p{int(100*q)}':float(v) for q,v in zip(QUANTILES,np.quantile(values,QUANTILES))})
    else:out.update(min=None,max=None,**{f'p{int(100*q)}':None for q in QUANTILES})
    return out


def tail(rows,key='fpr'):
    eligible=[r for r in rows if r[key] is not None]; a=np.array([r[key] for r in eligible],np.float64)
    ordered=sorted(eligible,key=lambda r:(-r[key],r['group_id'])); total=float(a.sum())
    return dict(groups=len(rows),defined_groups=len(a),mean=float(a.mean()) if len(a) else None,
        median=float(np.median(a)) if len(a) else None,p90=float(np.quantile(a,.9)) if len(a) else None,
        max=float(a.max()) if len(a) else None,groups_above_1percent=int((a>.01).sum()),
        top1_share_of_group_rate_sum=ordered[0][key]/total if total else None,
        top3_share_of_group_rate_sum=sum(r[key] for r in ordered[:3])/total if total else None,
        mean_without_worst=float(np.mean([r[key] for r in ordered[1:]])) if len(a)>1 else None,
        worst_groups=[dict(product=r['product'],group_id=r['group_id'],rate=r[key]) for r in ordered[:3]],
        full_mean_primary=True,leave_one_out_descriptive_only=True)


def extremes(x,reference):
    a=np.asarray(x); lo=reference['p1']; hi=reference['p99']
    return dict(count=len(a),below_p1_count=int((a<lo).sum()),above_p99_count=int((a>hi).sum()),
        below_p1_fraction=float((a<lo).mean()) if len(a) else None,
        above_p99_fraction=float((a>hi).mean()) if len(a) else None)


def bin_masks(prob):
    a=np.asarray(prob); assert np.isfinite(a).all() and ((a>=0)&(a<=1)).all()
    masks=[a<.1,(a>=.1)&(a<.5),a>=.5]
    assert np.array_equal(sum(m.astype(np.uint8) for m in masks),np.ones(a.shape,np.uint8))
    return masks


def self_test():
    a=np.array([0,.099999,.1,.499999,.5,1]); assert [int(m.sum()) for m in bin_masks(a)]==[2,2,2]
    rows=[dict(group_id=str(i),product=str(i),fpr=x) for i,x in enumerate([.8,.1,.05,0])]
    t=tail(rows); assert abs(t['mean']-.2375)<1e-14 and abs(t['top1_share_of_group_rate_sum']-.8/.95)<1e-14
    assert abs(t['top3_share_of_group_rate_sum']-1)<1e-14
    e=extremes(np.array([0,1,2,3]),dict(p1=1,p99=2)); assert e['below_p1_count']==e['above_p99_count']==1
    d=describe([0,np.nan,1]); assert d['finite_count']==2 and d['zero_count']==1
    return dict(bin_boundaries_exact=True,tail_group_weights=True,marginal_strict_bounds=True,nonfinite_summary=True)


if __name__=='__main__':print(self_test())
