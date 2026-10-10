"""Frozen observable-neighbor interventions; labels never enter feature construction."""
import hashlib
import numpy as np
from phase70_math import square_sum, checked


def shuffled_mask(product, valid, excluded):
    seed=74010+int(hashlib.sha256(product.encode()).hexdigest()[:8],16)
    order=np.random.default_rng(seed).permutation(int(valid.sum()))
    out=np.zeros(valid.shape,bool);out[valid]=excluded[valid][order]
    assert out.sum()==excluded[valid].sum()
    return out,dict(seed=seed,count=len(order),indices_sha256=hashlib.sha256(order.tobytes()).hexdigest(),
                    matching='whole-image exclusion count only; not per-window count')


def contrast(raw,valid,excluded,window=61):
    valid=np.asarray(valid,bool);excluded=np.asarray(excluded,bool)
    eligible=valid & ~excluded;radius=window//2
    original_n=square_sum(valid,radius)-valid
    kept=square_sum(eligible,radius)-eligible
    if np.any(original_n[valid]<=0):raise ValueError('No original neighbors')
    fallback=kept[valid]<16;means=[];baseline=[];values=[]
    for channel in [7,11,12]:
        b=raw[...,channel].astype(np.float64)
        old=(square_sum(np.where(valid,b,0.),radius)[valid]-b[valid])/original_n[valid]
        numerator=(square_sum(np.where(eligible,b,0.),radius)-np.where(eligible,b,0.))[valid]
        new=np.divide(numerator,kept[valid],out=old.copy(),where=~fallback)
        baseline.append(old);means.append(new);values.append((b[valid]-new)/(np.abs(new)+1e-6))
    return checked(np.column_stack(values)),np.column_stack(means),np.column_stack(baseline),dict(
        remaining=kept[valid],original=original_n[valid],fallback=fallback)


def interval(values):
    a=np.asarray(values,float)
    if not len(a) or not np.isfinite(a).all():raise ValueError('Missing/nonfinite group support')
    rng=np.random.default_rng(74010);b=a[rng.integers(0,len(a),(10000,len(a)))].mean(axis=1)
    return dict(mean=float(a.mean()),n_groups=len(a),ci95=np.quantile(b,[.025,.975]).tolist(),
                upper95=float(np.quantile(b,.95)),seed=74010,replicates=10000,descriptive=True)


def changes(before,after,positive):
    before=np.asarray(before,bool);after=np.asarray(after,bool);positive=np.asarray(positive,bool)
    return dict(new_fp=int((~positive & ~before & after).sum()),removed_fp=int((~positive & before & ~after).sum()),
                rescued_tp=int((positive & ~before & after).sum()),lost_tp=int((positive & before & ~after).sum()))


def distribution(values):
    a=np.asarray(values)
    if not a.size:return dict(n=0,mean=None,quantiles=None)
    return dict(n=int(a.size),mean=float(a.mean(dtype=np.float64)),quantiles=np.quantile(a,[0,.05,.5,.95,1]).tolist())


def continue_conditions(p):
    e,r,b=(p[n] for n in ['B_E_fit','B_R_fit','B'])
    return dict(E_FPR_below_B=e['fpr_strata']['all']['mean']<b['fpr_strata']['all']['mean'],
        E_recall_loss_le_1pp=e['macro_recall']>=b['macro_recall']-.01,
        E_AP_ge_B=e['macro_ap']>=b['macro_ap'],E_AP_gt_R=e['macro_ap']>r['macro_ap'],
        E_FPR_le_R=e['fpr_strata']['all']['mean']<=r['fpr_strata']['all']['mean'])
