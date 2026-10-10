"""Phase68 fixed conditional group loss and descriptive group statistics."""
import numpy as np
from phase65d_recovery_math import objective, group_threshold


def conditional_weights(labels):
    counts=[(int(y.sum()),len(y)-int(y.sum())) for y in labels]
    gp=sum(p>0 for p,n in counts);gn=sum(n>0 for p,n in counts)
    if not gp or not gn:raise ValueError('Missing sampled class support; no replacement draw')
    weights=[]
    for y,(p,n) in zip(labels,counts):
        weights.append(np.where(y,.5/(gp*p) if p else 0.,.5/(gn*n) if n else 0.))
    return np.concatenate(weights)


def fit_fixed_scaler(x,y,w,original):
    from scipy.optimize import minimize
    mean=np.asarray(original['mean']);scale=np.asarray(original['scale'])
    if x.shape[1]!=5 or not np.isfinite(x).all() or np.any(scale<=0):raise ValueError('Invalid frozen inputs')
    z=(x-mean)/scale
    result=minimize(objective,np.zeros(6),args=(z,y,w),jac=True,method='L-BFGS-B',
        bounds=[(0,None),(0,None)]+[(None,None)]*4,
        options={'maxiter':500,'gtol':1e-7,'ftol':1e-12})
    if not result.success:raise RuntimeError('Frozen optimizer failed: '+str(result.message))
    return dict(mean=original['mean'],scale=original['scale'],theta=result.x.tolist(),
        iterations=int(result.nit),optimizer_success=True,objective=float(result.fun),
        penalty=.001,positive_features=2,scaler='unchanged original L3 fit scaler')


def dual_threshold(negatives,positive_support):
    strata=[[a for a,p in zip(negatives,positive_support) if bool(p)==s] for s in [True,False]]
    thresholds=[group_threshold(a,.01) for a in strata]
    return max(thresholds),dict(shadow=thresholds[0],no_shadow=thresholds[1])


def interval(values):
    a=np.asarray(values,float)
    if not len(a) or not np.isfinite(a).all():raise ValueError('Missing/nonfinite group support')
    rng=np.random.default_rng(68010)
    means=a[rng.integers(0,len(a),(10000,len(a)))].mean(axis=1)
    return dict(mean=float(a.mean()),n_groups=len(a),ci95=np.quantile(means,[.025,.975]).tolist(),
        upper95=float(np.quantile(means,.95)),replicates=10000,seed=68010,
        conditional_on_fixed_model=True,independent_confirmation=False)


def continue_gate(p):
    a,b,d=[p[n] for n in ['A','B','D']]
    conditions=dict(recall_D_gt_B=d['macro_recall']>b['macro_recall'],
        recall_D_gt_A=d['macro_recall']>a['macro_recall'],
        shadow_fpr_D_le_1pct=d['fpr_strata']['shadow']['mean']<=.01,
        no_shadow_fpr_D_le_1pct=d['fpr_strata']['no_shadow']['mean']<=.01,
        macro_ap_D_ge_A=d['macro_ap']>=a['macro_ap'],
        pooled_miou_D_ge_A=d['miou_percent']>=a['miou_percent'],
        zero_recall_D_le_A=d['zero_recall_groups']<=a['zero_recall_groups'])
    return dict(conditions=conditions,all_met=all(conditions.values()),
        decision='eligible_to_propose_new_independent_protocol' if all(conditions.values()) else 'stop_this_hypothesis_no_additional_candidates',
        phase65c_authorized=False)
