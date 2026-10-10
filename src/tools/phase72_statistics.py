"""Fixed Phase72 group inference and sequential decision rules."""
import numpy as np

def interval(values):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or not np.isfinite(x).all():raise ValueError('Nonfinite group statistic')
    result=dict(n_groups=len(x),mean=float(x.mean()) if len(x) else None,ci95=None,
                lower95_one_sided=None,upper95_one_sided=None,seed=72010,resamples=10000)
    if len(x)<2:return result
    rng=np.random.default_rng(72010)
    means=x[rng.integers(0,len(x),(10000,len(x)))].mean(axis=1)
    result.update(ci95=np.quantile(means,[.025,.975]).tolist(),
                  lower95_one_sided=float(np.quantile(means,.05)),
                  upper95_one_sided=float(np.quantile(means,.95)),
                  bootstrap_degenerate=bool(np.ptp(means)==0))
    return result

def paired(left,right):
    assert len(left)==len(right)
    pairs=[]
    for a,b in zip(left,right):
        assert a['group_id']==b['group_id'] and a['product']==b['product']
        pairs.append(dict(group_id=a['group_id'],product=a['product'],
            delta_ap_pp=None if a['ranking']['ap'] is None or b['ranking']['ap'] is None else 100*(a['ranking']['ap']-b['ranking']['ap']),
            delta_recall_pp=None if a['recall'] is None or b['recall'] is None else 100*(a['recall']-b['recall']),
            delta_miou_pp=a['miou_percent']-b['miou_percent']))
    assert len({p['group_id'] for p in pairs})==len(pairs)
    ap=[p['delta_ap_pp'] for p in pairs if p['delta_ap_pp'] is not None]
    recall=[p['delta_recall_pp'] for p in pairs if p['delta_recall_pp'] is not None]
    miou=[p['delta_miou_pp'] for p in pairs]
    supported=[p for p in pairs if p['delta_ap_pp'] is not None]
    return dict(pairs=pairs,ap_pp=interval(ap),recall_pp=interval(recall),miou_pp=interval(miou),
        improved=sum(v>1 for v in ap),stable=sum(-1<=v<=1 for v in ap),declined=sum(v< -1 for v in ap),
        worst_ap=min(supported,key=lambda p:p['delta_ap_pp']) if supported else None,
        worst_recall=min((p for p in pairs if p['delta_recall_pp'] is not None),key=lambda p:p['delta_recall_pp'],default=None),
        leave_one_out_ap_mean_range_pp=[float(min((sum(ap)-v)/(len(ap)-1) for v in ap)),float(max((sum(ap)-v)/(len(ap)-1) for v in ap))] if len(ap)>1 else None)

def decision(primary,secondary,fpr):
    ap=primary['ap_pp'];rec=primary['recall_pp'];miou=primary['miou_pp']
    gates=dict(ap_increment=ap['ci95'] is not None and ap['ci95'][0]>0,
        recall_noninferiority=rec['lower95_one_sided'] is not None and rec['lower95_one_sided']>=-1,
        miou_noninferiority=miou['lower95_one_sided'] is not None and miou['lower95_one_sided']>=-1,
        fpr_budget=fpr['upper95_one_sided'] is not None and fpr['upper95_one_sided']<=.01)
    joint=all(gates.values());ci=secondary['ap_pp']['ci95']
    return dict(gates=gates,joint_B_vs_L3_confirmed=joint,
        specificity_test_performed=joint,
        spectral_specificity_confirmed=(ci is not None and ci[0]>0) if joint else None,
        specificity_role='confirmatory' if joint else 'descriptive_only_prerequisite_failed',
        all_claims_confirmed=joint and ci is not None and ci[0]>0)
