"""Fixed Phase70 observable features, product-paired nulls and group decisions."""
import hashlib
import numpy as np

BRANCHES={'A':6,'B':3,'C':3}


def square_sum(a,radius):
    h,w=a.shape;dtype=np.float64 if np.issubdtype(a.dtype,np.floating) else np.int64
    summed=np.pad(np.cumsum(np.cumsum(a,dtype=dtype,axis=0),dtype=dtype,axis=1),((1,0),(1,0)))
    y=np.arange(h);x=np.arange(w);yl=np.maximum(y-radius,0);yh=np.minimum(y+radius+1,h)
    xl=np.maximum(x-radius,0);xh=np.minimum(x+radius+1,w)
    return summed[yh[:,None],xh]-summed[yl[:,None],xh]-summed[yh[:,None],xl]+summed[yl[:,None],xl]


def interaction(original,model):
    z=(np.asarray(original,dtype=float)-model['mean'])/model['scale']
    # Frozen original L3 standardization applies to BOTH logits and spectral columns.
    return (z[:,:2,None]*z[:,None,2:]).reshape(len(z),6)


def local_contrast(raw,valid,window=61):
    if window<3 or window%2!=1:raise ValueError('Odd window >=3 required')
    den=square_sum(valid,window//2)-valid
    if np.any(den[valid]<=0):raise ValueError('Valid center has no image-valid neighbors')
    columns=[]
    for channel in [7,11,12]:
        band=raw[...,channel].astype(np.float64)
        mean=(square_sum(np.where(valid,band,0.),window//2)[valid]-band[valid])/den[valid]
        columns.append((band[valid]-mean)/(np.abs(mean)+1e-6))
    return np.column_stack(columns)


def normalized_differences(raw,valid):
    columns=[];protected=[]
    for first,second in [(2,7),(7,11),(2,11)]:
        a=raw[...,first][valid].astype(np.float64);b=raw[...,second][valid].astype(np.float64)
        denominator=np.abs(a+b);protected.append(int((denominator<1e-6).sum()))
        columns.append((a-b)/np.maximum(denominator,1e-6))
    return np.column_stack(columns),dict(protected_count=protected,observable_pixels=int(valid.sum()),
        protected_fraction=[n/int(valid.sum()) for n in protected],epsilon=1e-6,
        definition='abs(raw native TOA band sum)<1e-6; no index clipping')


def checked(values):
    with np.errstate(over='ignore',invalid='ignore'):out=np.asarray(values,dtype=np.float32)
    if not np.isfinite(values).all() or not np.isfinite(out).all():raise ValueError('Nonfinite new feature; no clipping or formula change')
    return out


def permutation(product,branch,count):
    seed=70010+('ABC'.index(branch)+1)*1000+int(hashlib.sha256(product.encode()).hexdigest()[:8],16)
    p=np.random.default_rng(seed).permutation(count)
    return p,dict(seed=seed,count=count,sha256=hashlib.sha256(p.tobytes()).hexdigest())


def interval(values):
    a=np.asarray(values,float)
    if not len(a) or not np.isfinite(a).all():raise ValueError('Missing/nonfinite group support')
    rng=np.random.default_rng(70010);means=a[rng.integers(0,len(a),(10000,len(a)))].mean(axis=1)
    return dict(mean=float(a.mean()),n_groups=len(a),ci95=np.quantile(means,[.025,.975]).tolist(),
        upper95=float(np.quantile(means,.95)),replicates=10000,seed=70010,independent_confirmation=False,
        conditional_on_fixed_model=True)


def promotion(p,comparisons,status):
    gates={};eligible=[];base=p['L3']
    for branch,dim in BRANCHES.items():
        if status[branch]['status']!='completed':
            gates[branch]=dict(eligible=False,status=status[branch]);continue
        r=p[branch];c=comparisons[branch+'_minus_L3'];delta=np.array([x['delta_ap'] for x in c['pairs']])
        loo=(delta.sum()-delta)/(len(delta)-1)
        conditions=dict(mean_ap_gain_ge_1pp=c['ap']['mean']>=.01,
            at_least_four_groups_gt_1pp=int((delta>.01).sum())>=4,all_leave_one_out_means_positive=bool(np.all(loo>0)),
            recall_ge_L3=r['macro_recall']>=base['macro_recall'],all_mean_fpr_le_1pct=r['fpr_strata']['all']['mean']<=.01,
            pooled_miou_ge_L3=r['miou_percent']>=base['miou_percent'],zero_recall_le_L3=r['zero_recall_groups']<=base['zero_recall_groups'],
            macro_ap_gt_matched_shuffle=r['macro_ap']>p[branch+'_shuffle']['macro_ap'])
        ok=all(conditions.values());gates[branch]=dict(eligible=ok,conditions=conditions,
            leave_one_out_ap_means=loo.tolist(),groups_gt_1pp=int((delta>.01).sum()),groups_gt_zero=int((delta>0).sum()))
        if ok:eligible.append(branch)
    eligible.sort(key=lambda n:(-comparisons[n+'_minus_L3']['ap']['mean'],BRANCHES[n],n))
    return dict(branches=gates,selected=eligible[:2],max_selected=2,group_improvement_rule='delta AP > .01 (user confirmed)',
        decision='propose_next_stage_only' if eligible else 'close_no_additional_experiments',phase65c_authorized=False)
