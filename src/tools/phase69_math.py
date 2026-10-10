"""Observable cloud-neighborhood features, fixed paired shuffle, group inference."""
import hashlib
import numpy as np


def box_sum(a,radius):
    """Clipped square, no reflected/padded observations; int64 integral image."""
    h,w=a.shape
    integral=np.pad(np.cumsum(np.cumsum(a.astype(np.int64),axis=0),axis=1),((1,0),(1,0)))
    y=np.arange(h);x=np.arange(w)
    lo_y=np.maximum(y-radius,0);hi_y=np.minimum(y+radius+1,h)
    lo_x=np.maximum(x-radius,0);hi_x=np.minimum(x+radius+1,w)
    return (integral[hi_y[:,None],hi_x]-integral[lo_y[:,None],hi_x]
        -integral[hi_y[:,None],lo_x]+integral[lo_y[:,None],lo_x])


def neighborhood(pred,image_valid,windows=(31,127)):
    pred=np.asarray(pred);valid=np.asarray(image_valid,dtype=bool)
    if pred.shape!=valid.shape or pred.ndim!=2:raise ValueError('Prediction/mask shape mismatch')
    if not valid.any() or not np.isin(pred[valid],[0,1,2]).all():raise ValueError('Invalid prediction/observable support')
    cloud=(pred==1)&valid;parts=[cloud[valid].astype(np.float32)]
    for window in windows:
        if window<3 or window%2!=1:raise ValueError('Window must be odd and >=3')
        count=box_sum(cloud,window//2)-cloud
        denom=box_sum(valid,window//2)-valid
        if np.any(denom[valid]<=0):raise ValueError('Image-valid center has zero valid neighbors; stop')
        parts.append((count[valid]/denom[valid]).astype(np.float32))
    return np.column_stack(parts)


def observable_features(pred,image_valid,product):
    real=neighborhood(pred,image_valid)
    seed=69010+int(hashlib.sha256(product.encode()).hexdigest()[:8],16)
    permutation=np.random.default_rng(seed).permutation(len(real))
    # C remains at its own center. Both neighborhoods move as ONE paired vector.
    joined=np.column_stack([real,real[permutation,1:]])
    return joined,dict(permutation_seed=seed,permutation_sha256=hashlib.sha256(permutation.tobytes()).hexdigest(),
        permutation_count=len(permutation),observable_values_sha256=hashlib.sha256(joined.tobytes()).hexdigest(),
        feature_order=['C','cloud31','cloud127','shuffled_cloud31','shuffled_cloud127'],
        feature_scope='all image-valid pixels before any label selection; no truth values used')


def inputs(original,observable,policy):
    if policy=='A':return np.asarray(original)
    if policy=='B':return np.column_stack([original,observable[:,0]])
    if policy=='D':return np.column_stack([original,observable[:,:3]])
    if policy=='S':return np.column_stack([original,observable[:,[0,3,4]]])
    raise ValueError('Only four frozen policies authorized')


def interval(values):
    a=np.asarray(values,float)
    if not len(a) or not np.isfinite(a).all():raise ValueError('Missing/nonfinite group support')
    rng=np.random.default_rng(69010)
    means=a[rng.integers(0,len(a),(10000,len(a)))].mean(axis=1)
    return dict(mean=float(a.mean()),n_groups=len(a),ci95=np.quantile(means,[.025,.975]).tolist(),
        upper95=float(np.quantile(means,.95)),replicates=10000,seed=69010,
        conditional_on_fixed_model=True,independent_confirmation=False)


def gate(p,comparisons):
    a,d=p['A'],p['D']
    conditions=dict(ap_D_minus_B_lower_gt_zero=comparisons['D_minus_B']['ap']['ci95'][0]>0,
        ap_D_minus_S_lower_gt_zero=comparisons['D_minus_S']['ap']['ci95'][0]>0,
        recall_D_ge_A=d['macro_recall']>=a['macro_recall'],all_group_fpr_D_le_1pct=d['fpr_strata']['all']['mean']<=.01,
        pooled_miou_D_ge_A=d['miou_percent']>=a['miou_percent'],zero_recall_D_le_A=d['zero_recall_groups']<=a['zero_recall_groups'])
    return dict(conditions=conditions,all_met=all(conditions.values()),
        decision='neighborhood_information_supported_in_development_only' if all(conditions.values()) else 'stop_no_additional_features_or_fits',
        phase65c_authorized=False)
