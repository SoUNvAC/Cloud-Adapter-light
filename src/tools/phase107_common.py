"""Matched readout contracts. No fit/inference on import."""
import hashlib,json
import numpy as np
from pathlib import Path
from phase104_common import REPO,read,write,sha,now,csvwrite
from phase65d_recovery_math import logit_score,predict_logit
from phase89_cpu_ap import average_precision_score as independent_ap
RESULT=REPO/'src/research_results/phase107_20261013'
SHARED=Path('/home/scv/shared');ROOT=SHARED/'phase107_20261013'
CAT=SHARED/'data/sentinel2_cloud_mask_catalogue_4172871'
CONTROL_SHA='c2a83964073918015460e0ad89a9857b60b6a764e0f8562652472b224915f5ad'

def array_sha(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def ap(y,s):
    y=np.asarray(y,bool);s=np.asarray(s,np.float32);assert len(y)==len(s) and np.isfinite(s).all()
    if not y.any() or y.all():return None
    order=np.argsort(-s,kind='stable');ss=s[order];yy=y[order]
    ends=np.flatnonzero(np.r_[ss[1:]!=ss[:-1],True]);tp=np.cumsum(yy,dtype=np.int64)[ends]
    value=float(np.sum(np.diff(np.r_[0,tp])*(tp/(ends+1))/int(y.sum())))
    other=independent_ap(y,s);assert abs(value-other)<1e-12
    return value

def scalar_replay(x,model,score):
    z=(x.astype(np.float64)-np.asarray(model['mean']))/np.asarray(model['scale'])
    v=z[:,0]*model['theta'][0]
    for i in range(1,4):v=v+z[:,i]*model['theta'][i]
    v=(v+model['theta'][-1]).astype(np.float32)
    error=np.abs(score.astype(float)-v.astype(float))
    assert np.all(error<=np.abs(np.spacing(score)).astype(float))
    return dict(max_abs_error=float(error.max()),bitwise_equal=int(np.count_nonzero(score==v)),samples=len(v),tolerance='one float32 ULP')

def comparisons(rows,domain):
    selected=[r for r in rows if r['A_AP'] is not None];n=len(selected)
    assert n==(7 if domain=='development' else 8)
    result=[]
    for left,right in [('S','R'),('R','A')]:
        differences=np.array([(r[left+'_AP']-r[right+'_AP'])*100 for r in selected])
        idx=np.random.default_rng(107013).integers(0,n,(10000,n))
        ci=np.quantile(differences[idx].mean(1),[.025,.975])
        cpu=sorted(sum(float(differences[i]) for i in indices)/n for indices in idx)
        def q(p):
            k=p*9999;i=int(k);return cpu[i]+(k-i)*(cpu[min(i+1,9999)]-cpu[i])
        assert np.max(np.abs(ci-[q(.025),q(.975)]))<1e-12
        result.append(dict(domain=domain,comparison=left+'-'+right,primary=domain=='PixBox' and left=='S',
            groups=n,mean_delta_AP_pp=float(differences.mean()),CI95_pp=ci.tolist(),width_pp=float(ci[1]-ci[0]),
            seed=107013,resamples=10000,improved=int((differences>1).sum()),stable=int((np.abs(differences)<=1).sum()),
            declined=int((differences< -1).sum()),worst_group=selected[int(np.argmin(differences))]['tile'],worst_delta_AP_pp=float(differences.min()),
            evidence='uncertain' if ci[0]<=0<=ci[1] else 'positive' if ci[0]>0 else 'negative',exploratory=True))
    return result
