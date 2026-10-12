"""Frozen float64 fusion and non-interpolated AP; never quantize fusion scores."""
import numpy as np
from phase89_cpu_ap import average_precision_score as independent_ap
from phase91_math import scan,cpu_validate

def fuse(g,logit):
    assert g.dtype==logit.dtype==np.float32 and g.shape==logit.shape
    assert np.isfinite(g).all() and np.isfinite(logit).all() and ((g>=0)&(g<=1)).all()
    pg=g.astype(np.float64);z=logit.astype(np.float64)
    ps=np.exp(-np.logaddexp(0.,-z))
    avg=(pg+ps)/2;minimum=np.minimum(pg,ps)
    # Independent stable conversion, and algebraically equivalent fusion checks.
    other=np.empty_like(z);pos=z>=0;other[pos]=1/(1+np.exp(-z[pos]));e=np.exp(z[~pos]);other[~pos]=e/(1+e)
    error=float(np.max(np.abs(ps-other))) if len(z) else 0.
    assert error<=1e-15
    assert np.array_equal(avg,pg/2+ps/2) and np.array_equal(minimum,np.where(pg<=ps,pg,ps))
    assert (minimum<=pg).all() and (minimum<=ps).all()
    assert (avg>=minimum).all() and (avg<=np.maximum(pg,ps)).all()
    assert avg.dtype==minimum.dtype==np.float64 and np.isfinite(avg).all() and np.isfinite(minimum).all()
    return dict(AVG=avg,MIN=minimum),dict(count=len(g),sigmoid_independent_max_error=error,
        G7_dtype=str(g.dtype),SourceS_logit_dtype=str(logit.dtype),fusion_dtype='float64',
        MIN_bounded=True,AVG_between=True,independent_formula_exact=True),ps

def ap(score,truth):
    assert score.dtype in [np.dtype('float32'),np.dtype('float64')]
    positive=truth==2
    if not positive.any() or positive.all():return None
    order=np.argsort(-score,kind='stable');ss=score[order];yy=positive[order]
    ends=np.flatnonzero(np.r_[ss[1:]!=ss[:-1],True])
    tp=np.cumsum(yy,dtype=np.int64)[ends];precision=tp/(ends+1)
    result=float(np.sum(np.diff(np.r_[0,tp])*precision/int(positive.sum())))
    value=independent_ap(positive,score)
    assert abs(result-value)<1e-12,(result,value)
    return result

def self_test():
    g=np.array([0,.1,.5,1],np.float32);z=np.array([-1000,-2,0,1000],np.float32)
    out,receipt,ps=fuse(g,z);assert out['MIN'][0]==0 and out['AVG'][-1]==1
    s=np.array([.5,.5+1e-10],np.float64);y=np.array([0,2],np.uint8)
    assert ap(s,y)==1 and ap(s.astype(np.float32),y)==.5
    point=scan([s],[y],[0])[0];cpu_validate(point,[s],[y]);assert point['threshold']==s[1]
    ties=np.array([.8,.8,.5,.5],np.float64);truth=np.array([0,0,2,0],np.uint8)
    point=scan([ties],[truth],[.34])[0];cpu_validate(point,[ties],[truth]);assert point['macro_fpr']<=.34
    assert ap(np.array([.5,.5,.5],np.float64),np.array([2,0,0]))==1/3
    return dict(float64_near_tie_AP_retained=True,float64_threshold_retained=True,atomic_ties_verified=True,
        stable_extreme_logit_conversion=True,fusion_bounds_and_independent_formulas=True)
