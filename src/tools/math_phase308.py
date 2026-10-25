"""Read-only fixed probe equations and explicitly defined descriptive statistics."""
import hashlib
import numpy as np
QS=(0,.01,.10,.50,.90,.99,1)
QN=('min','p1','p10','p50','p90','p99','max')
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def features(raw,model):
    x=np.asarray(raw,dtype=np.float64)
    assert x.ndim==2 and x.shape[1]==6 and np.isfinite(x).all()
    if model=='RGB3':return x[:,:3].copy()
    if model=='RGB6Q':return np.concatenate([x[:,:3],x[:,:3]**2],axis=1)
    assert model=='MS6';return x.copy()
def quant(x,w=None):
    x=np.asarray(x,dtype=np.float64)
    if not len(x):return dict.fromkeys(QN,None)
    assert np.isfinite(x).all()
    if w is None:v=np.quantile(x,QS,method='linear')
    else:
        w=np.asarray(w,dtype=np.float64);assert len(x)==len(w) and (w>0).all()
        order=np.argsort(x,kind='stable');xx=x[order];cw=np.cumsum(w[order]/w.sum())
        v=xx[np.minimum(np.searchsorted(cw,QS,side='left'),len(xx)-1)]
    return dict(zip(QN,map(float,v)))
def describe(x,w=None):
    return dict(n=len(x),mean=float(np.mean(x) if w is None else np.average(x,weights=w)) if len(x) else None,**quant(x,w))
def fixed(x,receipt):
    z=(x-np.asarray(receipt['mean']))/np.asarray(receipt['scale'])
    beta=np.asarray(receipt['theta'][:-1]);intercept=receipt['theta'][-1]
    logit=z@beta+intercept;contribution=z*beta
    error=float(np.max(np.abs(contribution.sum(1)+intercept-logit))) if len(z) else 0.
    assert error<=1e-12
    return z,logit,np.exp(-np.logaddexp(0,-logit)),contribution,error
def tests():
    assert quant([1,2,3,4],[1,1,1,1])['p50']==2
    assert quant([7,7,7],[1,2,9])['p99']==7
    assert quant([3],[2])['p1']==3 and quant([])['p50'] is None
    assert quant([1,2,3],[8,1,1])['p50']==1
    x=np.array([-2.,0.,2.]);assert np.array_equal(np.exp(-np.logaddexp(0,-x))>=.5,[False,True,True])
    return dict(weighted_uniform=True,ties=True,single=True,empty=True,unequal_weights=True,fixed_direction_and_half=True)
