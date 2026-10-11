"""Frozen Phase89 spatial/decision contracts; no fitting or I/O."""
import numpy as np
from phase65d_recovery_math import logit_score, predict_logit

NAMES = ('Source', 'MsRE', 'L2', 'L3')

def area(a, factor):
    a = np.asarray(a)
    h, w = a.shape[:2]
    assert h % factor == w % factor == 0
    return a.reshape(h//factor, factor, w//factor, factor, *a.shape[2:]).mean(axis=(1,3), dtype=np.float64)

def all_valid(a, factor):
    h,w=a.shape
    return a.reshape(h//factor,factor,w//factor,factor).all(axis=(1,3))

def sigmoid(a): return np.exp(-np.logaddexp(0., -np.asarray(a,dtype=np.float64)))

def calibrate(ms, ss, spectra, models):
    assert spectra.shape[:2] == ms.shape == ss.shape and spectra.shape[2] == 3
    x=np.column_stack([logit_score(ms.ravel()),logit_score(ss.ravel()),spectra.reshape(-1,3)]).astype(np.float32)
    return {n:sigmoid(predict_logit(x[:,:k],models[key])).reshape(ms.shape)
            for n,key,k in [('L2','msre_source',2),('L3','msre_source_spectral',5)]}

def decisions(prob, models):
    nonshadow=prob['MsRE'][...,:2].argmax(-1).astype(np.uint8)
    out={n:prob[n].argmax(-1).astype(np.uint8) for n in NAMES[:2]}
    for n,key in [('L2','msre_source'),('L3','msre_source_spectral')]:
        out[n]=np.where(prob[n]>=sigmoid(models[key]['threshold']),2,nonshadow).astype(np.uint8)
    return out

def interval(values):
    x=np.asarray(values,dtype=np.float64)
    assert x.ndim==1 and np.isfinite(x).all()
    r=dict(n_groups=len(x),mean=float(x.mean()) if len(x) else None,ci95=None,upper95_one_sided=None,
           seed=89011,resamples=10000,descriptive_only=True,unit='equal related-group representative')
    if len(x)>1:
        b=x[np.random.default_rng(89011).integers(0,len(x),(10000,len(x)))].mean(1)
        r.update(ci95=np.quantile(b,[.025,.975]).tolist(),upper95_one_sided=float(np.quantile(b,.95)))
    return r
