"""Fixed pseudo-Surface-positive logistic math, independent of supervised data entrances."""
import numpy as np
from scipy.optimize import minimize
from phase65d_recovery_math import objective,standardize

def features(raw,name):
    x=np.asarray(raw,dtype=np.float64);assert x.ndim==2 and x.shape[1]==6 and np.isfinite(x).all()
    if name=='RGB3':return x[:,:3].copy()
    if name=='RGB6Q':return np.c_[x[:,:3],x[:,:3]**2]
    assert name=='MS6';return x.copy()
def fit(x,y,w,sizes):
    assert set(np.unique(y))=={0,1} and len(x)==len(y)==len(w)==sum(sizes)
    w=w/w.sum();mean,scale=standardize(x,w);constant=np.all(x==x[0],axis=0)
    mean[constant]=x[0,constant];scale[constant]=1
    offsets=np.r_[0,np.cumsum(sizes)]
    independent=np.mean([x[offsets[i]:offsets[i+1]].mean(0) for i in range(len(sizes))],axis=0)
    variance=np.mean([((x[offsets[i]:offsets[i+1]]-independent)**2).mean(0) for i in range(len(sizes))],axis=0)
    independent_scale=np.sqrt(variance);independent_scale[independent_scale==0]=1
    independent[constant]=x[0,constant];independent_scale[constant]=1
    assert np.allclose(mean,independent,rtol=1e-12,atol=1e-12) and np.allclose(scale,independent_scale,rtol=1e-12,atol=1e-12)
    z=(x-mean)/scale
    result=minimize(objective,np.zeros(x.shape[1]+1),args=(z,y,w),jac=True,method='L-BFGS-B',bounds=[(None,None)]*(x.shape[1]+1),options=dict(maxiter=500,gtol=1e-7,ftol=1e-12))
    return dict(mean=mean.tolist(),scale=scale.tolist(),theta=result.x.tolist(),constant_columns=np.flatnonzero(constant).tolist(),iterations=int(result.nit),objective=float(result.fun),optimizer_success=bool(result.success),optimizer_message=str(result.message),optimizer_status=int(result.status),gradient=result.jac.tolist(),final_gradient_inf=float(np.abs(result.jac).max()),penalty=.001,positive_class='Source-Surface=1; Source-Cloud=0',scaler_independently_recomputed=True,fit_initial='all zero',maxiter=500,gtol=1e-7,ftol=1e-12)
def predict(x,model):
    z=(x-np.asarray(model['mean']))/np.asarray(model['scale']);theta=np.asarray(model['theta'])
    p=np.exp(-np.logaddexp(0,-(z@theta[:-1]+theta[-1])))
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all();return p
def topk(scores,ids,k):
    assert len(scores)==len(ids) and 0<=k<=len(ids) and np.isfinite(scores).all()
    out=np.zeros(len(ids),bool);out[np.lexsort((ids,-scores))[:k]]=True;return out
def tests():
    x=np.array([[.1,.2],[-.4,.5],[.7,-.8]],np.float64);y=np.array([0.,1.,1.]);w=np.array([.2,.3,.5]);theta=np.array([.3,-.2,.7])
    value,g=objective(theta,x,y,w);finite=[]
    for i in range(3):
        a=theta.copy();b=theta.copy();a[i]+=1e-6;b[i]-=1e-6;finite.append((objective(a,x,y,w)[0]-objective(b,x,y,w)[0])/2e-6)
    error=float(np.abs(g-finite).max());assert error<=2e-7
    v0,g0=objective(theta,x,y,w,penalty=0);assert abs(value-v0-.5*.001*np.dot(theta[:-1],theta[:-1]))<1e-14 and g[-1]==g0[-1]
    raw=np.array([[2.,3.,4.,5.,6.,7.],[1.,2.,3.,4.,5.,6.]])
    assert np.array_equal(features(raw,'RGB6Q'),np.c_[raw[:,:3],raw[:,:3]**2]) and np.array_equal(features(raw,'MS6'),raw) and np.array_equal(features(raw,'RGB3'),raw[:,:3])
    xx=np.array([[1.,0.],[1.,2.],[1.,10.]]);ww=np.array([.25,.25,.5]);mean,scale=standardize(xx,ww)
    assert np.allclose(mean,[1,5.5]) and scale[0]==1 and abs(scale[1]**2-20.75)<1e-12
    before=(mean.copy(),scale.copy());predict(np.array([[1.,1e12]]),dict(mean=mean,scale=scale,theta=[0.,1.,0.]));assert np.array_equal(mean,before[0]) and np.array_equal(scale,before[1])
    pp=np.array([np.nextafter(.90,0),.90,np.nextafter(.90,1)]);assert (pp>=.9).tolist()==[False,True,True]
    scores=np.array([.5,.9,.9]);ids=np.array([30,20,10]);assert topk(scores,ids,1).tolist()==[False,False,True] and topk(scores,ids,2).tolist()==[False,True,True]
    assert not topk(scores,ids,0).any() and topk(scores,ids,3).all() and not topk(np.array([]),np.array([],np.int64),0).any()
    assert predict(np.array([[-1.],[1.]]),dict(mean=[0.],scale=[1.],theta=[1.,0.]))[1]>.5
    return dict(status='passed',gradient_finite_difference_max_error=error,unpenalized_intercept=True,positive_is_pseudo_Surface=True,raw_TOA_features_and_squares=True,equal_product_weights_train_only_scaler=True,empty_and_constant_cases=True,threshold_point90_neighbors=True,topk_ties_flat_ascending_and_zero_full=True)
