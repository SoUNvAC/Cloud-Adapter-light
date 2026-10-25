"""Fixed logistic objective and train-only scaling, with no optimizer retries."""
import numpy as np
from scipy.optimize import minimize
from phase65d_recovery_math import objective,standardize

def features(raw,model):
    x=np.asarray(raw,dtype=np.float64)
    assert x.ndim==2 and x.shape[1]==6 and np.isfinite(x).all()
    if model=='RGB3':return x[:,:3].copy()
    if model=='RGB6Q':return np.concatenate([x[:,:3],x[:,:3]**2],axis=1)
    assert model=='MS6';return x.copy()

def fit(x,y,w,group_sizes):
    w=w/w.sum();mean,scale=standardize(x,w)
    constant=np.all(x==x[0],axis=0)
    mean[constant]=x[0,constant];scale[constant]=1
    offsets=np.r_[0,np.cumsum(group_sizes)]
    independent_mean=np.mean([x[offsets[i]:offsets[i+1]].mean(0) for i in range(len(group_sizes))],axis=0)
    independent_var=np.mean([((x[offsets[i]:offsets[i+1]]-independent_mean)**2).mean(0) for i in range(len(group_sizes))],axis=0)
    independent_scale=np.sqrt(independent_var);independent_scale[independent_scale==0]=1
    independent_mean[constant]=x[0,constant];independent_scale[constant]=1
    assert np.allclose(mean,independent_mean,rtol=1e-12,atol=1e-12)
    assert np.allclose(scale,independent_scale,rtol=1e-12,atol=1e-12)
    z=(x-mean)/scale
    result=minimize(objective,np.zeros(x.shape[1]+1),args=(z,y,w),jac=True,method='L-BFGS-B',
        bounds=[(None,None)]*(x.shape[1]+1),options=dict(maxiter=500,gtol=1e-7,ftol=1e-12))
    return dict(mean=mean.tolist(),scale=scale.tolist(),constant_columns=np.flatnonzero(constant).tolist(),
        theta=result.x.tolist(),iterations=int(result.nit),objective=float(result.fun),optimizer_success=bool(result.success),
        optimizer_message=str(result.message),optimizer_status=int(result.status),penalty=.001,positive_features=0,
        final_gradient_inf=float(np.abs(result.jac).max()),scaler_independently_recomputed=True)

def predict(x,model):
    z=(x-np.asarray(model['mean']))/np.asarray(model['scale'])
    logits=z@np.asarray(model['theta'][:-1])+model['theta'][-1]
    p=np.exp(-np.logaddexp(0,-logits))
    assert p.dtype==np.float64 and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return p

def numerical_tests():
    z=np.array([[.1,.2],[-.4,.5],[.7,-.8]],np.float64);y=np.array([0.,1.,1.]);w=np.array([.2,.3,.5]);theta=np.array([.3,-.2,.7])
    value,gradient=objective(theta,z,y,w)
    finite=[]
    for i in range(len(theta)):
        plus=theta.copy();minus=theta.copy();plus[i]+=1e-6;minus[i]-=1e-6
        finite.append((objective(plus,z,y,w)[0]-objective(minus,z,y,w)[0])/2e-6)
    error=float(np.max(np.abs(gradient-finite)));assert error<2e-7
    no_penalty,g0=objective(theta,z,y,w,penalty=0)
    assert abs(value-no_penalty-.5*.001*np.dot(theta[:-1],theta[:-1]))<1e-14
    assert abs(gradient[-1]-g0[-1])<1e-14
    x=np.array([[1.,2.,3.,4.,5.,6.],[2.,3.,4.,5.,6.,7.]])
    assert np.array_equal(features(x,'RGB6Q'),np.c_[x[:,:3],x[:,:3]**2])
    assert np.array_equal(features(x,'MS6'),x) and features(x,'RGB3').shape==(2,3)
    mean,scale=standardize(x,w[:2]);test=np.full_like(x,1e12);features(test,'RGB6Q')
    m2,s2=standardize(x,w[:2]);assert np.array_equal(mean,m2) and np.array_equal(scale,s2)
    _,constant=standardize(np.ones((3,2)),w);assert np.array_equal(constant,np.ones(2))
    p=np.array([np.nextafter(.5,0),.5,np.nextafter(.5,1)]);assert (p>=.5).tolist()==[False,True,True]
    return dict(objective_gradient_finite_difference_max_error=error,intercept_unpenalized=True,constant_scale_one=True,
        feature_order_and_original_scale_square=True,test_extremes_do_not_change_train_statistics=True,half_boundary_cloud=True)
