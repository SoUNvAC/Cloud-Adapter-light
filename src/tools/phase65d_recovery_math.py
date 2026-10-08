"""Fixed group-weighted shallow recovery; no scene/pixel iid significance."""
import numpy as np

def logit_score(x):
    x=np.clip(np.asarray(x,dtype=np.float64),1e-6,1-1e-6)
    return np.log(x)-np.log1p(-x)

def standardize(x,w):
    w=np.asarray(w,dtype=np.float64);w=w/w.sum()
    mean=np.sum(x*w[:,None],axis=0);scale=np.sqrt(np.sum((x-mean)**2*w[:,None],axis=0))
    scale[scale==0]=1
    return mean,scale

def objective(theta,z,y,w,penalty=.001):
    logits=z@theta[:-1]+theta[-1]
    loss=np.sum(w*(np.logaddexp(0,logits)-y*logits))+.5*penalty*np.dot(theta[:-1],theta[:-1])
    # Stable sigmoid, independent of optional SciPy at test time.
    prob=np.exp(-np.logaddexp(0,-logits));res=w*(prob-y)
    return float(loss),np.r_[z.T@res+penalty*theta[:-1],res.sum()]

def fit_model(x,y,w,positive_features):
    from scipy.optimize import minimize
    w=w/w.sum();mean,scale=standardize(x,w);z=(x-mean)/scale
    result=minimize(objective,np.zeros(x.shape[1]+1),args=(z,y,w),jac=True,method='L-BFGS-B',
        bounds=[(0,None) if i<positive_features else (None,None) for i in range(x.shape[1])]+[(None,None)],
        options={'maxiter':500,'gtol':1e-7,'ftol':1e-12})
    if not result.success:raise RuntimeError('Frozen optimizer did not converge: '+str(result.message))
    return dict(mean=mean.tolist(),scale=scale.tolist(),theta=result.x.tolist(),iterations=int(result.nit),
                optimizer_success=True,objective=float(result.fun),penalty=.001,positive_features=positive_features)

def predict_logit(x,model):
    return (((x-np.asarray(model['mean']))/np.asarray(model['scale']))@np.asarray(model['theta'][:-1])+model['theta'][-1]).astype(np.float32)

def group_threshold(negative_arrays,budget=.01):
    """Global >= threshold, full negatives, equal related-group FPR budget."""
    if not negative_arrays or any(not len(a) for a in negative_arrays):raise ValueError('No negative support')
    scores=np.concatenate(negative_arrays).astype(np.float32)
    weights=np.concatenate([np.full(len(a),1/(len(negative_arrays)*len(a)),np.float64) for a in negative_arrays])
    if not np.isfinite(scores).all():raise ValueError('Nonfinite threshold score')
    order=np.argsort(-scores,kind='stable');ss=scores[order]
    ends=np.flatnonzero(np.r_[ss[1:]!=ss[:-1],True]);rate=np.cumsum(weights[order])[ends]
    eligible=np.flatnonzero(rate<=budget)
    if not len(eligible):return float(np.nextafter(ss[0],np.float32(np.inf)))
    return float(ss[ends[eligible[-1]]])

def paired_interval(delta):
    delta=np.asarray(delta,float);rng=np.random.default_rng(65310)
    means=delta[rng.integers(0,len(delta),(2000,len(delta)))].mean(axis=1)
    return dict(n_groups=len(delta),mean=float(delta.mean()),descriptive_95_percentile=np.quantile(means,[.025,.975]).tolist(),
                unit='related-group representative',conditional_on_fitted_model=True,formal_confirmation=False)
