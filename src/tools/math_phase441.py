"""Weighted geometry and scene-only resampling. No sklearn dependency."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import wasserstein_distance

def scene_moments(x,g,groups):
    d=x.shape[1];mu=np.zeros((len(groups),d));second=np.zeros((len(groups),d,d));n=np.zeros(len(groups),int)
    for i,s in enumerate(groups):
        a=x[g==s].astype(np.float64);n[i]=len(a)
        if len(a):mu[i]=a.mean(0);second[i]=a.T@a/len(a)
    return mu,second,n

def combine(mu,second,counts,weights=None):
    w=(counts>0).astype(float) if weights is None else np.asarray(weights)*(counts>0)
    w=w/w.sum(axis=-1,keepdims=True)
    m=w@mu;s=np.einsum('...n,nij->...ij',w,second,optimize=True)-m[...,None]*m[...,None,:]
    return m,(s+np.swapaxes(s,-1,-2))/2

def pca_from_moments(mu,second,n,domains,chosen=None,k=64):
    keep=n>0
    if chosen is not None:keep &= chosen
    w=np.zeros(len(n))
    for d in [0,1]:
        sel=keep&(domains==d);assert sel.any();w[sel]=.5/sel.sum()
    center,cov=combine(mu,second,n,w)
    val,vec=np.linalg.eigh(cov);order=np.argsort(val)[::-1];val=np.maximum(val[order],0);vec=vec[:,order]
    return center,vec[:,:min(k,len(val))],float(val[:k].sum()/max(val.sum(),1e-30)),val

def gaussian(ma,ca,mb,cb):
    va,ua=np.linalg.eigh(ca);root=(ua*np.sqrt(np.maximum(va,0))[...,None,:])@np.swapaxes(ua,-1,-2)
    middle=root@cb@root;middle=(middle+np.swapaxes(middle,-1,-2))/2
    cross=np.sqrt(np.maximum(np.linalg.eigvalsh(middle),0)).sum(-1)
    value=np.square(ma-mb).sum(-1)+np.trace(ca,axis1=-2,axis2=-1)+np.trace(cb,axis1=-2,axis2=-1)-2*cross
    return np.maximum(value,0)

def geometry(ma,ca,mb,cb):
    # Last two mean axes: class, feature. Leading axes may be bootstrap.
    delta=mb-ma;gap=np.linalg.norm(ma[...,None,:,:]-ma[...,:,None,:],axis=-1)
    gap=gap+np.eye(3)*1e100;den=gap.min(-1)
    target=np.linalg.norm(mb[...,None,:,:]-mb[...,:,None,:],axis=-1)+np.eye(3)*1e100
    p=np.zeros(ma.shape[:-1]+(3,))
    for c in range(3):
        for j in range(3):
            if c!=j:
                v=ma[...,j,:]-ma[...,c,:];p[...,c,j]=(delta[...,c,:]*v).sum(-1)/np.maximum(np.square(v).sum(-1),1e-30)
    return dict(F=gaussian(ma,ca,mb,cb),R=np.linalg.norm(delta,axis=-1)/np.maximum(den,1e-15),
        source_gap=den,target_gap=target.min(-1),shift=np.linalg.norm(delta,axis=-1),P=p)

def token_weights(g,domains):
    w=np.zeros(len(g))
    for d in [0,1]:
        ids=np.unique(g[domains==d]);assert len(ids)
        for s in ids:
            sel=g==s;w[sel]=.5/len(ids)/sel.sum()
    return w

def logistic_train(x,y,w,lam=.01):
    def objective(beta):
        z=x@beta[:-1]+beta[-1];loss=np.dot(w,np.logaddexp(0,z)-y*z)+.5*lam*np.dot(beta[:-1],beta[:-1])
        residual=w*(expit(z)-y);grad=np.r_[x.T@residual+lam*beta[:-1],residual.sum()]
        return loss,grad
    fit=minimize(objective,np.zeros(x.shape[1]+1),jac=True,method='L-BFGS-B',options=dict(maxiter=400,ftol=1e-10,gtol=1e-7))
    assert fit.success,fit.message
    return fit.x,dict(iterations=int(fit.nit),success=bool(fit.success),lambda_L2=lam)

def probe_fold(x,y,g,train_groups,test_groups,center,basis):
    train=np.isin(g,train_groups);test=np.isin(g,test_groups);assert not set(train_groups)&set(test_groups)
    a=(x[train]-center)@basis;b=(x[test]-center)@basis;w=token_weights(g[train],y[train])
    mean=w@a;std=np.sqrt(w@np.square(a-mean)).clip(1e-8)
    a=(a-mean)/std;b=(b-mean)/std;beta,meta=logistic_train(a,y[train],w)
    pred=(b@beta[:-1]+beta[-1]>=0).astype(int)
    scores={int(s):float((pred[g[test]==s]==y[test][g[test]==s]).mean()) for s in test_groups}
    return scores,dict(beta=beta,mean=mean,std=std,center=center,basis=basis),meta

def sliced(x,g,groups,domains,counts_boot,directions,bins=512):
    projected=x@directions;lo=projected.min(0);hi=projected.max(0);width=np.maximum(hi-lo,1e-12)/bins
    idx=np.minimum(((projected-lo)/width).astype(int),bins-1)
    hist=np.zeros((len(groups),directions.shape[1],bins));present=np.zeros(len(groups),bool)
    for i,s in enumerate(groups):
        v=idx[g==s]
        if not len(v):continue
        present[i]=True
        for j in range(v.shape[1]):hist[i,j]=np.bincount(v[:,j],minlength=bins)/len(v)
    def aggregate(weights,d):
        w=weights*present*(domains==d);w=w/w.sum(-1,keepdims=True)
        return (w@hist.reshape(len(groups),-1)).reshape(w.shape[:-1]+hist.shape[1:])
    weights=np.ones(len(groups));ha=aggregate(weights,0);hb=aggregate(weights,1)
    point=float((np.abs(np.cumsum(ha-hb,axis=-1)).sum(-1)*width).mean())
    ba=aggregate(counts_boot,0);bb=aggregate(counts_boot,1)
    boot=(np.abs(np.cumsum(ba-bb,axis=-1)).sum(-1)*width).mean(-1)
    td=domains[g];tw=token_weights(g,td)
    exact=float(np.mean([wasserstein_distance(projected[td==0,j],projected[td==1,j],tw[td==0],tw[td==1]) for j in range(directions.shape[1])]))
    bound=float(2*width.mean());assert abs(point-exact)<=bound+1e-8
    return dict(SW1=point,SW1_exact=exact,SW1_bin_error_bound=bound,SW1_CI=np.quantile(boot,[.025,.975]).tolist())

def self_test():
    mu=np.zeros((3,4));cov=np.tile(np.eye(4),(3,1,1))
    assert np.allclose(gaussian(mu,cov,mu,cov),0,atol=1e-10)
    assert np.allclose(gaussian(mu,cov,mu+1,cov),4)
    ma=np.array([[0.,0.],[2.,0.],[0.,3.]])[None];ca=np.tile(np.eye(2),(1,3,1,1));mb=ma.copy();mb[0,0]=[1.,0.]
    out=geometry(ma,ca,mb,ca);assert np.isclose(out['P'][0,0,1],.5) and np.isclose(out['R'][0,0],.5)
    return dict(passed=True,gaussian_identity=True,translation=True,direction_and_ratio=True)
