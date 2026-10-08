"""Group leave-one-out ridge diagnostic with fold-local scaling, no tuning."""
import numpy as np

def loo_ridge(x,y,penalty=10.):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(y)<3 or x.shape[0]!=len(y) or not np.isfinite(x).all():raise ValueError('Invalid group inputs')
    out=[];base=[]
    for i in range(len(y)):
        keep=np.arange(len(y))!=i;train=x[keep];target=y[keep]
        mean=train.mean(0);scale=train.std(0);scale[scale==0]=1
        z=(train-mean)/scale;ym=target.mean()
        beta=np.linalg.solve(z.T@z+penalty*np.eye(x.shape[1]),z.T@(target-ym))
        out.append(float(ym+((x[i]-mean)/scale)@beta));base.append(float(ym))
    return dict(predictions=out,baseline_predictions=base,mae=float(np.abs(np.asarray(out)-y).mean()),
                baseline_mae=float(np.abs(np.asarray(base)-y).mean()),penalty=penalty,
                unit='one frozen related-group representative per fold',scaling='fit-fold only')
