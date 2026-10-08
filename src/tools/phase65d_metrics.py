"""Exact tied-score ranking, conservative FPR thresholds, no pixel iid inference."""
import numpy as np

FPRS=(.001,.005,.01,.02,.05,.10)


def curve(scores,positive,fprs=FPRS,plot_points=2001):
    scores=np.asarray(scores,dtype=np.float32).ravel();positive=np.asarray(positive,dtype=bool).ravel()
    if len(scores)!=len(positive) or not np.isfinite(scores).all():raise ValueError('Invalid scores')
    p=int(positive.sum());n=len(positive)-p
    if not p or not n:return dict(positive=p,negative=n,ap=None,roc_auc=None,matched_fpr={})
    order=np.argsort(-scores,kind='stable');ss=scores[order];yy=positive[order]
    ends=np.flatnonzero(np.r_[ss[1:]!=ss[:-1],True])
    tp=np.cumsum(yy,dtype=np.int64)[ends];fp=ends+1-tp
    recall=tp/p;precision=tp/(ends+1);fpr=fp/n
    ap=float(np.sum(np.diff(np.r_[0,tp])*precision/p))
    rr=np.r_[0,recall];ff=np.r_[0,fpr]
    auc=float(np.sum((rr[1:]+rr[:-1])*.5*np.diff(ff)))
    points={}
    for rate in fprs:
        i=int(np.searchsorted(fp,rate*n,side='right'))-1
        points[str(rate)]=dict(recall=float(recall[i]) if i>=0 else 0.,fpr=float(fpr[i]) if i>=0 else 0.,
                              threshold=float(ss[ends[i]]) if i>=0 else None)
    # Figures use a deterministic subset of thresholds; AP and operating points use all pixels/ties.
    selected=np.unique(np.r_[0,np.searchsorted(recall,np.linspace(0,1,plot_points)).clip(0,len(ends)-1),
                             np.searchsorted(fpr,np.linspace(0,1,plot_points)).clip(0,len(ends)-1),len(ends)-1])
    return dict(positive=p,negative=n,ap=ap,roc_auc=auc,matched_fpr=points,
                precision=np.r_[1.,precision[selected]].tolist(),recall=np.r_[0.,recall[selected]].tolist(),
                fpr=np.r_[0.,fpr[selected]].tolist(),thresholds=[None]+ss[ends[selected]].astype(float).tolist(),
                exact_ties=True,plotted_thresholds=len(selected)+1)


def threshold_at_budget(negative_scores,allowed_fp):
    s=np.asarray(negative_scores,dtype=np.float32).ravel();k=int(allowed_fp)
    if not len(s) or k<0 or k>=len(s):raise ValueError('Invalid FPR budget')
    boundary=np.partition(s,len(s)-k-1)[len(s)-k-1]
    # >= threshold admits only values strictly above the tied negative boundary.
    return float(np.nextafter(boundary,np.float32(np.inf)))


def confusion(truth,pred):
    truth=np.asarray(truth).ravel();pred=np.asarray(pred).ravel();valid=truth!=255
    return np.bincount(truth[valid]*3+pred[valid],minlength=9).reshape(3,3)


def measures(cm):
    cm=np.asarray(cm,dtype=np.int64);p=int(cm[2].sum());n=int(cm[:2].sum())
    tp=int(cm[2,2]);fp=int(cm[:2,2].sum());union=cm.sum(0)+cm.sum(1)-np.diag(cm)
    iou=np.divide(np.diag(cm),union,out=np.full(3,np.nan),where=union>0)
    return dict(confusion=cm.tolist(),positive=p,negative=n,true_positive=tp,false_positive=fp,
                recall=tp/p if p else None,fpr=fp/n if n else None,precision=tp/(tp+fp) if tp+fp else None,
                miou_percent=float(np.nanmean(iou)*100),iou_percent=[float(v*100) if np.isfinite(v) else None for v in iou],
                shadow_to_surface=int(cm[2,0]),shadow_to_cloud=int(cm[2,1]))


def corrected(score,threshold,nonshadow):
    return np.where(np.asarray(score)>=threshold,2,np.asarray(nonshadow)).astype(np.uint8)


def score_summary(values):
    values=np.asarray(values).ravel()
    if not len(values):return dict(n=0,quantiles=None)
    q=np.linspace(0,1,201)
    return dict(n=len(values),mean=float(values.mean(dtype=np.float64)),
                median=float(np.median(values)),q=q.tolist(),quantiles=np.quantile(values,q).tolist())
