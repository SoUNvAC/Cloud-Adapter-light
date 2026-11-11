"""Binary tied-score AP and ROC AUC without optional dependencies."""
import numpy as np
def roc_auc_score(y,s):
 y=np.asarray(y,dtype=bool);s=np.asarray(s);assert len(y)==len(s) and np.isfinite(s).all()
 pos=int(y.sum());neg=len(y)-pos;assert min(pos,neg)>0
 order=np.argsort(s,kind='mergesort');v=s[order];yy=y[order]
 starts=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1];ends=np.r_[starts[1:],len(v)]
 ranks=(starts+1+ends)*.5;positive_counts=np.add.reduceat(yy.astype(np.int64),starts)
 return float((np.dot(ranks,positive_counts)-pos*(pos+1)*.5)/(pos*neg))
def average_precision_score(y,s):
 y=np.asarray(y,dtype=bool);s=np.asarray(s);assert len(y)==len(s) and np.isfinite(s).all()
 pos=int(y.sum());assert pos>0
 order=np.argsort(-s,kind='mergesort');v=s[order];tp=np.cumsum(y[order],dtype=np.int64)
 ends=np.r_[np.flatnonzero(v[1:]!=v[:-1]),len(v)-1];tp=tp[ends].astype(np.float64)
 return float(np.sum(np.diff(np.r_[0.,tp])/pos*tp/(ends+1)))
def self_test():
 y=np.array([0,0,1,1]);s=np.array([.1,.4,.35,.8]);assert abs(roc_auc_score(y,s)-.75)<1e-14;assert abs(average_precision_score(y,s)-5/6)<1e-14
 assert roc_auc_score([1,0],[1,1])==.5 and average_precision_score([1,0],[1,1])==.5
 rng=np.random.default_rng(24701)
 for _ in range(30):
  y=rng.integers(0,2,100).astype(bool);s=rng.integers(0,7,100)
  p=s[y];n=s[~y];direct=float(np.mean((p[:,None]>n[None,:])+.5*(p[:,None]==n[None,:])))
  assert abs(roc_auc_score(y,s)-direct)<1e-14
  manual=0.;last=0
  for threshold in sorted(set(s),reverse=True):
   selected=s>=threshold;tp=int(y[selected].sum());manual+=(tp-last)/int(y.sum())*tp/int(selected.sum());last=tp
  assert abs(average_precision_score(y,s)-manual)<1e-14
 return {'passed':True,'hand_examples':2,'random_tied_score_cases':30}
