"""Full-order noninterpolated AP and tie-aware ROC AUC. No threshold fitting."""
import numpy as np
def primary(y,s):
 y=np.asarray(y,dtype=np.uint8);s=np.asarray(s);assert y.shape==s.shape and np.isfinite(s).all() and np.isin(y,[0,1]).all()
 P=int(y.sum());N=len(y)-P
 if not P or not N:return {'positive':P,'negative':N,'AUC':None,'AP':None}
 order=np.argsort(-s,kind='stable');v=s[order];t=y[order].astype(np.int64);ends=np.r_[np.flatnonzero(v[1:]!=v[:-1]),len(v)-1]
 tp=np.cumsum(t,dtype=np.int64)[ends];fp=ends+1-tp;dp=np.diff(np.r_[0,tp]);df=np.diff(np.r_[0,fp]);auc=float(np.sum(df*(tp+np.r_[0,tp[:-1]]))/(2.0*P*N));ap=float(np.sum(dp*(tp/(ends+1)))/P)
 return {'positive':P,'negative':N,'AUC':auc,'AP':ap,'distinct_score_groups':len(ends),'pixels_in_rank_ties':int(np.diff(np.r_[-1,ends])[np.diff(np.r_[-1,ends])>1].sum())}
def independent(y,s):
 """Ascending tie blocks: Mann-Whitney AUC and suffix precision AP."""
 y=np.asarray(y,dtype=np.uint8);s=np.asarray(s);P=int(np.count_nonzero(y));N=len(y)-P
 if not P or not N:return None,None
 ix=np.argsort(s,kind='stable');ss=s[ix];yy=y[ix].astype(np.int64);starts=np.r_[0,np.flatnonzero(ss[1:]!=ss[:-1])+1];sizes=np.diff(np.r_[starts,len(ss)]);positive=np.add.reduceat(yy,starts);negative=sizes-positive
 below_negative=np.cumsum(negative,dtype=np.int64)-negative
 auc=float(np.dot(positive.astype(np.float64),below_negative+.5*negative)/(P*N))
 suffix_positive=np.cumsum(positive[::-1],dtype=np.int64)[::-1];suffix_total=np.cumsum(sizes[::-1],dtype=np.int64)[::-1]
 ap=float(np.dot(positive.astype(np.float64),suffix_positive/suffix_total)/P)
 return auc,ap
def handchecks():
 cases=[([1,0],[.5,.5],.5,.5),([1,0,1],[.7,.7,.2],.25,7/12),([1,0,1,0],[.9,.8,.8,.1],.875,5/6),([0,1],[0,1],1,1),([0,1],[1,0],0,.5)]
 for y,s,a,p in cases:
  x=primary(y,s);b,q=independent(y,s);assert abs(x['AUC']-a)<1e-14 and abs(x['AP']-p)<1e-14 and abs(b-a)<1e-14 and abs(q-p)<1e-14
 for y in [[1,1],[0,0]]:
  x=primary(y,[.4,.4]);assert x['AUC'] is None and x['AP'] is None and independent(y,[.4,.4])==(None,None)
 return {'manual_cases':len(cases),'one_class_NA_cases':2,'ROC_ties':'half credit','AP':'noninterpolated exact tie blocks; not trapezoidal PR area','passed':True}
if __name__=='__main__':print(handchecks())
