"""Exact finite float32 tie blocks with bounded-memory cumulative statistics."""
import numpy as np
class TiedAP:
 def __init__(self,positive,total):self.P=positive;self.N=total;self.tp=0;self.seen=0;self.area=0.;self.pending=None
 def flush(self,p,n):self.tp+=int(p);self.seen+=int(n);self.area+=int(p)*self.tp/self.seen
 def push(self,s,y):
  if not len(s) or not self.P or self.P==self.N:return
  ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1];nums=np.diff(np.r_[-1,ends]);pos=np.diff(np.r_[0,np.cumsum(y,dtype=np.int64)[ends]])
  if self.pending is not None:
   value,p,n=self.pending
   if value==s[0]:pos[0]+=p;nums[0]+=n
   else:self.flush(p,n)
  if len(ends)>1:
   pp=pos[:-1];nn=nums[:-1];self.area+=float(np.sum(pp*(self.tp+np.cumsum(pp))/(self.seen+np.cumsum(nn))));self.tp+=int(pp.sum());self.seen+=int(nn.sum())
  self.pending=(float(s[-1]),int(pos[-1]),int(nums[-1]))
 def finish(self):
  if not self.P or self.P==self.N:return None
  assert self.pending is not None;self.flush(self.pending[1],self.pending[2]);assert self.tp==self.P and self.seen==self.N;return self.area/self.P
class Staircase:
 def __init__(self,positives,negatives,target_FPR,target_recall):
  self.ps=np.asarray(positives,np.int64);self.ns=np.asarray(negatives,np.int64);self.P=int(self.ps.sum());self.Neg=int(self.ns.sum());self.np=int((self.ps>0).sum());self.nn=int((self.ns>0).sum());assert self.np and self.nn
  self.wp=np.zeros(2*len(self.ps));self.wn=self.wp.copy()
  for i in range(len(self.ps)):
   if self.ps[i]:self.wp[2*i+1]=1/(self.np*int(self.ps[i]))
   if self.ns[i]:self.wn[2*i]=1/(self.nn*int(self.ns[i]))
  self.tf=target_FPR/100;self.tr=target_recall/100;self.tp=self.fp=self.unique=0;self.r=self.f=0.;self.pending=None;self.maximum=None;self.minimum=None
  self.none=self.point('above_max_score',0,0,0.,0.,0,0);self.bestF=self.none;self.bestR=self.none if self.tr<=0 else None;self.lowerF=self.none;self.upperF=None;self.lowerR=self.none if self.tr>0 else None;self.upperR=self.none if self.tr<=0 else None
 def point(self,t,tp,fp,r,f,tie,index):return dict(threshold=t,selected_pixels=int(tp+fp),tie_block_pixels=int(tie),TP=int(tp),FP=int(fp),FN=self.P-int(tp),TN=self.Neg-int(fp),macro_recall_pct=100*float(r),macro_FPR_pct=100*float(f),pooled_recall_pct=100*int(tp)/self.P,pooled_FPR_pct=100*int(fp)/self.Neg,precision_pct=100*int(tp)/(int(tp)+int(fp)) if tp+fp else None,Shadow_IoU_pct=100*int(tp)/(self.P+int(fp)),three_class_mIoU_pct=None,staircase_index=int(index))
 def push(self,s,codes):
  if not len(s):return
  assert s.dtype==np.float32 and np.isfinite(s).all() and np.all(s[:-1]>=s[1:]);assert self.minimum is None or s[0]<=self.minimum
  if self.maximum is None:self.maximum=float(s[0])
  self.minimum=float(s[-1]);ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1];starts=np.r_[0,ends[:-1]+1];nums=np.diff(np.r_[-1,ends]);pos=np.add.reduceat((codes&1).astype(np.int64),starts);wp=np.add.reduceat(self.wp[codes],starts);wn=np.add.reduceat(self.wn[codes],starts);values=s[ends]
  if self.pending is not None:
   value,p,n,r,f=self.pending
   if value==float(s[0]):pos[0]+=p;nums[0]+=n;wp[0]+=r;wn[0]+=f
   else:self.consume(np.array([value]),np.array([p],np.int64),np.array([n],np.int64),np.array([r]),np.array([f]))
  if len(ends)>1:self.consume(values[:-1],pos[:-1],nums[:-1],wp[:-1],wn[:-1])
  self.pending=(float(values[-1]),int(pos[-1]),int(nums[-1]),float(wp[-1]),float(wn[-1]))
 def consume(self,score,pos,num,wp,wn):
  tp=self.tp+np.cumsum(pos);fp=self.fp+np.cumsum(num-pos);r=self.r+np.cumsum(wp);f=self.f+np.cumsum(wn);r[tp==self.P]=1.;f[fp==self.Neg]=1.
  assert np.all(np.diff(tp)>=0) and np.all(np.diff(fp)>=0) and np.all(np.diff(r)>=0) and np.all(np.diff(f)>=0)
  def at(i):return self.point(float(score[i]),int(tp[i]),int(fp[i]),float(r[i]),float(f[i]),int(num[i]),self.unique+i+1)
  i=int(np.searchsorted(f,self.tf,side='right'))-1
  if i>=0:
   self.lowerF=at(i);j=int(np.searchsorted(r,r[i],side='left'));candidate=at(j)
   if (candidate['macro_recall_pct'],-candidate['macro_FPR_pct'])>(self.bestF['macro_recall_pct'],-self.bestF['macro_FPR_pct']):self.bestF=candidate
  if i+1<len(f) and self.upperF is None:self.upperF=at(i+1)
  j=int(np.searchsorted(r,self.tr,side='left'))
  if j>0:self.lowerR=at(j-1)
  if j<len(r):
   if self.upperR is None:self.upperR=at(j)
   k=int(np.searchsorted(f,f[j],side='right'))-1;candidate=at(k)
   if self.bestR is None or (candidate['macro_FPR_pct'],-candidate['macro_recall_pct'])<(self.bestR['macro_FPR_pct'],-self.bestR['macro_recall_pct']):self.bestR=candidate
  self.tp=int(tp[-1]);self.fp=int(fp[-1]);self.r=float(r[-1]);self.f=float(f[-1]);self.unique+=len(score)
 def finish(self):
  value,p,n,r,f=self.pending;self.consume(np.array([value]),np.array([p]),np.array([n]),np.array([r]),np.array([f]));assert self.tp==self.P and self.fp==self.Neg and abs(self.r-1)<1e-9 and abs(self.f-1)<1e-9
  return dict(exact_unique_score_count=self.unique,score_max=self.maximum,score_min=self.minimum,CB_original_FPR=dict(selected=self.bestF,lower_neighbor=self.lowerF,upper_neighbor=self.upperF),CB_original_recall=dict(selected=self.bestR,lower_neighbor=self.lowerR,upper_neighbor=self.upperR),all_selected_endpoint=dict(threshold=self.minimum,TP=self.tp,FP=self.fp),none_selected_endpoint=self.none)
