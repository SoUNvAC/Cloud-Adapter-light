"""Synthetic-only checks run on gzs after definitions are locked."""
from phase320_common import *
import numpy as np
from phase320_staircase import Staircase,TiedAP
from errors_phase320 import locations
def brute(s,y,g,tf,tr):
 ps=[int(((g==i)&y).sum()) for i in range(int(g.max())+1)];ns=[int(((g==i)&~y).sum()) for i in range(len(ps))];out=[]
 for t in ['above_max_score']+sorted(set(map(float,s)),reverse=True):
  p=np.zeros(len(s),bool) if isinstance(t,str) else s>=t;tp=int((p&y).sum());fp=int((p&~y).sum());r=mean([int((p&y&(g==i)).sum())/ps[i] if ps[i] else None for i in range(len(ps))]);f=mean([int((p&~y&(g==i)).sum())/ns[i] if ns[i] else None for i in range(len(ps))]);out.append(dict(t=t,TP=tp,FP=fp,r=r*100,f=f*100))
 a=min([p for p in out if p['f']<=tf],key=lambda p:(-p['r'],p['f'],-(float('inf') if isinstance(p['t'],str) else p['t'])));b=min([p for p in out if p['r']>=tr],key=lambda p:(p['f'],-p['r'],-(float('inf') if isinstance(p['t'],str) else p['t'])));return a,b,ps,ns,out
def main():
 guard();checks=[]
 # Includes a mixed scene, a pure positive scene, a pure negative scene, tied scores across all three.
 s=np.array([.9,.8,.8,.1,.8,.2,.9,.8,.2,.1],np.float32);y=np.array([0,1,1,0,1,1,0,0,0,0],bool);g=np.array([0,0,0,0,1,1,2,2,2,2],np.uint8)
 for tf,tr in [(0,0),(25,50),(50,75),(100,100)]:
  a,b,ps,ns,points=brute(s,y,g,tf,tr);order=np.argsort(s)[::-1];ss=s[order];codes=2*g[order]+y[order].astype(np.uint8)
  for chunk in [1,2,3,7,100]:
   st=Staircase(ps,ns,tf,tr)
   for j in range(0,len(s),chunk):st.push(ss[j:j+chunk],codes[j:j+chunk])
   result=st.finish()
   for key,e in [('CB_original_FPR',a),('CB_original_recall',b)]:
    p=result[key]['selected'];assert p['threshold']==e['t'] and p['TP']==e['TP'] and p['FP']==e['FP'] and abs(p['macro_recall_pct']-e['r'])<1e-10 and abs(p['macro_FPR_pct']-e['f'])<1e-10
   assert result['exact_unique_score_count']==len(set(s.tolist()))
 checks.append(dict(test='exact ties across chunk boundaries, all/none, pure scenes, both selection tie-breaks match brute force',passed=True,cases=20))
 ap=TiedAP(2,4);ap.push(np.array([.9,.8],np.float32),np.array([0,1],bool));ap.push(np.array([.8,.1],np.float32),np.array([1,0],bool));assert abs(ap.finish()-2/3)<1e-12
 assert TiedAP(0,3).finish() is None and TiedAP(3,3).finish() is None;checks.append(dict(test='317 tied AP convention and pure-scene NA',passed=True))
 truth=np.zeros((20,20),np.uint8);truth[5:15,10:]=1;truth[10:15,5:10]=2;valid=np.ones_like(truth,bool);valid[8,8]=False;bins=locations(truth,valid);assert bins[2,6]==4 and bins[3,3]==0 and bins[8,6]==4 and bins[4,10]==1 and bins[14,5]==2 and bins[11,10]==3
 empty=np.zeros((20,20),np.uint8);vv=np.ones_like(empty,bool);vv[10,10]=False;empty[10,10]=1;b=locations(empty,vv);assert b[10,13]==0 and b[10,12]==4
 checks.append(dict(test='two-sided four-neighbor boundaries, Chebyshev2, invalid neighbor excluded, invalid-distance and outermost pixel-distance<=2 context',passed=True))
 write(ROOT/'SYNTHETIC_QA.json',dict(status='passed',tests=checks,real_labels_used=False));print('Phase320 synthetic exact-staircase and boundary QA passed',flush=True)
if __name__=='__main__':main()
