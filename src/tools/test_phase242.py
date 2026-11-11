"""Small-array exactness tests, no experimental references or neural calls."""
from phase242_common import *
import sys
def brute(p,n):
 if not len(p) or not len(n):return None,None
 auc=sum(float(a>b)+.5*float(a==b) for a in p for b in n)/(len(p)*len(n));s=np.r_[p,n];y=np.r_[np.ones(len(p)),np.zeros(len(n))];ap=sum(np.sum((s==v)&(y==1))*y[s>=v].mean() for v in np.unique(s))/len(p);return float(ap),float(auc)
def main():
 from analyze_phase242 import matrix_summary,cm,close,decision
 folder=R/'kernel_test';assert not folder.exists();folder.mkdir(parents=True);runs={};truth={};checks=[]
 data={'a':(np.array([1.,.5,.5]),np.array([.5,-1.])),'b':(np.array([.8,-.2]),np.array([1.,.5,.1])),'c':(np.array([],np.float64),np.array([0.,0.]))}
 for name,(p,n) in data.items():
  runs[name]={}
  for label,a in ((1,p),(0,n)):
   fp=folder/(name+str(label)+'.f64');sortedfile(fp,a);runs[name][label]=fp
 cells=[]
 for i,(p,_) in data.items():
  for j,(_,n) in data.items():
   a=ranked([(runs[i][1],1,0),(runs[j][0],0,0)]);ap,auc=brute(p,n);assert close(a['AUC'],auc) and close(a['AP'],ap);checks.append('directed ties/NA '+i+j);cells.append({'positive_scene':i,'negative_scene':j,**a})
 source=matrix_summary(cells,list(data));allp=np.concatenate([v[0] for v in data.values()]);alln=np.concatenate([v[1] for v in data.values()]);assert close(source['pooled_AUC'],brute(allp,alln)[1]);checks.append('weighted directed matrix equals pooled')
 for i,(p,n) in data.items():
  for shift in (.7,-2.,0.):
   a=ranked([(runs[i][1],1,shift),(runs[i][0],0,shift)]);ap,auc=brute(p,n);assert close(a['AP'],ap) and close(a['AUC'],auc)
 checks.append('J0 global and J1 diagonal translation invariance')
 ss=brute(data['a'][0],data['b'][1])[1];cs=brute(data['a'][0]+.4,data['b'][1])[1];sc=brute(data['a'][0],data['b'][1]-.3)[1];cc=brute(data['a'][0]+.4,data['b'][1]-.3)[1];sp=.5*((cs-ss)+(cc-sc));sn=.5*((sc-ss)+(cc-cs));assert abs(sp+sn-(cc-ss))<1e-15;checks.append('four-score-source symmetric decomposition')
 y=np.array([0,0,1,1,2,2],np.uint8);ps=np.array([0,1,1,0,2,0]);pc=np.array([1,0,1,2,0,2]);t=np.bincount(9*y.astype(np.int64)+3*ps+pc,minlength=27).reshape(3,3,3);assert np.array_equal(t.sum(2),cm(y,ps)) and np.array_equal(t.sum(1),cm(y,pc));checks.append('27-cell matched-pixel transitions')
 p=np.array([[.7,.1,.2],[.2,.2,.7],[.1,.7,.1]],np.float32);yy=np.array([0,2,1],np.uint8);a=decision(p,yy,0,.5,True);b=decision(p,yy,0,.5,False);assert a['argmax']['confusion']==b['argmax']['confusion'] and a['frozen_threshold']['confusion']==b['frozen_threshold']['confusion'];checks.append('decision identity, independent argmax and fixed threshold')
 save('KERNEL_ACCEPTANCE.json',{'passed':True,'checks':checks,'network_calls':0,'new_thresholds':0});print(json.dumps(js(R/'KERNEL_ACCEPTANCE.json')))
if __name__=='__main__':main()
