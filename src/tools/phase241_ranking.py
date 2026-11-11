"""Exact external ranking. Only temporary files created here may be removed."""
from phase241_cpu_common import *
import subprocess
TASKS={'Surface_OvR':(0,None),'Cloud_OvR':(1,None),'Shadow_OvR':(2,None),'Shadow_Surface':(2,0),'Cloud_Surface':(1,0),'Shadow_Cloud':(2,1)}

def sorted_run(folder,key,score,label):
 folder.mkdir(parents=True,exist_ok=True);order=np.argsort(score,kind='quicksort')[::-1]
 a=folder/(key+'.f64');b=folder/(key+'.u8');assert not a.exists() and not b.exists()
 np.asarray(score[order],'<f8').tofile(a);np.asarray(label[order],'u1').tofile(b)
 return (a,b,int(np.count_nonzero(label)),int(label.size-np.count_nonzero(label)))

def merge(runs,folder,key,threshold=False):
 m=folder/(key+'.manifest');assert not m.exists()
 m.write_text(''.join(f'{a} {b}'+(f' {P+N}' if threshold else '')+'\n' for a,b,P,N in runs))
 args=[str(R/'code/rank_merge'), 'threshold',str(m)] if threshold else [str(R/'code/rank_merge'),str(m),str(sum(x[2] for x in runs)),str(sum(x[3] for x in runs))]
 return json.loads(subprocess.check_output(args,text=True))

def clean(folder):
 assert folder.parent==R/'scratch' and folder.is_dir()
 # All files are disposable score sorts from the current named operation, never raw caches.
 for p in folder.iterdir():assert p.is_file();p.unlink()
 folder.rmdir()

def score_task(p,y,task):
 pos,neg=TASKS[task];valid=(y!=255) if neg is None else ((y==pos)|(y==neg))
 if neg is None:s=np.asarray(p[pos][valid],np.float64)
 else:s=np.log(np.maximum(np.asarray(p[pos][valid],np.float64),1e-12))-np.log(np.maximum(np.asarray(p[neg][valid],np.float64),1e-12))
 return s,(y[valid]==pos).astype(np.uint8)

def selftests():
 folder=R/'scratch/kernel_selftest';folder.mkdir(parents=True,exist_ok=False);tests=[]
 def brute(s,y):
  P=int(y.sum());N=len(y)-P
  if not P or not N:return None,None
  auc=sum(float(a>b)+.5*float(a==b) for a in s[y==1] for b in s[y==0])/(P*N)
  ap=sum(int(((s==v)&(y==1)).sum())*float(y[s>=v].mean()) for v in np.unique(s))/P
  return ap,auc
 try:
  for i,(s,y) in enumerate([(np.array([.8,.5,.5,.1]),np.array([1,1,0,0],np.uint8)),(np.array([1.,1.,1.]),np.array([0,1,0],np.uint8)),(np.array([0.,1.]),np.array([0,0],np.uint8))]):
   runs=[sorted_run(folder,f'{i}a',s[:2],y[:2]),sorted_run(folder,f'{i}b',s[2:],y[2:])];v=merge(runs,folder,str(i));ap,auc=brute(s,y)
   assert (v['AP'] is None if ap is None else abs(v['AP']-ap)<1e-14);assert (v['AUC'] is None if auc is None else abs(v['AUC']-auc)<1e-14)
   if ap is not None:
    z=merge([sorted_run(folder,f'{i}mono',3*s+2,y)],folder,f'{i}mono');assert abs(z['AP']-ap)<1e-14 and abs(z['AUC']-auc)<1e-14
   tests.append(v)
  # 100 negatives, tied highest 2 scores cannot meet 1%; all reject is required.
  z=merge([sorted_run(folder,'threshold',np.r_[.9,.9,np.full(98,.1)],np.zeros(100,np.uint8))],folder,'threshold',True)
  assert z['threshold']> .9 and z['mean_val_FPR']==0 and z['inclusive_boundary_FPR']==.02
  p=np.array([.2,.3,.5])[:,None,None];assert np.max(abs(calibrated(p,[0,0,0])-p))<1e-15
  return {'passed':True,'exact_ties_and_NA':tests,'threshold_tie':z,'identity':True,'strict_monotone_binary':True}
 finally:clean(folder)
