"""Six fixed CPU fits and twelve exact-ties val-only thresholds."""
from phase241_ranking import *
import sys,traceback
from fractions import Fraction

def main():
 global cpu
 cpu=CPU('calibrate');assert js(R/'VAL_COMPLETE.json')['passed'];assert not (R/'TEST_STARTED.json').exists();assert not (R/'CALIBRATION_STARTED.json').exists()
 save('CALIBRATION_STARTED.json',{'epoch':time.time(),'PID':os.getpid()});save('KERNEL_SELFTEST.json',selftests())
 from scipy.optimize import minimize
 locks=[];positions=[];rng=np.random.default_rng(24101)
 for domain in ('L8','SPARCS'):
  source=grouped('val',domain,'Source');selected={}
  for scene,rows in source.items():
   sizes=[]
   for v in rows:
    with arrays(v) as (p,y):sizes.append(int((y!=255).sum()))
   total=sum(sizes);chosen=np.sort(rng.choice(total,min(8192,total),replace=False));offset=0;selected[scene]=[]
   for v,size in zip(rows,sizes):
    rank=chosen[(chosen>=offset)&(chosen<offset+size)]-offset;offset+=size
    if not len(rank):continue
    with arrays(v) as (p,y):idx=np.flatnonzero(y.ravel()!=255)[rank];truth=y.ravel()[idx].tolist()
    selected[scene].append({'name':v['name'],'indices':idx.tolist(),'truth':truth,'label_SHA':v['label_SHA']})
   positions.append({'domain':domain,'scene':scene,'valid_population':total,'sample_size':len(chosen),'selection':selected[scene]})
  for model in ('Source','C21701','C21702'):
   rows={v['name']:v for group in grouped('val',domain,model).values() for v in group};samples=[]
   for scene,items in selected.items():
    zs=[];ys=[]
    for item in items:
     v=rows[item['name']];assert item['label_SHA']==v['label_SHA']
     with arrays(v) as (p,y):zs.append(np.log(np.maximum(np.asarray(p).reshape(3,-1)[:,item['indices']].T.astype(np.float64),1e-12)));ys.extend(item['truth'])
    samples.append((np.concatenate(zs),np.asarray(ys,np.int64)))
   def objective(b):
    value=1e-4*np.dot(b,b);grad=2e-4*b
    for z,y in samples:
     a=z+np.r_[0.,b];a-=a.max(1,keepdims=True);q=np.exp(a);den=q.sum(1);value+=float(np.mean(np.log(den)-a[np.arange(len(y)),y]))/len(samples);q/=den[:,None];q[np.arange(len(y)),y]-=1;grad=grad+q[:,1:].mean(0)/len(samples)
    return float(value),grad
   fit=minimize(objective,np.zeros(2),jac=True,method='L-BFGS-B',bounds=[(-4,4),(-4,4)],options={'maxiter':200,'ftol':1e-10,'gtol':1e-8})
   locks.append({'domain':domain,'model':model,'success':bool(fit.success),'bias':[0.,*fit.x.tolist()],'objective':float(fit.fun),'gradient':fit.jac.tolist(),'iterations':int(fit.nit),'evaluations':int(fit.nfev),'status':int(fit.status),'message':str(fit.message),'boundary':[abs(float(x))>=4-1e-10 for x in fit.x]});cpu.check()
 save('CALIBRATION_POSITIONS.json',{'seed':24101,'scene_order':'sorted; one RNG across L8 then SPARCS','samples':positions})
 save('CALIBRATION_LOCK.json',{'fits':locks,'positions_SHA':sha(R/'CALIBRATION_POSITIONS.json'),'val_index_SHA':sha(R/'VAL_SCORE_INDEX.jsonl'),'rule':'scene equal, within scene uniform without replacement; L-BFGS-B fixed; no test'})
 thresholds=[]
 for fit in locks:
  for state in ('raw','calibrated'):
   if state=='calibrated' and not fit['success']:
    thresholds.append({'domain':fit['domain'],'model':fit['model'],'state':state,'success':False,'reason':'fixed bias solver nonconvergent'});continue
   folder=R/'scratch'/f"threshold_{fit['domain']}_{fit['model']}_{state}";folder.mkdir(parents=True,exist_ok=False);runs=[];groups=grouped('val',fit['domain'],fit['model'])
   for si,(scene,rows) in enumerate(groups.items()):
    parts=[]
    for v in rows:
     with arrays(v) as (p,y):
      a=calibrated(p,fit['bias']) if state=='calibrated' else p;parts.append(np.asarray(a[2][(y!=255)&(y!=2)],np.float64))
    s=np.concatenate(parts);del parts
    if len(s):runs.append(sorted_run(folder,str(si),s,np.zeros(len(s),np.uint8)))
    del s;cpu.check()
   threshold=merge(runs,folder,'pooled',True);rates=[];inclusive=[];counts=[]
   for scene,rows in groups.items():
    n=k=inc=0
    for v in rows:
     with arrays(v) as (p,y):
      a=calibrated(p,fit['bias']) if state=='calibrated' else p;s=np.asarray(a[2],np.float64);neg=(y!=255)&(y!=2);n+=int(neg.sum());k+=int(((s>=threshold['threshold'])&neg).sum());inc+=int(((s>=threshold['boundary_score'])&neg).sum())
    if n:rates.append(Fraction(k,n));inclusive.append(Fraction(inc,n));counts.append({'scene':scene,'FP':k,'negative':n,'boundary_inclusive_FP':inc})
   assert sum(rates)/len(rates)<=Fraction(1,100);assert sum(inclusive)/len(inclusive)>Fraction(1,100)
   thresholds.append({'domain':fit['domain'],'model':fit['model'],'state':state,'success':True,**threshold,'full_val_counts':counts,'comparison':'float64 score>=threshold; no test scanning'});clean(folder);cpu.check()
 save('THRESHOLD_LOCK.json',{'thresholds':thresholds,'val_index_SHA':sha(R/'VAL_SCORE_INDEX.jsonl'),'bias_lock_SHA':sha(R/'CALIBRATION_LOCK.json')})
 files=['CALIBRATION_LOCK.json','CALIBRATION_POSITIONS.json','THRESHOLD_LOCK.json','KERNEL_SELFTEST.json']
 files += [p.relative_to(R).as_posix() for p in sorted((R/'code').glob('*')) if p.is_file()]
 save('ALIGNMENT_SEAL.json',{'epoch':time.time(),'files':{n:sha(R/n) for n in files},'test_access_not_started':not (R/'TEST_STARTED.json').exists()});cpu.finish()
if __name__=='__main__':
 try:main()
 except BaseException:
  save('CALIBRATION_FAILURE.json',{'traceback':traceback.format_exc()})
  if 'cpu' in globals():cpu.finish('failed')
  raise
