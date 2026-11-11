"""Independent direct-count validation; never calls the main threshold selector."""
import ast,csv,hashlib,io,json,math,sys,time,zipfile
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase209/phase209_20261020';P=R/'outputs/phase208/phase208_20261020'
import analyze_phase209 as main
COUNT=0
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(n):
 with (O/n).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def check(v):
 global COUNT
 assert v
 COUNT+=1
def sha(p):return main.sha(p)
def save(n,d):
 with (O/n).open('x',encoding='utf8') as f:json.dump(d,f,indent=2,allow_nan=False)
def validate():
 start=time.time();check(load(O/'BASELINE_ACCEPTANCE.json')['status']=='passed');check(load(O/'LOCAL_ANALYSIS_ACCEPTANCE.json')['status']=='passed');threshold=load(O/'THRESHOLD_FROZEN.json');t=threshold['threshold'];baseline=rows('BASELINE_GROUPS.csv');B=rows('B_GROUPS.csv');A=load(O/'TABLE_A.json');btables=load(O/'TABLE_B.json');bdiffs=load(O/'B_RECALL_DIFFERENCES.json')
 # Independent mini-cases, including a threshold unrepresentable in float32.
 x=np.array([.1,.2,.2,.8],np.float32).astype(np.float64);border=float(np.float32(.2));after=np.nextafter(border,np.inf);check((x>=after).tolist()==[False,False,False,True]);check((x>=border).tolist()==[False,True,True,True]);check((x>=0).all() and not (x>=1).any())
 for pool in ['Catalogue','PixBox']:
  gg,sup=main.groups(pool);gids=list(gg);pos=np.array([(g['y']==2).any() for g in gg.values()]);pden=np.array([(g['y']==2).sum() for g in gg.values()]);nden=np.array([(g['y']!=2).sum() for g in gg.values()]);n=len(gg);posids=np.flatnonzero(pos);noids=np.flatnonzero(~pos)
  # Build our own deterministic stratified sampling schedule.
  rng=np.random.default_rng(209020);W=np.zeros((10000,n),np.int64)
  for ids in [posids,noids]:
   idsamples=rng.integers(len(ids),size=(10000,len(ids)))
   for col in range(len(ids)):W[np.arange(10000),ids[idsamples[:,col]]]+=1
  for name in ['M','B']:
   trace=rows(pool+'_'+name+'_A_BOOTSTRAP.csv');check(len(trace)==10000);weights=np.array([json.loads(q['weights']) for q in trace]);check(np.array_equal(weights,W));th=np.array([float(q['threshold']) for q in trace]);previous=np.nextafter(th,-np.inf);ach=np.zeros(10000,np.longdouble);unsafe=np.zeros(10000,np.longdouble);target=np.zeros(10000,np.longdouble);rec=np.zeros(10000);adapter=np.zeros(10000)
   for i,g in enumerate(gg.values()):
    y=g['y']==2;s=g['scores']['R4'].astype(np.float64);sn=np.sort(s[~y]);sp=np.sort(s[y]);fp=int(((g['pred'][name]==2)&~y).sum());tp=int(((g['pred'][name]==2)&y).sum());wc=weights[:,i].astype(np.longdouble)
    c=len(sn)-np.searchsorted(sn,th,'left');cu=len(sn)-np.searchsorted(sn,previous,'left');ach+=wc*c/len(sn)/n;unsafe+=wc*cu/len(sn)/n;target+=wc*fp/len(sn)/n
    if len(sp):
     cpos=len(sp)-np.searchsorted(sp,th,'left');rec+=weights[:,i]*cpos/len(sp)/len(posids);adapter+=weights[:,i]*tp/len(sp)/len(posids)
   check((ach<=target+np.longdouble('1e-17')).all());check(((unsafe>target-np.longdouble('1e-17'))|(target>=1)).all());check(np.max(np.abs(np.asarray(ach,float)-[float(q['actual_mean_negative']) for q in trace]))<1e-12);check(np.max(np.abs(np.asarray(target,float)-[float(q['target_mean_negative']) for q in trace]))<1e-12)
   delta=100*(adapter-rec);check(np.max(np.abs(delta-[float(q['delta_recall_pp']) for q in trace]))<1e-10);q=next(q for q in A if q['pool']==pool and q['target_source']==name);check(np.max(np.abs(np.quantile(delta,[.025,.975])-q['CI95_pp']))<1e-10)
   target0=math.fsum(float(((g['pred'][name]==2)&(g['y']!=2)).sum())/int((g['y']!=2).sum()) for g in gg.values())/n;oth=q['oracle_threshold'];fp0=[];rec0=[];unsafe0=[]
   for g in gg.values():
    y=g['y']==2;s=g['scores']['R4'].astype(np.float64);fp0.append(int(((s>=oth)&~y).sum())/int((~y).sum()));unsafe0.append(int(((s>=np.nextafter(oth,-np.inf))&~y).sum())/int((~y).sum()))
    if y.any():rec0.append(int(((s>=oth)&y).sum())/int(y.sum()))
   check(math.fsum(fp0)/n<=target0+1e-16 and math.fsum(unsafe0)/n>target0-1e-16);check(abs(np.mean(rec0)*100-q['R4_matched_macro_recall_pp'])<1e-10)
  for r in B:
   if r['pool']!=pool:continue
   g=gg[r['group_id']];y=g['y']==2;pred=(g['scores']['R4'].astype(np.float64)>=t) if r['strategy']=='R4_calibrated' else g['pred'][r['strategy']]==2
   tp=int((pred&y).sum());fp=int((pred&~y).sum());check(tp==float(r['TP']) and fp==float(r['FP']));check(float(r['negative'])==int((~y).sum()) and float(r['positive'])==int(y.sum()));check(r['AP']=='' if not y.any() else abs(float(r['AP'])-float(next(b['AP'] for b in baseline if b['pool']==pool and b['strategy']==('R4' if r['strategy']=='R4_calibrated' else r['strategy']) and b['group_id']==r['group_id'])))<1e-12)
  for s in [q for q in btables if q['pool']==pool]:
   rr=[r for r in B if r['pool']==pool and r['strategy']==s['strategy']];fpr=np.array([float(r['FP'])/float(r['negative']) for r in rr]);rng=np.random.default_rng(209020);samples=rng.integers(n,size=(10000,n));upper=np.quantile(np.sum(fpr[samples],axis=1)/n,.95);check(abs(upper*100-s['negative_upper95_pp'])<1e-10);check(abs(math.fsum(fpr)/n*100-s['macro_negative_pp'])<1e-10);check(s['groups_over1pct']==sum(x>.01 for x in fpr))
   if s['strategy']=='R4_calibrated':check(s['Catalogue_three_class_mIoU_pp'] is None)
  for q in [q for q in bdiffs if q['pool']==pool]:
   name=q['contrast'].split('-')[0];v=[]
   for gid in gids:
    left=next(r for r in B if r['pool']==pool and r['strategy']==name and r['group_id']==gid);right=next(r for r in B if r['pool']==pool and r['strategy']=='R4_calibrated' and r['group_id']==gid)
    v.append((float(left['recall'])-float(right['recall']))*100 if left['recall'] else 0)
   boot=np.sum(W[:,pos]*np.array(v)[pos],axis=1)/pos.sum();check(np.max(abs(np.quantile(boot,[.025,.975])-q['CI95_pp']))<1e-10)
  del gg
 # Direct full53 count and membership check independent of threshold-search code.
 plan=load(O/'CALIBRATION_PLAN.json');check(plan['zero_group_overlap'] and len(plan['calibration_groups'])==53);check(not(set(plan['calibration_groups'])&set(plan['fit_groups'])) and not(set(plan['calibration_groups'])&set(plan['development_groups'])))
 d=O/'calibration';seal=load(d/'SCORES_SEALED.json');cal=[];direct=[];unsafe=[]
 check(len(seal['rows'])==53 and {q['group_id'] for q in seal['rows']}==set(plan['calibration_groups']) and len({q['product_id'] for q in seal['rows']})==53)
 with zipfile.ZipFile(R/'data/sentinel2_cloud_mask_catalogue_4172871/masks.zip') as z:
  for q in seal['rows']:
   p=d/'scores'/(q['product_id']+'.npz');check(sha(p)==q['sha256'])
   with np.load(p,allow_pickle=False) as v:s=v['scores'].astype(np.float64);flat=v['flat_indices'];check(v['valid'].all())
   y=np.load(io.BytesIO(z.read('masks/'+q['product_id']+'.npy')),allow_pickle=False).reshape(-1,3)[flat].argmax(1)==2;direct.append(int(((s>=t)&~y).sum())/int((~y).sum()));unsafe.append(int(((s>=threshold['boundary'])&~y).sum())/int((~y).sum()))
 check(math.fsum(direct)/53<=.01 and math.fsum(unsafe)/53>.01)
 events=[json.loads(s) for s in (d/'CALL_LEDGER.jsonl').read_text().splitlines()];check([q['call'] for q in events]==list(range(1,len(events)+1)));check(sum(q['kind']=='calibration_R4' for q in events)==212);check(len(events)==seal['total_calls']<=4000);status=load(d/'RUN_STATUS.json');check(status['seconds']<=7200 and status['cpu_peak_bytes']<=16*1024**3 and status['gpu_peak_bytes']<=20*1024**3 and status['minimum_host_free_bytes']>=4000000000);check(seal['model_state_before']==seal['model_state_after'] and seal['all_eval'])
 save('INDEPENDENT_ACCEPTANCE.json',dict(status='passed',checks=COUNT,seconds=time.time()-start,code_sha256=sha(Path(__file__)),independent_A_all40000_resampled_thresholds_recounted=True,whole_tie_minimality_checked=True,independent_B_confusion_counts=True,AP_invariance=True,neural_calls_added=0,new_fits=0))
 print(json.dumps(dict(status='passed',checks=COUNT,seconds=time.time()-start)),flush=True)
if __name__=='__main__':validate()
