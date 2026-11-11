"""Reference-local fixed threshold/oracle diagnostics; no model import or fit."""
import argparse,ast,csv,gc,hashlib,io,json,math,os,sys,time,zipfile
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase209/phase209_20261020';P=R/'outputs/phase208/phase208_20261020';sys.path.insert(0,str(R/'outputs/phase202/phase202_20261017/software'))
import numpy as np
import analyze_phase208_local as a
NAMES=('R4','M','B')
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return a.sha(p)
def save(n,d):
 with (O/n).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def csvsave(n,data):
 with (O/n).open('x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def groups(pool):
 a.NAMES=NAMES;a.OLDNAMES=('R4',)
 sealed={'R4':load(a.OLD/'infer/SCORES_SEALED.json')['scores']['R4'],**load(a.D/'infer/SCORES_SEALED.json')['scores']}
 plan=load(a.D/'EVALUATION_INPUT_PLAN.json')['rows'];expected={q['product_id']:q for q in load(a.D/'EXPECTED_SUPPORT.json')['rows']}
 g,sup=(a.cat if pool=='Catalogue' else a.pix)([q for q in plan if q['pool']==pool],sealed,expected)
 return g,sup
def stratified_weights(pos,draws=10000):
 rng=np.random.default_rng(209020);n=len(pos);W=np.zeros((draws,n),np.int64)
 for indices in [np.flatnonzero(pos),np.flatnonzero(~pos)]:
  sampled=rng.integers(len(indices),size=(draws,len(indices)));np.add.at(W,(np.arange(draws)[:,None],indices[sampled]),1)
 return W
def rates(fp,den,W):
 # Stable extended-precision weighted group mean, including zero-positive groups.
 return np.sum(W.astype(np.longdouble)*(np.asarray(fp,np.longdouble)/np.asarray(den,np.longdouble)),axis=1)/W.sum(1)
def select(neg,W,budget):
 """Lowest exact whole-tie operating point on positive float32 score lattice.

 Returns a float64 nextafter threshold, never rounded back to float32.
 All bootstrap rows have their own resampled target and selected threshold.
 """
 den=np.array([len(x) for x in neg],np.int64);assert (den>0).all()
 assert all(x.dtype==np.uint32 and (x[:-1]<=x[1:]).all() for x in neg)
 n=len(W);budget=np.broadcast_to(np.asarray(budget,np.longdouble),(n,));lo=np.zeros(n,np.uint32);hi=np.full(n,max(int(x[-1]) for x in neg),np.uint32)
 def at(b):
  cnt=np.column_stack([len(x)-np.searchsorted(x,b,side='right') for x in neg]);return rates(cnt,den,W),cnt
 while (lo<hi).any():
  mid=lo+(hi-lo)//2;rr,_=at(mid);safe=rr<=budget;hi=np.where(safe,mid,hi);lo=np.where(safe,lo,mid+np.uint32(1))
 b=lo.view(np.float32).astype(np.float64);t=np.nextafter(b,np.inf);rr,cnt=at(lo)
 # For budget1 all negatives may be selected. This edge is covered independently.
 all_allowed=budget>=1;t[all_allowed]=0
 if all_allowed.any():cnt[all_allowed]=den;rr[all_allowed]=1
 assert (rr<=budget).all()
 return t,cnt,np.asarray(rr,np.float64)
def r4_stats(g,t):
 y=g['y']==2;s=g['scores']['R4'].astype(np.float64);pred=s>=np.float64(t);p=int(y.sum());n=int((~y).sum());tp=int((pred&y).sum());fp=int((pred&~y).sum());return dict(positive=p,negative=n,TP=tp,FP=fp,FN=p-tp,TN=n-fp,recall=tp/p if p else None,negative_shadow_fraction=fp/n,precision=tp/(tp+fp) if tp+fp else None,Shadow_IoU=tp/(p+fp) if p+fp else None)
def test():
 neg=[np.sort(np.array(x,np.float32)).view(np.uint32) for x in [[.2,.2,.8],[.1,.8,.8]]];W=np.ones((1,2),np.int64)
 for budget in [0,.1,.5,1]:
  t,c,r=select(neg,W,budget);direct=[int((x.view(np.float32).astype(np.float64)>=t[0]).sum()) for x in neg];assert direct==c[0].tolist();assert sum(d/len(x) for d,x in zip(direct,neg))/2<=budget
  if budget<1:
   boundary=np.nextafter(t[0],-np.inf);unsafe=[int((x.view(np.float32).astype(np.float64)>=boundary).sum()) for x in neg];assert sum(d/len(x) for d,x in zip(unsafe,neg))/2>budget
 assert r4_stats(dict(y=np.array([2,0]),scores={'R4':np.array([.2,.2],np.float32)}),np.nextafter(float(np.float32(.2)),np.inf))['TP']==0
 print('exact ties/float64 boundary/zero-and-all tests passed',flush=True)
def baseline_A():
 start=time.time();support=[];baseline=[];atable=[];arows=[];bootchecks=[]
 for pool in ['Catalogue','PixBox']:
  gg,sup=groups(pool);support+=sup;gids=list(gg);old={(q['strategy'],q['group_id']):q for q in csv.DictReader((P/(pool+'_PER_GROUP.csv')).open(encoding='utf-8-sig'))}
  for gid,g in gg.items():
   for name in NAMES:
    met,mat=a.metric(g['y'],g['pred'][name],g['scores'][name]);previous=old[name,gid]
    for k in ['TP','FP','FN','TN','positive','negative','Shadow_to_Surface','Shadow_to_Cloud']:assert met[k]==int(previous[k])
    assert met['AP'] is None and previous['AP']=='' or abs(met['AP']-float(previous['AP']))<1e-12
    baseline.append(dict(pool=pool,strategy=name,group_id=gid,tile=previous['tile'],products=g['products'],**met))
  pos=np.array([(g['y']==2).any() for g in gg.values()]);den=np.array([(g['y']!=2).sum() for g in gg.values()]);npos=np.array([(g['y']==2).sum() for g in gg.values()]);neg=[np.sort(g['scores']['R4'][g['y']!=2]).view(np.uint32) for g in gg.values()];positive_scores=[np.sort(g['scores']['R4'][g['y']==2]).astype(np.float64) for g in gg.values()];W=stratified_weights(pos)
  for name in ['M','B']:
   fp=np.array([((g['pred'][name]==2)&(g['y']!=2)).sum() for g in gg.values()]);rec=np.array([float(((g['pred'][name]==2)&(g['y']==2)).sum()/p) if p else 0 for p,g in zip(npos,gg.values())]);target=rates(fp,den,np.ones((1,len(gg)),np.int64));t,c,rr=select(neg,np.ones((1,len(gg)),np.int64),target);stats=[r4_stats(g,t[0]) for g in gg.values()]
   assert [q['FP'] for q in stats]==c[0].tolist();rrec=np.mean([q['recall'] for q in stats if q['recall'] is not None]);mrec=float(rec[pos].mean())
   targets=rates(fp,den,W);bt,bc,br=select(neg,W,targets);recall=np.column_stack([(len(s)-np.searchsorted(s,bt,side='left'))/p if p else np.zeros(len(W)) for s,p in zip(positive_scores,npos)]);delta=100*((W[:,pos]*rec[pos]).sum(1)/pos.sum()-(W[:,pos]*recall[:,pos]).sum(1)/pos.sum());CI=np.quantile(delta,[.025,.975]).tolist()
   atable.append(dict(pool=pool,target_source=name,target_mean_negative_pp=float(target[0])*100,R4_actual_mean_negative_pp=rr[0]*100,budget_gap_pp=float(target[0]-rr[0])*100,adapter_default_macro_recall_pp=mrec*100,R4_matched_macro_recall_pp=rrec*100,delta_recall_pp=(mrec-rrec)*100,CI95_pp=CI,R4_max_group_negative_pp=max(q['negative_shadow_fraction'] for q in stats)*100,oracle_threshold=float(t[0]),AP_groups=int(pos.sum()),constraint_groups=len(gg),bootstrap_draws=10000,bootstrap_seed=209020,bootstrap='stratified positive/no-positive groups; recalibrate target and oracle threshold within every replicate',deployment_threshold=False))
   for gid,q in zip(gids,stats):arows.append(dict(pool=pool,target_source=name,group_id=gid,oracle_threshold=t[0],**q))
   # Save enough per-replicate evidence for independent recounts without large arrays.
   csvsave(pool+'_'+name+'_A_BOOTSTRAP.csv',[dict(draw=i,threshold=bt[i],target_mean_negative=float(targets[i]),actual_mean_negative=br[i],delta_recall_pp=delta[i],weights=json.dumps(W[i].tolist())) for i in range(len(W))]);print(pool,name,atable[-1],flush=True)
  del gg,neg,positive_scores,W;gc.collect()
 csvsave('BASELINE_GROUPS.csv',baseline);csvsave('TABLE_A.csv',atable);save('TABLE_A.json',atable);csvsave('A_GROUPS.csv',arows);save('BASELINE_ACCEPTANCE.json',dict(status='passed',reference_local_only=True,original208_all_three_reproduced=True,independent_AP_confusion_checks=a.CHECKS,groups=42,AP_groups=[7,8],support=support,seconds=time.time()-start,code_sha256=sha(Path(__file__)),new_model_calls=0,new_fits=0))
def calibrate():
 start=time.time();d=O/'calibration';manifest=load(d/'DELIVERY_MANIFEST.json');receipt=load(O/'DELIVERY_RECEIPT.json');assert sha(O/'phase209_calibration.zip')==receipt['sha256']
 for n,h in manifest['files'].items():assert sha(d/n)==h
 sealed=load(d/'SCORES_SEALED.json');assert sealed['full53_support'] and sealed['all_parameters_buffers_unchanged'] and sealed['model_weight_sha256']=='7f6e4202e17ee73efa4aba7abb5c34f4f90a9f7eb42480820714994dff2db660'
 groups={q['group']['group_id']:q['group'] for q in load(O/'CALIBRATION_PLAN.json')['rows']};neg=[];data=[]
 with zipfile.ZipFile(R/'data/sentinel2_cloud_mask_catalogue_4172871/masks.zip') as z:
  for q in sealed['rows']:
   g=groups[q['group_id']];p=d/'scores'/(q['product_id']+'.npz');assert sha(p)==q['sha256']
   with np.load(p,allow_pickle=False) as a0:s=a0['scores'].astype(np.float64);flat=a0['flat_indices'];assert a0['valid'].all() and len(flat)==g['common_valid']
   b=z.read('masks/'+q['product_id']+'.npy');assert hashlib.sha256(b).hexdigest()==g['mask_member_sha256'];y=np.load(io.BytesIO(b),allow_pickle=False).reshape(-1,3)[flat];assert (y.sum(1)==1).all();positive=y.argmax(1)==2;assert positive.sum()==g['positive'];neg.append(np.sort(s[~positive]).astype(np.float32).view(np.uint32));data.append((q,s,positive))
 # Use the accepted float64 ordered-lattice implementation from Phase202, with a separate direct count.
 from analyze_phase202_local import exact_threshold
 accepted=exact_threshold([x.view(np.float32).astype(np.float64) for x in neg]);t,fp,fpr=select(neg,np.ones((1,53),np.int64),np.longdouble('.01'));assert t[0]==accepted['threshold'] and fp[0].tolist()==accepted['FP'];counts=[]
 for q,s,y in data:
  n=int((~y).sum());f=int(((s>=t[0])&~y).sum());u=int(((s>=accepted['boundary'])&~y).sum());counts.append(dict(group_id=q['group_id'],product_id=q['product_id'],positive=int(y.sum()),negative=n,FP=f,unsafe_FP=u,FPR=f/n))
 assert math.fsum(q['FPR'] for q in counts)/53<=.01 and math.fsum(q['unsafe_FP']/q['negative'] for q in counts)/53>.01
 csvsave('CALIBRATION_53_COUNTS.csv',counts);save('THRESHOLD_FROZEN.json',dict(threshold=float(t[0]),boundary=accepted['boundary'],macro_negative_fraction=float(fpr[0]),groups=53,rule='lowest feasible global score>=t; whole ties; equal-group mean FPR<=.01; float64 nextafter',counts_sha256=sha(O/'CALIBRATION_53_COUNTS.csv'),score_seal_sha256=sha(d/'SCORES_SEALED.json'),model_weight_sha256=sealed['model_weight_sha256'],eval_labels_not_used_for_this_threshold=True,A_oracle_thresholds_not_used=True,independent_direct_counts_passed=True,seconds=time.time()-start))
 print('unique53-group threshold frozen',float(t[0]),flush=True)
def evaluate_B():
 start=time.time();lock=load(O/'THRESHOLD_FROZEN.json');locksha=sha(O/'THRESHOLD_FROZEN.json');t=lock['threshold'];baseline=list(csv.DictReader((O/'BASELINE_GROUPS.csv').open(encoding='utf-8-sig')));summ=[];allrows=[];diff=[]
 for pool in ['Catalogue','PixBox']:
  gg,sup=groups(pool);per=[]
  for gid,g in gg.items():
   q=r4_stats(g,t);prior=next(r for r in baseline if r['pool']==pool and r['strategy']=='R4' and r['group_id']==gid);ap=float(prior['AP']) if prior['AP'] else None
   per.append(dict(pool=pool,strategy='R4_calibrated',group_id=gid,tile=prior['tile'],products=g['products'],AP=ap,**q))
  for r in baseline:
   if r['pool']!=pool:continue
   nr={k:(None if v=='' else float(v)) for k,v in r.items() if k in ['positive','negative','AP','TP','FP','FN','TN','recall','negative_shadow_fraction','precision']};nr.update({k:r[k] for k in ['pool','strategy','group_id','tile','products']});nr['Shadow_IoU']=nr['TP']/(nr['positive']+nr['FP']) if nr['positive']+nr['FP'] else None;per.append(nr)
  allrows+=per
  for name in ['R4','R4_calibrated','M','B']:
   rr=[r for r in per if r['strategy']==name];fpr=np.array([r['negative_shadow_fraction'] for r in rr]);rng=np.random.default_rng(209020);idx=rng.integers(len(rr),size=(10000,len(rr)));upper=float(np.quantile(fpr[idx].mean(1),.95));p=sum(r['positive'] for r in rr);n=sum(r['negative'] for r in rr);tp=sum(r['TP'] for r in rr);fp=sum(r['FP'] for r in rr)
   priorname='R4' if name=='R4_calibrated' else name;prior=next(r for r in load(P/(pool+'_ALL_METRICS.json')) if r['strategy']==priorname);newap=np.mean([r['AP'] for r in rr if r['AP'] is not None])*100;assert abs(newap-prior['macro_AP_pp'])<1e-10
   summ.append(dict(pool=pool,strategy=name,macro_AP_pp=newap,pooled_AP_pp=prior['pooled_AP_pp'],macro_recall_pp=float(np.mean([r['recall'] for r in rr if r['recall'] is not None]))*100,macro_negative_pp=float(fpr.mean())*100,negative_upper95_pp=upper*100,max_group_negative_pp=float(fpr.max())*100,groups_over1pct=int((fpr>.01).sum()),zero_recall_groups=sum(r['recall']==0 for r in rr),pooled_recall_pp=tp/p*100,pooled_precision_pp=tp/(tp+fp)*100 if tp+fp else None,pooled_negative_pp=fp/n*100,Shadow_IoU_pp=tp/(p+fp)*100 if pool=='Catalogue' else None,Catalogue_three_class_mIoU_pp=prior['Catalogue_pooled_mIoU_pp'] if name!='R4_calibrated' else None,threshold=t if name=='R4_calibrated' else None,mIoU_missing_reason='Cached raw/parent top1 cannot reconstruct nonShadow conditional argmax' if name=='R4_calibrated' else None))
  valid=[gid for gid,g in gg.items() if (g['y']==2).any()];W=stratified_weights(np.array([(g['y']==2).any() for g in gg.values()]));ids=list(gg)
  for name in ['M','B']:
   vals=[]
   for gid in valid:
    left=next(r for r in per if r['strategy']==name and r['group_id']==gid);right=next(r for r in per if r['strategy']=='R4_calibrated' and r['group_id']==gid);vals.append((left['recall']-right['recall'])*100)
   weights=W[:,[ids.index(gid) for gid in valid]];boot=(weights*np.array(vals)).sum(1)/len(valid);diff.append(dict(pool=pool,contrast=name+'-R4_calibrated',mean_delta_recall_pp=float(np.mean(vals)),CI95_pp=np.quantile(boot,[.025,.975]).tolist(),AP_groups=len(valid),bootstrap_seed=209020,bootstrap_draws=10000,conditional_on_frozen_threshold=True))
  del gg;gc.collect()
 assert sha(O/'THRESHOLD_FROZEN.json')==locksha
 csvsave('TABLE_B.csv',summ);save('TABLE_B.json',summ);csvsave('B_GROUPS.csv',allrows);save('B_RECALL_DIFFERENCES.json',diff);save('LOCAL_ANALYSIS_ACCEPTANCE.json',dict(status='passed',threshold_lock_sha256=locksha,AP_threshold_invariant=True,neural_calls_added=0,new_fits=0,reference_only_local=True,seconds=time.time()-start))
 print('B evaluation complete',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['test','baseline_A','calibrate','evaluate_B']);args=p.parse_args();globals()[args.stage]()
