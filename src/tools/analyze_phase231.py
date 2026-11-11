"""Only after all three train-only biases sealed and independently checked."""
from pathlib import Path
import csv,hashlib,json,time
import numpy as np
from phase230_metrics import primary
from phase231_metrics import metrics,summary
from analyze_phase230 import cpu_peak
REPO=Path(__file__).resolve().parents[2];R=REPO/'outputs/phase231/phase231_20261026';D=R/'delivery';OLD=REPO/'outputs/phase230/phase230_20261026';START=time.monotonic()
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(R/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvout(n,rows):
 keys=list(dict.fromkeys(k for v in rows for k in v))
 with (R/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for x in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in x.items()})
def main():
 accepted=js(R/'BIAS_SEAL_ACCEPTED_LOCAL.json');assert accepted['status']=='passed';frozen=js(D/'BIAS_FROZEN.json');assert sha(D/'BIAS_FROZEN.json')==accepted['sealed_bias_SHA'];assert time.time()>accepted['epoch']>frozen['epoch_sealed']
 save('VALIDATION_ACCESS_STARTED.json',{'epoch':time.time(),'BIAS_FROZEN_SHA':sha(D/'BIAS_FROZEN.json'),'BIAS_SEAL_ACCEPTED_LOCAL_SHA':sha(R/'BIAS_SEAL_ACCEPTED_LOCAL.json'),'all_three_sealed_before_validation_arrays':True,'independent_solver_replay_may_follow_validation_without_altering_sealed_biases':True})
 lock=js(R/'LOCK.json')
 for n,h in lock['identities'].items():assert sha(REPO/n)==h
 receipts=js(OLD/'delivery/SCORE_RECEIPTS.json');fitmap={v['model']:v for v in frozen['biases']};table1=[];scene_rows=[];table2=[];rank=[];flows=[];transitions=[];used=[]
 for fit in frozen['biases']:table1.append({k:v for k,v in fit.items() if k not in ['trace','scene_support']})
 for model in lock['models_to_calibrate']:
  b=fitmap[model]['b'];expb=float(np.exp(np.float64(b)));pool_pre=[];pool_post=[]
  for rec in [v for v in receipts if v['model']==model]:
   assert time.monotonic()-START<3600 and cpu_peak()<8*1024**3
   pp=OLD/'delivery'/rec['score_path'];yp=OLD/'delivery'/rec['label_path'];assert sha(pp)==rec['score_SHA'] and sha(yp)==rec['label_SHA'];used.append({'model':model,'scene':rec['scene'],'score_path':str(pp),'score_SHA':rec['score_SHA'],'label_path':str(yp),'label_SHA':rec['label_SHA']});p=np.load(pp,allow_pickle=False);y=np.load(yp,allow_pickle=False);valid=y!=255;before=p.argmax(0).astype(np.uint8);assert hashlib.sha256(before.tobytes()).hexdigest()==rec['prediction_SHA']
   q=p.astype(np.float64);q[1]*=expb;after=q.argmax(0).astype(np.uint8);zero=p.astype(np.float64);zero[1]*=np.exp(np.float64(0));assert np.array_equal(zero.argmax(0),before)
   pre=np.bincount(3*y[valid].astype(np.int64)+before[valid],minlength=9).reshape(3,3);post=np.bincount(3*y[valid].astype(np.int64)+after[valid],minlength=9).reshape(3,3);assert pre.tolist()==rec['confusion']
   for mode,c,pred in [('raw',pre,before),('calibrated',post,after)]:scene_rows.append({'model':model,'scene':rec['scene'],'decision':mode,'b':0 if mode=='raw' else b,'prediction_SHA':hashlib.sha256(pred.tobytes()).hexdigest(),**metrics(c)})
   mask=(y==0)|(y==1);labels=(y[mask]==1).astype(np.uint8);d=(np.log(np.maximum(p[1],1e-12))-np.log(np.maximum(p[0],1e-12)))[mask].astype(np.float64);db=d+b;a=primary(labels,d);z=primary(labels,db)
   assert a['AUC']==z['AUC'] and a['AP']==z['AP'];unique=np.unique(d);assert np.all(np.diff(unique+b)>0)
   rank.append({'model':model,'scene':rec['scene'],'Cloud_positive':a['positive'],'Surface_negative':a['negative'],'AUC_raw':a['AUC'],'AUC_calibrated':z['AUC'],'AP_raw':a['AP'],'AP_calibrated':z['AP'],'exact_sorting_metric_invariant':True,'rank_support':'Cloud/Surface only','shift_arithmetic':'d(float32) promoted tofloat64 then b added','binary_Cloud_recall_raw':float((d[labels==1]>0).mean()) if a['positive'] else None,'binary_Cloud_recall_calibrated':float((db[labels==1]>0).mean()) if a['positive'] else None});pool_pre.append((labels,d));pool_post.append(db)
   delta=post-pre;flow={'model':model,'scene':rec['scene'],'confusion_net_delta':delta.tolist(),'Cloud_reference':int(pre[1].sum()),'Shadow_reference':int(pre[2].sum())}
   for i,c1 in enumerate(['Surface','Cloud','Shadow']):
    for j,c2 in enumerate(['Surface','Cloud','Shadow']):
     if i!=j:flow[c1+'_to_'+c2+'_net_delta']=int(delta[i,j])
   flows.append(flow);J=np.bincount(9*y[valid].astype(np.int64)+3*before[valid].astype(np.int64)+after[valid],minlength=27).reshape(3,3,3);assert np.array_equal(J.sum(2),pre) and np.array_equal(J.sum(1),post);transitions.append({'model':model,'scene':rec['scene'],'reference_raw_calibrated_J':J.tolist(),'raw_to_calibrated_predictions':J.sum(0).tolist(),'definition':'Direct same-pixel transition counts; distinct from net confusion differences. No per-pixel prediction arrays persisted.'})
  ypool=np.concatenate([v[0] for v in pool_pre]);dpool=np.concatenate([v[1] for v in pool_pre]);postpool=np.concatenate(pool_post);a=primary(ypool,dpool);z=primary(ypool,postpool);assert a['AUC']==z['AUC'] and a['AP']==z['AP'];rank.append({'model':model,'scene':'POOLED','Cloud_positive':a['positive'],'Surface_negative':a['negative'],'AUC_raw':a['AUC'],'AUC_calibrated':z['AUC'],'AP_raw':a['AP'],'AP_calibrated':z['AP'],'exact_sorting_metric_invariant':True})
  for mode in ['raw','calibrated']:
   per=[v for v in scene_rows if v['model']==model and v['decision']==mode];table2.append({'model':model,'decision':mode,'b':b if mode=='calibrated' else 0,'scenes':10,**summary(per),'AUC_AP_pooled':{'AUC':a['AUC'],'AP':a['AP']},'ranking_invariant':True})
  del pool_pre,pool_post,ypool,dpool,postpool
 # Historical references use confusion counts only, not their score/label caches.
 oldentry=js(OLD/'DELIVERY_MANIFEST.json')['THREE_CLASS_CONFUSIONS.json'];assert sha(OLD/'THREE_CLASS_CONFUSIONS.json')==oldentry['SHA'];oldcm=js(OLD/'THREE_CLASS_CONFUSIONS.json')
 for model in ['SPARCS_21701','SPARCS_21702']:
  per=[metrics(v['confusion']) for v in oldcm if v['model']==model];assert len(per)==10;table2.append({'model':model,'decision':'historical_uncalibrated_reference','b':0,'scenes':10,**summary(per),'not_equivalent_supervision_budget':True})
 table3=[]
 for seed in [21701,21702]:
  for scene in sorted({v['scene'] for v in scene_rows}):
   def get(m,mode):return next(v for v in scene_rows if v['model']==m and v['decision']==mode and v['scene']==scene)
   a=get(f'L8_{seed}','calibrated');raw=get(f'L8_{seed}','raw');src=get('Source','calibrated');flow=next(v for v in flows if v['model']==f'L8_{seed}' and v['scene']==scene)
   table3.append({'scene':scene,'seed':seed,'cal_L8_minus_raw_L8_mIoU_pp':a['mIoU']-raw['mIoU'],'cal_L8_minus_cal_Source_mIoU_pp':a['mIoU']-src['mIoU'],'Cloud_to_Surface_net_delta':flow['Cloud_to_Surface_net_delta'],'Surface_to_Cloud_net_delta':flow['Surface_to_Cloud_net_delta'],'Shadow_recall_delta_pp':a['Shadow_recall']-raw['Shadow_recall'] if a['Shadow_recall'] is not None else None,'Cloud_reference':a['Cloud_pixels'],'Surface_reference':a['Surface_pixels'],'Shadow_reference':a['Shadow_pixels']})
 for n,v in [('TABLE1',table1),('TABLE2',table2),('TABLE3',table3),('VAL_PER_SCENE',scene_rows),('RANKING_INVARIANTS',rank),('SIX_WAY_NET_ERRORS',flows),('DIRECT_PIXEL_TRANSITIONS',transitions)]:save(n+'.json',v);csvout(n+'.csv',v)
 save('VALIDATION_INPUTS_USED.json',{'epoch_completed':time.time(),'used_score_and_label_caches':used,'historical_SP_reference_confusions_SHA':oldentry['SHA'],'validation_forward_calls':0})
 save('CPU_ANALYSIS_COMPLETE.json',{'status':'passed','CPU_seconds':time.monotonic()-START,'CPU_peak_bytes':cpu_peak(),'zero_bias_all30_predictions_and_confusions_exact230':True,'AUC_AP_all30_scenes_plus3_pools_exact_invariant':True,'new_model_calls':0,'new_bias_fits_on_validation':0,'new_significance_intervals':0})
 print(json.dumps({'table1':table1,'table2':table2,'table3_focus':[v for v in table3 if v['scene'] in ['LC80380362013144LGN00','LC81490372013362LGN00','LC80500162014247LGN00']]},ensure_ascii=False))
if __name__=='__main__':main()
