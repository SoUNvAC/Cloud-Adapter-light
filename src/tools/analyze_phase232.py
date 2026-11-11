"""Frozen Raw/B2/B3 evaluation on original230 validation caches only."""
from pathlib import Path
import csv,hashlib,json,time
import numpy as np
from phase231_metrics import metrics,summary
from phase230_metrics import primary,independent
from analyze_phase230 import cpu_peak
REPO=Path(__file__).resolve().parents[2];R=REPO/'outputs/phase232/phase232_20261027';D=R/'delivery';OLD=REPO/'outputs/phase230/phase230_20261026';B2=REPO/'outputs/phase231/phase231_20261026';START=time.monotonic()
KEYS=['mIoU','Cloud_recall','Surface_to_Cloud_FPR','Shadow_recall','nonShadow_to_Shadow_FPR']
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(R/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def output(n,rows):
 save(n+'.json',rows);keys=list(dict.fromkeys(k for r in rows for k in r))
 with (R/(n+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def delta(a,b):return {k:a[k]-b[k] if a[k] is not None and b[k] is not None else None for k in KEYS}
def main():
 assert not (R/'VALIDATION_ACCESS_STARTED.json').exists();frozen=js(D/'BIAS_FROZEN.json');assert sha(D/'BIAS_FROZEN.json')==js(D/'BIAS_SEAL.json')['BIAS_FROZEN_SHA'];assert js(D/'INDEPENDENT_BIAS_ACCEPTANCE.json')['status']=='passed';assert frozen['validation_arrays_opened_before_seal'] is False
 lock=js(R/'LOCK.json')
 for n,h in lock['identities'].items():assert sha(REPO/n)==h,n
 save('VALIDATION_ACCESS_STARTED.json',{'epoch':time.time(),'sealed_bias_SHA':sha(D/'BIAS_FROZEN.json'),'all3_sealed_before_validation':True,'independent_numeric_solver_passed':True})
 oldfit={v['model']:v for v in js(B2/'delivery/BIAS_FROZEN.json')['biases']};newfit={v['model']:v for v in frozen['biases']};history={(v['model'],v['scene'],v['decision']):v for v in js(B2/'VAL_PER_SCENE.json')};receipts=js(OLD/'delivery/SCORE_RECEIPTS.json');per=[];tab2=[];trans=[];flows=[];ranking=[];used=[];tab3=[];perdelta=[]
 tab1=[{'model':m,'support_by_class':v['support_by_class'],'b2':oldfit[m]['b'],'b3':v['b'],'L3_zero':v['L3_zero'],'L3_b':v['L3_b'],'K_mean':v['K_mean'],'g_b':v['g_b'],'boundary':v['boundary'],'scenes':v['scenes']} for m,v in newfit.items()]
 for model in lock['models']:
  biases={'Raw':0.,'B2':oldfit[model]['b'],'B3':newfit[model]['b']};pool=[]
  for rec in [v for v in receipts if v['model']==model]:
   assert time.monotonic()-START<1800 and cpu_peak()<8*1024**3
   pp=OLD/'delivery'/rec['score_path'];yp=OLD/'delivery'/rec['label_path'];assert sha(pp)==rec['score_SHA'] and sha(yp)==rec['label_SHA'];p=np.load(pp);y=np.load(yp);valid=y!=255;pred={};cm={};used.append({'model':model,'scene':rec['scene'],'score_SHA':rec['score_SHA'],'label_SHA':rec['label_SHA']})
   for mode,b in biases.items():
    q=p.astype(np.float64);q[1]*=np.exp(np.float64(b));pred[mode]=q.argmax(0).astype(np.uint8);cm[mode]=np.bincount(3*y[valid].astype(np.int64)+pred[mode][valid],minlength=9).reshape(3,3)
    if mode!='B3':assert cm[mode].tolist()==history[model,rec['scene'],'raw' if mode=='Raw' else 'calibrated']['confusion']
    per.append({'model':model,'scene':rec['scene'],'decision':mode,'b':b,'prediction_SHA':hashlib.sha256(pred[mode].tobytes()).hexdigest(),**metrics(cm[mode])})
   for before,after in [('Raw','B2'),('Raw','B3'),('B2','B3')]:
    J=np.bincount(9*y[valid].astype(np.int64)+3*pred[before][valid].astype(np.int64)+pred[after][valid],minlength=27).reshape(3,3,3);assert np.array_equal(J.sum(2),cm[before]) and np.array_equal(J.sum(1),cm[after]);trans.append({'model':model,'scene':rec['scene'],'comparison':before+'->'+after,'J_reference_before_after':J.tolist(),'before_correct_Shadow_lost_to_Cloud':int(J[2,2,1]),'before_wrong_Shadow_to_Cloud_restored':int(J[2,1,2])})
    dif=cm[after]-cm[before];row={'model':model,'scene':rec['scene'],'comparison':after+'-'+before}
    for i,a in enumerate(['Surface','Cloud','Shadow']):
     for j,b in enumerate(['Surface','Cloud','Shadow']):
      if i!=j:row[a+'->'+b]=int(dif[i,j])
    flows.append(row)
   for mode in ['B2','Raw']:
    direct=[v for v in trans if v['model']==model and v['scene']==rec['scene']];rb2=next(v for v in direct if v['comparison']=='Raw->B2');rb3=next(v for v in direct if v['comparison']=='Raw->B3')
    perdelta.append({'model':model,'scene':rec['scene'],'comparison':'B3-'+mode,**delta(metrics(cm['B3']),metrics(cm[mode])),'raw_correct_Shadow_to_Cloud_reduction_B3_vs_B2':rb2['before_correct_Shadow_lost_to_Cloud']-rb3['before_correct_Shadow_lost_to_Cloud']})
   mask=(y==0)|(y==1);yy=(y[mask]==1).astype(np.uint8);d=(np.log(np.maximum(p[1],1e-12))-np.log(np.maximum(p[0],1e-12)))[mask].astype(np.float64);pool.append((yy,d));base=primary(yy,d);ia,ip=independent(yy,d);check={'AUC':ia,'AP':ip}
   for k in ['AUC','AP']:assert (base[k] is None and check[k] is None) or abs(base[k]-check[k])<1e-12
   vals={mode:primary(yy,d+b) for mode,b in biases.items()};assert all(v['AUC']==base['AUC'] and v['AP']==base['AP'] for v in vals.values());ranking.append({'model':model,'scene':rec['scene'],'values':vals,'exact_invariant':True})
  yy=np.concatenate([x[0] for x in pool]);dd=np.concatenate([x[1] for x in pool]);a=primary(yy,dd);vals={mode:primary(yy,dd+b) for mode,b in biases.items()};assert all(v['AUC']==a['AUC'] and v['AP']==a['AP'] for v in vals.values());ranking.append({'model':model,'scene':'POOLED','values':vals,'exact_invariant':True})
  for mode,b in biases.items():tab2.append({'model':model,'decision':mode,'b':b,**summary([v for v in per if v['model']==model and v['decision']==mode]),'pooled_AUC_AP':a})
  before=next(v for v in tab2 if v['model']==model and v['decision']=='B2');after=next(v for v in tab2 if v['model']==model and v['decision']=='B3');reduction=sum(v['raw_correct_Shadow_to_Cloud_reduction_B3_vs_B2'] for v in perdelta if v['model']==model and v['comparison']=='B3-B2')
  for agg in ['pooled','scene_mean']:tab3.append({'model':model,'aggregation':agg,'comparison':'B3-B2',**delta(after[agg],before[agg]),'raw_correct_Shadow_to_Cloud_reduction':reduction})
 for agg in ['pooled','scene_mean']:
  rows=[v for v in tab3 if v['model'].startswith('L8_') and v['aggregation']==agg];tab3.append({'model':'L8_two_seed_metric_difference_mean','aggregation':agg,'comparison':'B3-B2',**{k:float(np.mean([v[k] for v in rows])) if all(v[k] is not None for v in rows) else None for k in KEYS},'raw_correct_Shadow_to_Cloud_reduction':float(np.mean([v['raw_correct_Shadow_to_Cloud_reduction'] for v in rows]))})
 for n,rows in [('TABLE1',tab1),('TABLE2',tab2),('TABLE3',tab3),('VAL_PER_SCENE',per),('PER_SCENE_DIFFERENCES',perdelta),('DIRECT_PIXEL_TRANSITIONS',trans),('SIX_WAY_NET_ERRORS',flows),('RANKING_INVARIANTS',ranking)]:output(n,rows)
 output('FOCUS_SCENES',[v for v in perdelta if v['scene'] in ['LC81500412014244LGN00','LC81490372013362LGN00','LC80380362013144LGN00']]);save('VALIDATION_INPUTS_USED.json',used);save('CPU_ANALYSIS_COMPLETE.json',{'status':'passed','CPU_seconds':time.monotonic()-START,'CPU_peak_bytes':cpu_peak(),'Raw_B2_exact231':True,'validation_calls':0,'new_bootstrap':0})
 print(json.dumps({'TABLE1':tab1,'TABLE3':tab3}))
if __name__=='__main__':main()
