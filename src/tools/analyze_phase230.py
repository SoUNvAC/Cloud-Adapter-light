"""Fixed three-task diagnosis; one image/model cache loaded at a time."""
from pathlib import Path
import csv,hashlib,json,sys,time,ctypes
import numpy as np
from phase230_metrics import primary,independent,handchecks
R=Path(__file__).resolve().parents[2]/'outputs/phase230/phase230_20261026';D=R/'delivery';START=time.monotonic();TASKS=['Cloud_vs_Surface','Cloud_vs_Surface_Shadow','Shadow_vs_Surface']
def cpu_peak():
 if sys.platform=='win32':
  from ctypes import wintypes
  class PMC(ctypes.Structure):
   _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
  x=PMC();x.cb=ctypes.sizeof(x);kernel=ctypes.windll.kernel32;kernel.GetCurrentProcess.restype=wintypes.HANDLE;fn=ctypes.windll.psapi.GetProcessMemoryInfo;fn.argtypes=[wintypes.HANDLE,ctypes.POINTER(PMC),wintypes.DWORD];assert fn(kernel.GetCurrentProcess(),ctypes.byref(x),x.cb);return int(x.PeakWorkingSetSize)
 import resource
 return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(R/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvout(n,rows):
 keys=list(dict.fromkeys(k for x in rows for k in x))
 with (R/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
def mean(xs):
 a=[x for x in xs if x is not None];return float(np.mean(a)) if a else None
def diff(a,b):return a-b if a is not None and b is not None else None
def scorevectors(p,y,task):
 valid=y!=255
 if task=='Cloud_vs_Surface':keep=valid&((y==0)|(y==1));v=np.log(np.maximum(p[1],1e-12))-np.log(np.maximum(p[0],1e-12));positive=1
 elif task=='Cloud_vs_Surface_Shadow':keep=valid;v=p[1];positive=1
 else:keep=valid&((y==0)|(y==2));v=np.log(np.maximum(p[2],1e-12))-np.log(np.maximum(p[0],1e-12));positive=2
 return (y[keep]==positive).astype(np.uint8),v[keep]
def stats(y,s,task):
 v=primary(y,s)
 if task=='Cloud_vs_Surface':
  v.update({'cloud_d_P10_median_P90':np.quantile(s[y==1],[.1,.5,.9]).tolist() if v['positive'] else None,'surface_d_P10_median_P90':np.quantile(s[y==0],[.1,.5,.9]).tolist() if v['negative'] else None,'binary_cloud_recall':float(np.count_nonzero((s>0)&(y==1))/v['positive']) if v['positive'] else None,'binary_surface_FPR':float(np.count_nonzero((s>0)&(y==0))/v['negative']) if v['negative'] else None,'d_zero_ties':int(np.count_nonzero(s==0)),'d_zero_positive':int(np.count_nonzero((s==0)&(y==1))),'d_zero_negative':int(np.count_nonzero((s==0)&(y==0)))})
 return v
def main():
 assert (D/'EXIT_CODE').read_text().strip()=='0';lock=js(R/'LOCK.json');receipts=js(D/'SCORE_RECEIPTS.json');assert len(receipts)==50;tests=handchecks();save('MANUAL_TIES_AP_CHECK.json',tests)
 out=[];pooled=[];summaries=[];validation=[];clamps=[];confusions=[];models=lock['models'];zero_valid_total=0;comparison_checks=0
 for model in models:
  chunks={task:[] for task in TASKS};total=np.zeros((3,3),dtype=np.int64);ref=None
  for rec in [v for v in receipts if v['model']==model]:
   assert time.monotonic()-START<3600 and cpu_peak()<8*1024**3
   sp=D/rec['score_path'];lp=D/rec['label_path'];assert sha(sp)==rec['score_SHA'] and sha(lp)==rec['label_SHA'];p=np.load(sp,allow_pickle=False);y=np.load(lp,allow_pickle=False);assert p.shape==(3,512,512) and p.dtype==np.float32 and y.shape==(512,512) and y.dtype==np.uint8
   assert np.isfinite(p).all() and (p>=0).all();pred=p.argmax(0).astype(np.uint8);valid=y!=255;assert not rec['zero_sum_valid_pixels'];zero_valid_total+=rec['zero_sum_valid_pixels'];assert hashlib.sha256(pred.tobytes()).hexdigest()==rec['prediction_SHA']
   c=np.bincount(3*y[valid].astype(np.int64)+pred[valid],minlength=9).reshape(3,3);assert c.tolist()==rec['confusion'];expected=next(v for v in js(R/f'expected/{model}.json')['per_input'] if v['scene']==rec['scene']);assert c.tolist()==expected['confusion'] and rec['prediction_SHA']==expected['prediction_SHA'];total+=c
   confusions.append({'model':model,'scene':rec['scene'],'confusion':c.tolist(),'valid_pixels':int(c.sum()),'reference_support':c.sum(1).tolist(),'prediction_SHA':rec['prediction_SHA'],'exact_229':True})
   clamps.append({'model':model,'scene':rec['scene'],'valid_pixels':rec['valid_pixels'],'all_pixels':512*512,'zero_sum_all_pixels':rec['zero_sum_all_pixels'],'zero_sum_valid_pixels':rec['zero_sum_valid_pixels'],'zero_sum_all_fraction':rec['zero_sum_all_pixels']/(512*512),'denominator_clamp_all_fraction':rec['denominator_clamp_all_pixels']/(512*512),'log_clamp_valid_channel_counts':rec['score_log_clamp_valid_channel_counts'],'log_clamp_valid_channel_fractions':[n/rec['valid_pixels'] if rec['valid_pixels'] else None for n in rec['score_log_clamp_valid_channel_counts']],'log_clamp_all_channel_fractions':[n/(512*512) for n in rec['score_log_clamp_all_channel_counts']],'normalization_max_abs_error_nonclamped':rec['normalization_max_abs_error_nonclamped']})
   for task in TASKS:
    b,s=scorevectors(p,y,task);v=stats(b,s,task);out.append({'model':model,'scene':rec['scene'],'task':task,**v});chunks[task].append((b,s))
    if task=='Cloud_vs_Surface':
     a,ap=independent(b,s);assert (a is None and v['AUC'] is None) or abs(a-v['AUC'])<1e-12;assert (ap is None and v['AP'] is None) or abs(ap-v['AP'])<1e-12;validation.append({'model':model,'scene':rec['scene'],'AUC_AP_independent_pass':True});comparison_checks+=1
   del p,y,pred
  assert total.tolist()==js(R/f'expected/{model}.json')['confusion']
  tp=np.diag(total);union=total.sum(0)+total.sum(1)-tp;miou=float(100*np.mean(tp[union>0]/union[union>0]))
  for task in TASKS:
   b=np.concatenate([x[0] for x in chunks[task]]);s=np.concatenate([x[1] for x in chunks[task]]);v=stats(b,s,task);pooled.append({'model':model,'task':task,**v});ss=[x for x in out if x['model']==model and x['task']==task]
   if task=='Cloud_vs_Surface':
    a,ap=independent(b,s);assert abs(a-v['AUC'])<1e-12 and abs(ap-v['AP'])<1e-12;comparison_checks+=1
   summaries.append({'model':model,'task':task,'scenes':10,'both_classes_scenes':sum(x['positive']>0 and x['negative']>0 for x in ss),'scene_mean_AUC':mean([x['AUC'] for x in ss]),'scene_mean_AP':mean([x['AP'] for x in ss]),'pooled_AUC':v['AUC'],'pooled_AP':v['AP'],'pooled_positive':v['positive'],'pooled_negative':v['negative'],'scene_mean_binary_cloud_recall':mean([x.get('binary_cloud_recall') for x in ss]),'scene_mean_binary_surface_FPR':mean([x.get('binary_surface_FPR') for x in ss]),'binary_recall_defined_scenes':sum(x['positive']>0 for x in ss) if task=='Cloud_vs_Surface' else None,'binary_FPR_defined_scenes':sum(x['negative']>0 for x in ss) if task=='Cloud_vs_Surface' else None,'pooled_binary_cloud_recall':v.get('binary_cloud_recall'),'pooled_binary_surface_FPR':v.get('binary_surface_FPR'),'three_class_mIoU_percent':miou})
  del chunks,b,s
 table2=[]
 for scene in [x['scene'] for x in js(R/'SPARCS_VAL_INPUTS.json')]:
  def row(model):return next(x for x in out if x['model']==model and x['scene']==scene and x['task']=='Cloud_vs_Surface')
  src=row('Source')
  for seed in [21701,21702]:
   l=row(f'L8_{seed}');sp=row(f'SPARCS_{seed}');assert (l['positive'],l['negative'])==(src['positive'],src['negative'])==(sp['positive'],sp['negative'])
   def med(v,key):return v[key][1] if v[key] is not None else None
   lc=next(v['confusion'] for v in confusions if v['scene']==scene and v['model']==f'L8_{seed}');sc=next(v['confusion'] for v in confusions if v['scene']==scene and v['model']=='Source')
   table2.append({'scene':scene,'seed':seed,'Cloud_positive':l['positive'],'Surface_negative':l['negative'],'L8_minus_Source_AUC_pp':None if l['AUC'] is None else 100*diff(l['AUC'],src['AUC']),'L8_minus_Source_AP_pp':None if l['AP'] is None else 100*diff(l['AP'],src['AP']),'L8_minus_SPARCS_AUC_pp':None if l['AUC'] is None else 100*diff(l['AUC'],sp['AUC']),'L8_minus_SPARCS_AP_pp':None if l['AP'] is None else 100*diff(l['AP'],sp['AP']),'L8_minus_Source_Cloud_d_median':diff(med(l,'cloud_d_P10_median_P90'),med(src,'cloud_d_P10_median_P90')),'L8_minus_Source_Surface_d_median':diff(med(l,'surface_d_P10_median_P90'),med(src,'surface_d_P10_median_P90')),'L8_minus_Source_binary_recall_pp':None if l['binary_cloud_recall'] is None else 100*diff(l['binary_cloud_recall'],src['binary_cloud_recall']),'Cloud_to_Surface_net_error_delta_vs_Source':lc[1][0]-sc[1][0]})
 mainrows=[x for x in out if x['task']=='Cloud_vs_Surface'];aux=[x for x in out if x['task']!='Cloud_vs_Surface'];main_summary=[x for x in summaries if x['task']=='Cloud_vs_Surface'];focus=['LC80380362013144LGN00','LC81490372013362LGN00','LC80500162014247LGN00']
 for n,v in [('TABLE1',main_summary),('TABLE2',table2),('MAIN_PER_SCENE',mainrows),('AUXILIARY_PER_SCENE',aux),('ALL_TASK_SUMMARY',summaries),('POOLED_TASK_METRICS',pooled),('NORMALIZATION_AND_CLAMPS',clamps),('THREE_CLASS_CONFUSIONS',confusions),('FOCUS_THREE_SCENES',[x for x in table2 if x['scene'] in focus])]:save(n+'.json',v);csvout(n+'.csv',v)
 assert comparison_checks==55 and zero_valid_total==0
 assert cpu_peak()<8*1024**3 and time.monotonic()-START<3600
 save('CPU_VALIDATION.json',{'status':'passed','independent_main_AUC_AP_checks':comparison_checks,'all50_argmax_SHA_and_confusions_exact229':True,'manual_ties_AP':tests,'pixels_subsampled':False,'thresholds_fitted':False,'significance_intervals':0,'score_dtype':'float32','d_dtype':'float32 log ratio; natural logarithm','quantile_method':'NumPy linear','CPU_seconds':time.monotonic()-START,'CPU_peak_bytes':cpu_peak()})
 print(json.dumps({'status':'complete','main_summary':main_summary,'focus':[x for x in table2 if x['scene'] in focus]},ensure_ascii=False))
if __name__=='__main__':main()
