"""Independent CPU checks of every main-task metric, support and confusion."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from phase230_metrics import independent,handchecks
from analyze_phase230 import cpu_peak
R=Path(__file__).resolve().parents[2]/'outputs/phase230/phase230_20261026';D=R/'delivery';START=time.monotonic()
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def eq(a,b,tol=1e-12):assert (a is None and b is None) or (a is not None and b is not None and abs(a-b)<tol),(a,b)
def qs(x):
 if len(x)==0:return None
 x=np.sort(x);out=[]
 for q in [.1,.5,.9]:
  z=(len(x)-1)*q;i=int(z);j=min(i+1,len(x)-1);out.append(float(x[i])+(float(x[j])-float(x[i]))*(z-i))
 return out
def main():
 checks=[];confchecks=0;rows=js(R/'MAIN_PER_SCENE.json');pool=js(R/'POOLED_TASK_METRICS.json');receipts=js(D/'SCORE_RECEIPTS.json');lock=js(R/'LOCK.json')
 def check(y,s,row):
  P=int(np.count_nonzero(y));N=len(y)-P;assert (P,N)==(row['positive'],row['negative']);a,ap=independent(y,s);eq(a,row['AUC']);eq(ap,row['AP'])
  eq(float(np.count_nonzero(s[y==1]>0)/P) if P else None,row['binary_cloud_recall']);eq(float(np.count_nonzero(s[y==0]>0)/N) if N else None,row['binary_surface_FPR'])
  assert row['d_zero_ties']==int(np.count_nonzero(s==0));assert row['d_zero_positive']==int(np.count_nonzero(s[y==1]==0));assert row['d_zero_negative']==int(np.count_nonzero(s[y==0]==0))
  for cls,key in [(1,'cloud_d_P10_median_P90'),(0,'surface_d_P10_median_P90')]:
   v=qs(s[y==cls]);expected=row[key]
   if v is None:assert expected is None
   else:
    for x,z in zip(v,expected):eq(x,z,1e-10)
 for model in lock['models']:
  assert time.monotonic()-START<3600 and cpu_peak()<8*1024**3
  pieces=[]
  for rec in [v for v in receipts if v['model']==model]:
   p=np.load(D/rec['score_path'],allow_pickle=False);ref=np.load(D/rec['label_path'],allow_pickle=False);valid=ref!=255;assert np.isin(ref,[0,1,2,255]).all();assert np.isfinite(p).all() and (p>=0).all();assert np.max(np.abs(p.sum(0)[valid]-1))<1e-6
   pred=np.argmax(p,axis=0).astype(np.uint8);assert hashlib.sha256(pred.tobytes()).hexdigest()==rec['prediction_SHA'];c=[[int(np.count_nonzero((ref==i)&(pred==j))) for j in range(3)] for i in range(3)];assert c==rec['confusion'];confchecks+=1
   old=next(v for v in js(R/f'expected/{model}.json')['per_input'] if v['scene']==rec['scene']);assert c==old['confusion'] and rec['prediction_SHA']==old['prediction_SHA']
   mask=(ref==0)|(ref==1);y=(ref[mask]==1).astype(np.uint8);d=(np.log(np.maximum(p[1],np.float32(1e-12)))-np.log(np.maximum(p[0],np.float32(1e-12))))[mask]
   row=next(v for v in rows if v['model']==model and v['scene']==rec['scene']);check(y,d,row);checks.append({'model':model,'scene':rec['scene'],'all_main_metrics_pass':True});pieces.append((y,d))
   assert [int(np.count_nonzero(p[k][valid]<1e-12)) for k in range(3)]==rec['score_log_clamp_valid_channel_counts']
  y=np.concatenate([v[0] for v in pieces]);d=np.concatenate([v[1] for v in pieces]);pr=next(v for v in pool if v['model']==model and v['task']=='Cloud_vs_Surface');check(y,d,pr);checks.append({'model':model,'scope':'pooled','all_main_metrics_pass':True})
 summary=js(R/'TABLE1.json')
 for model in lock['models']:
  ss=[v for v in rows if v['model']==model];row=next(v for v in summary if v['model']==model)
  for src,dst in [('AUC','scene_mean_AUC'),('AP','scene_mean_AP'),('binary_cloud_recall','scene_mean_binary_cloud_recall'),('binary_surface_FPR','scene_mean_binary_surface_FPR')]:
   v=[x[src] for x in ss if x[src] is not None];eq(sum(v)/len(v) if v else None,row[dst])
  assert row['both_classes_scenes']==sum(v['positive']>0 and v['negative']>0 for v in ss)
 assert len(checks)==55 and confchecks==50
 assert cpu_peak()<8*1024**3
 result={'status':'passed','all55_main_scene_and_pooled_metrics':checks,'argmax_and_3x3_confusions_independently_recomputed':confchecks,'quantiles':'independent sorted-order linear interpolation','binary_decision':'strict d>0;ties Surface','AUC':'ascending Mann-Whitney half ties','AP':'suffix precision per exact tie block, noninterpolated','hand_arrays':handchecks(),'CPU_seconds':time.monotonic()-START,'CPU_peak_bytes':cpu_peak(),'no_new_model_calls':True,'no_bootstrap_or_new_thresholds':True}
 (R/'INDEPENDENT_ACCEPTANCE.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='all55_main_scene_and_pooled_metrics'}))
if __name__=='__main__':main()
