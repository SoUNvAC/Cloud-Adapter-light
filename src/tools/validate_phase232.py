"""Independent first-tie predictions, Boolean confusion and transition replay."""
from analyze_phase232 import R,D,OLD,B2,js,sha,save,KEYS,START
from pathlib import Path
import hashlib,time
import numpy as np
from analyze_phase230 import cpu_peak
def independent_metrics(c):
 c=np.array(c,dtype=np.int64);iou=[]
 for k in range(3):
  den=sum(int(x) for x in c[k])+sum(int(x) for x in c[:,k])-int(c[k,k]);iou.append(100*int(c[k,k])/den if den else None)
 ratio=lambda n,d:100*n/d if d else None
 return {'mIoU':sum(x for x in iou if x is not None)/sum(x is not None for x in iou),'Cloud_recall':ratio(int(c[1,1]),int(c[1].sum())),'Surface_to_Cloud_FPR':ratio(int(c[0,1]),int(c[0].sum())),'Shadow_recall':ratio(int(c[2,2]),int(c[2].sum())),'nonShadow_to_Shadow_FPR':ratio(int(c[0,2]+c[1,2]),int(c[0].sum()+c[1].sum()))}
def equal(a,b):assert (a is None and b is None) or (a is not None and b is not None and abs(a-b)<1e-10),(a,b)
def main():
 assert js(R/'CPU_ANALYSIS_COMPLETE.json')['status']=='passed';per=js(R/'VAL_PER_SCENE.json');trans=js(R/'DIRECT_PIXEL_TRANSITIONS.json');new={x['model']:x['b'] for x in js(D/'BIAS_FROZEN.json')['biases']};old={x['model']:x['b'] for x in js(B2/'delivery/BIAS_FROZEN.json')['biases']};receipts=js(OLD/'delivery/SCORE_RECEIPTS.json');checked=0
 for rec in [v for v in receipts if v['model'] in new]:
  m=rec['model'];p=np.load(OLD/'delivery'/rec['score_path']);y=np.load(OLD/'delivery'/rec['label_path']);valid=y!=255;pred={}
  for mode,b in [('Raw',0.),('B2',old[m]),('B3',new[m])]:
   a=p[0].astype(np.float64);c=p[1].astype(np.float64)*np.exp(b);s=p[2].astype(np.float64);z=np.zeros(y.shape,dtype=np.uint8);z[(c>a)&(c>=s)]=1;z[(s>a)&(s>c)]=2;pred[mode]=z
   cm=[[int(np.count_nonzero(valid&(y==i)&(z==j))) for j in range(3)] for i in range(3)];stored=next(v for v in per if v['model']==m and v['scene']==rec['scene'] and v['decision']==mode);assert cm==stored['confusion'] and hashlib.sha256(z.tobytes()).hexdigest()==stored['prediction_SHA']
   for k,v in independent_metrics(cm).items():equal(v,stored[k])
   checked+=1
  for before,after in [('Raw','B2'),('Raw','B3'),('B2','B3')]:
   J=[[[int(np.count_nonzero(valid&(y==i)&(pred[before]==j)&(pred[after]==k))) for k in range(3)] for j in range(3)] for i in range(3)];stored=next(v for v in trans if v['model']==m and v['scene']==rec['scene'] and v['comparison']==before+'->'+after);assert J==stored['J_reference_before_after']
 for row in js(R/'TABLE2.json'):
  rows=[v for v in per if v['model']==row['model'] and v['decision']==row['decision']];pooled=independent_metrics(np.sum([v['confusion'] for v in rows],axis=0))
  for k in KEYS:
   equal(pooled[k],row['pooled'][k]);vals=[v[k] for v in rows if v[k] is not None];equal(sum(vals)/len(vals) if vals else None,row['scene_mean'][k]);assert len(vals)==row['defined_scenes'][k]
 for row in js(R/'PER_SCENE_DIFFERENCES.json'):
  m=row['model'];scene=row['scene'];mode=row['comparison'].split('-')[1];get=lambda md:next(v for v in per if v['model']==m and v['scene']==scene and v['decision']==md)
  for k in KEYS:
   a,b=get('B3')[k],get(mode)[k];equal(a-b if a is not None and b is not None else None,row[k])
 assert checked==90 and cpu_peak()<8*1024**3
 save('INDEPENDENT_ACCEPTANCE.json',{'status':'passed','scene_decisions':90,'direct_transition_tables':90,'nine_summary_rows':True,'all_scene_differences':True,'CPU_seconds':time.monotonic()-START,'CPU_peak_bytes':cpu_peak(),'argmax':'Boolean strict comparisons preserving first-index ties','validation_calls':0})
 print('independent acceptance passed')
if __name__=='__main__':main()
