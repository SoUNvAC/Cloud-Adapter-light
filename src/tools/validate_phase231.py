"""Independent scalar confusion/decision and ranking verification after bias seal."""
from pathlib import Path
import hashlib,json,math,time
import numpy as np
from phase230_metrics import independent
from analyze_phase230 import cpu_peak
REPO=Path(__file__).resolve().parents[2];R=REPO/'outputs/phase231/phase231_20261026';D=R/'delivery';OLD=REPO/'outputs/phase230/phase230_20261026';START=time.monotonic()
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def eq(a,b):assert (a is None and b is None) or (a is not None and b is not None and abs(a-b)<1e-10),(a,b)
def scalar(c):
 vals=[]
 for i in range(3):
  u=sum(c[i])+sum(c[j][i] for j in range(3))-c[i][i]
  if u:vals.append(100*c[i][i]/u)
 def div(n,d):return 100*n/d if d else None
 return {'mIoU':math.fsum(vals)/len(vals),'Cloud_recall':div(c[1][1],sum(c[1])),'Surface_to_Cloud_FPR':div(c[0][1],sum(c[0])),'Shadow_recall':div(c[2][2],sum(c[2])),'nonShadow_to_Shadow_FPR':div(c[0][2]+c[1][2],sum(c[0])+sum(c[1]))}
def main():
 accept=js(R/'BIAS_SEAL_ACCEPTED_LOCAL.json');assert accept['status']=='passed';assert js(R/'VALIDATION_ACCESS_STARTED.json')['epoch']>accept['epoch'];frozen=js(D/'BIAS_FROZEN.json');rec=js(OLD/'delivery/SCORE_RECEIPTS.json');out=js(R/'VAL_PER_SCENE.json');summaries=js(R/'TABLE2.json');ranks=js(R/'RANKING_INVARIANTS.json');flows=js(R/'SIX_WAY_NET_ERRORS.json');joints=js(R/'DIRECT_PIXEL_TRANSITIONS.json');checks=0;rank_checks=0
 for fit in frozen['biases']:
  model=fit['model'];b=fit['b'];chunks=[]
  for v in [x for x in rec if x['model']==model]:
   assert time.monotonic()-START<3600 and cpu_peak()<8*1024**3
   p=np.load(OLD/'delivery'/v['score_path'],allow_pickle=False);y=np.load(OLD/'delivery'/v['label_path'],allow_pickle=False)
   # Explicit first-index ties, independent of the analyzer argmax implementation.
   p0,p1,p2=[p[i].astype(np.float64) for i in range(3)];p1*=math.exp(b);after=np.where((p0>=p1)&(p0>=p2),0,np.where(p1>=p2,1,2)).astype(np.uint8);before=p.argmax(0).astype(np.uint8)
   cs={}
   for mode,pred in [('raw',before),('calibrated',after)]:
    c=[[int(np.count_nonzero((y==i)&(pred==j))) for j in range(3)] for i in range(3)];cs[mode]=c;r=next(x for x in out if x['model']==model and x['scene']==v['scene'] and x['decision']==mode);assert c==r['confusion'];assert hashlib.sha256(pred.tobytes()).hexdigest()==r['prediction_SHA']
    for k,z in scalar(c).items():eq(z,r[k])
    checks+=1
   flow=next(x for x in flows if x['model']==model and x['scene']==v['scene']);assert [[cs['calibrated'][i][j]-cs['raw'][i][j] for j in range(3)] for i in range(3)]==flow['confusion_net_delta']
   joint=next(x for x in joints if x['model']==model and x['scene']==v['scene']);actual=[[[int(np.count_nonzero((y==i)&(before==j)&(after==k))) for k in range(3)] for j in range(3)] for i in range(3)];assert actual==joint['reference_raw_calibrated_J']
   mask=(y==0)|(y==1);labels=(y[mask]==1).astype(np.uint8);d=(np.log(np.maximum(p[1],1e-12))-np.log(np.maximum(p[0],1e-12)))[mask].astype(np.float64);a,ap=independent(labels,d);aa,aap=independent(labels,d+b);r=next(x for x in ranks if x['model']==model and x['scene']==v['scene']);eq(a,r['AUC_raw']);eq(aa,r['AUC_calibrated']);eq(ap,r['AP_raw']);eq(aap,r['AP_calibrated']);assert (a,ap)==(aa,aap);rank_checks+=1;chunks.append((labels,d))
  yy=np.concatenate([x[0] for x in chunks]);dd=np.concatenate([x[1] for x in chunks]);a,ap=independent(yy,dd);aa,aap=independent(yy,dd+b);assert (a,ap)==(aa,aap);rr=next(x for x in ranks if x['model']==model and x['scene']=='POOLED');eq(a,rr['AUC_raw']);eq(ap,rr['AP_raw']);rank_checks+=1
 for row in summaries:
  if row['decision']=='historical_uncalibrated_reference':
   raw=js(OLD/'THREE_CLASS_CONFUSIONS.json');per=[x for x in raw if x['model']==row['model']]
  else:per=[x for x in out if x['model']==row['model'] and x['decision']==row['decision']]
  assert len(per)==10;c=[[sum(x['confusion'][i][j] for x in per) for j in range(3)] for i in range(3)];assert c==row['pooled']['confusion']
  for k,v in scalar(c).items():eq(v,row['pooled'][k])
  for k in ['mIoU','Cloud_recall','Surface_to_Cloud_FPR','Shadow_recall','nonShadow_to_Shadow_FPR']:
   values=[scalar(x['confusion'])[k] for x in per];defined=[v for v in values if v is not None];eq(math.fsum(defined)/len(defined) if defined else None,row['scene_mean'][k]);assert len(defined)==row['defined_scenes'][k]
 for row in js(R/'TABLE3.json'):
  def get(m,mode):return next(x for x in out if x['model']==m and x['decision']==mode and x['scene']==row['scene'])
  a=get(f"L8_{row['seed']}",'calibrated');raw=get(f"L8_{row['seed']}",'raw');src=get('Source','calibrated');eq(a['mIoU']-raw['mIoU'],row['cal_L8_minus_raw_L8_mIoU_pp']);eq(a['mIoU']-src['mIoU'],row['cal_L8_minus_cal_Source_mIoU_pp']);assert a['confusion'][1][0]-raw['confusion'][1][0]==row['Cloud_to_Surface_net_delta'] and a['confusion'][0][1]-raw['confusion'][0][1]==row['Surface_to_Cloud_net_delta'];eq(a['Shadow_recall']-raw['Shadow_recall'] if a['Shadow_recall'] is not None else None,row['Shadow_recall_delta_pp'])
 assert checks==60 and rank_checks==33
 v={'status':'passed','all60_raw_calibrated_scene_decisions_and_metrics':checks,'all8_summary_rows_scalar_recomputed':True,'main_AUC_AP_independent_invariance_checks':rank_checks,'all30_direct_pixel_J_and6way_net_deltas_pass':True,'all20_TABLE3_rows_pass':True,'bias_before_validation_receipts_pass':True,'CPU_seconds':time.monotonic()-START,'CPU_peak_bytes':cpu_peak(),'new_neural_calls':0,'new_bias_or_threshold_selection':False}
 (R/'INDEPENDENT_ACCEPTANCE.json').write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');print(json.dumps(v))
if __name__=='__main__':main()
