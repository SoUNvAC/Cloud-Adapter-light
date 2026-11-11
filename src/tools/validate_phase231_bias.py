"""Train-only independent bias solver verification; no validation cache reads."""
from pathlib import Path
import hashlib,json,math,time
import numpy as np
from analyze_phase230 import cpu_peak
R=Path(__file__).resolve().parents[2]/'outputs/phase231/phase231_20261026';D=R/'delivery';START=time.monotonic()
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 frozen=js(D/'BIAS_FROZEN.json');assert sha(D/'BIAS_FROZEN.json')==js(D/'BIAS_SEAL.json')['BIAS_FROZEN_SHA'];assert frozen['validation_arrays_opened_before_seal'] is False and not frozen['forbidden_accesses'];assert frozen['input_file_SHA']==sha(D/'CALIBRATION_ACTUAL_INPUTS.json') and frozen['cache_receipts_SHA']==sha(D/'CALIBRATION_RECEIPTS.json')
 assert frozen['fit_code_SHA']==sha(D/'code/fit_phase231.py') and frozen['operator_code_SHA']==sha(D/'code/phase231_bias.py')
 actual=js(D/'CALIBRATION_ACTUAL_INPUTS.json');assert [v['scene'] for v in actual]==[v['scene'] for v in js(R/'CALIBRATION_INPUTS.json')];actual={v['scene']:v for v in actual}
 results=[];receipts=js(D/'CALIBRATION_RECEIPTS.json')
 for fit in frozen['biases']:
  assert time.monotonic()-START<3600 and cpu_peak()<8*1024**3;chunks=[];scene_ids=[]
  for v in [x for x in receipts if x['model']==fit['model']]:
   assert v['image_SHA']==actual[v['scene']]['image_sha256'] and v['mask_SHA']==actual[v['scene']]['mask_sha256']
   dp=D/v['d_path'];yp=D/v['label_path'];assert sha(dp)==v['d_SHA'] and sha(yp)==v['label_SHA'];d=np.load(dp,allow_pickle=False);ref=np.load(yp,allow_pickle=False);mask=np.isin(ref,[0,1]);scene_ids.append(v['scene'])
   assert d.dtype==np.float32 and np.isfinite(d).all();assert int((ref==1).sum())==v['Cloud_positive'] and int((ref==0).sum())==v['Surface_negative']
   if mask.any():chunks.append((d[mask].astype(np.float64),(ref[mask]==1).astype(np.float64)))
  assert len(scene_ids)==len(set(scene_ids))==60
  def g(b):
   means=[]
   for d,y in chunks:
    x=d+b;e=np.exp(-np.abs(x));p=np.where(x>=0,1/(1+e),e/(1+e));means.append((float(np.sum(p,dtype=np.float64))-float(np.sum(y,dtype=np.float64)))/len(y))
   return math.fsum(means)/len(means)
  def loss(b):
   return math.fsum(float(np.mean(np.maximum(d+b,0)-(d+b)*y+np.log1p(np.exp(-np.abs(d+b))),dtype=np.float64)) for d,y in chunks)/len(chunks)
  lo,hi=-5.,5.;gl,gh=g(lo),g(hi)
  if gl>=0:b=-5.;boundary=True
  elif gh<=0:b=5.;boundary=True
  else:
   for k in range(60):
    m=(lo+hi)/2
    if g(m)>0:hi=m
    else:lo=m
   b=(lo+hi)/2;boundary=False
  assert abs(b-fit['b'])<1e-10 and boundary==fit['boundary'];assert abs(loss(0)-fit['L0'])<1e-11 and abs(loss(b)-fit['Lb'])<1e-11 and abs(g(b)-fit['g_b'])<1e-11
  assert fit['iterations']==(0 if boundary else 60) and loss(b)<=loss(0)+1e-12
  results.append({'model':fit['model'],'stored_b':fit['b'],'independent_b':b,'absolute_difference':abs(b-fit['b']),'independent_g_b':g(b),'boundary':boundary,'scene_support':len(chunks),'independent_L0':loss(0),'independent_Lb':loss(b),'passed':True});del chunks
 assert len(results)==3
 v={'status':'passed','epoch':time.time(),'sealed_bias_SHA':sha(D/'BIAS_FROZEN.json'),'results':results,'validation_arrays_opened':False,'CPU_seconds':time.monotonic()-START,'CPU_peak_bytes':cpu_peak(),'method':'same fixed60 derivative bisections, independently implemented exp(abs) sigmoid and stable softplus, equal-scene math.fsum aggregation; no alternative fitted candidate'}
 (R/'BIAS_ACCEPTED_LOCAL.json').write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');print(json.dumps(v))
if __name__=='__main__':main()
