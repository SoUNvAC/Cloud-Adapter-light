"""Three-class CE via its exact Cloud-vs-nonCloud BCE statistic and K."""
from pathlib import Path
import hashlib,json,math,os,time,traceback,sys
if os.name != "nt":import resource
import numpy as np
from phase231_bias import solve,handcheck,sigmoid
R=Path('/home/scv/shared/phase232_20261027');START=time.monotonic()
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(R/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def formula_check():
 p=np.array([[.1,.7,.2],[0.,1.,0.],[.6,.1,.3],[.3,.2,.5]],dtype=np.float64);y=np.array([0,1,2,1]);s=np.log(np.maximum(p,1e-12));d=s[:,1]-np.logaddexp(s[:,0],s[:,2]);binary=(y==1)
 def ce(b):
  q=s.copy();q[:,1]+=b;return float(np.mean(np.logaddexp.reduce(q,axis=1)-q[np.arange(4),y]))
 def bce(b):return float(np.mean(np.logaddexp(0,d+b)-binary*(d+b)))
 K=ce(0)-bce(0)
 for b in [-5.,-.4,0.,2.,5.]:
  assert abs(ce(b)-(K+bce(b)))<1e-12
  g=float(np.mean(sigmoid(d+b)-binary));assert abs((ce(b+1e-5)-ce(b-1e-5))/2e-5-g)<1e-9
 return {'direct_CE_BCE_K_identity':True,'finite_difference_gradient':True,'BCE_boundary_handchecks':handcheck()}
def main():
 assert (R/'GPU_EXIT_CODE').read_text().strip()=='0' and not (R/'BIAS_FROZEN.json').exists();checks=formula_check();receipts=js(R/'CALIBRATION_RECEIPTS.json');out=[];independent=[]
 allowed={str((R/v['d_path']).resolve()) for v in receipts}|{str(Path(v['label_path']).resolve()) for v in receipts}
 def audit(e,a):
  if e=='open' and a and isinstance(a[0],(str,bytes,os.PathLike)):
   p=Path(os.fsdecode(a[0]))
   if p.suffix.lower() in ['.npy','.npz','.png','.jpg','.tif'] and str(p.resolve()) not in allowed:raise RuntimeError('232 fitting forbids noncalibration arrays')
 sys.addaudithook(audit)
 for model in js(R/'LOCK.json')['models']:
  chunks=[];support=[];ks=[]
  for v in [v for v in receipts if v['model']==model]:
   dp=R/v['d_path'];yp=Path(v['label_path']);assert sha(dp)==v['d_SHA'] and sha(yp)==v['label_SHA'];d=np.load(dp);y=np.load(yp);valid=y!=255;assert d.dtype==np.float64 and np.isfinite(d).all();chunks.append((d[valid],(y[valid]==1).astype(np.float64)));support.append({'scene':v['scene'],'support':v['support'],'ignore':v['ignore']});ks.append(v['K'])
  assert len(chunks)==60;fit=solve(chunks);K=float(np.mean(ks));fit['BCE0']=fit.pop('L0');fit['BCEb']=fit.pop('Lb');fit['L3_zero']=K+fit['BCE0'];fit['L3_b']=K+fit['BCEb'];fit['K_mean']=K;fit['nonCloud_negative']=fit.pop('Surface_negative');fit['support_by_class']=np.sum([v['support'] for v in support],axis=0).tolist();out.append({'model':model,**fit,'scene_support':support})
  # Independent aggregation, sigmoid and softplus. Same prescribed root, not another candidate.
  def g(b):
   means=[]
   for d,y in chunks:
    x=d+b;e=np.exp(-np.abs(x));q=np.where(x>=0,1/(1+e),e/(1+e));means.append((float(q.sum())-float(y.sum()))/len(y))
   return math.fsum(means)/60
  lo,hi=-5.,5.
  if g(lo)>=0:b=lo
  elif g(hi)<=0:b=hi
  else:
   for i in range(60):
    m=(lo+hi)/2
    if g(m)>0:hi=m
    else:lo=m
   b=(lo+hi)/2
  def ce(b):return K+math.fsum(float(np.mean(np.maximum(d+b,0)-(d+b)*y+np.log1p(np.exp(-np.abs(d+b))))) for d,y in chunks)/60
  assert abs(b-fit['b'])<1e-10 and abs(ce(b)-fit['L3_b'])<1e-11 and abs(g(b)-fit['g_b'])<1e-11
  independent.append({'model':model,'independent_b':b,'abs_difference':abs(b-fit['b']),'independent_L3':ce(b),'independent_g':g(b),'passed':True});del chunks
  assert time.monotonic()-START<3400 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<8*1024**3
 frozen={'biases':out,'epoch_sealed':time.time(),'input_file_SHA':sha(R/'CALIBRATION_ACTUAL_INPUTS.json'),'cache_receipts_SHA':sha(R/'CALIBRATION_RECEIPTS.json'),'code_SHA':{p.name:sha(p) for p in (R/'code').glob('*.py')},'validation_arrays_opened_before_seal':False}
 save('BIAS_FROZEN.json',frozen);save('BIAS_SEAL.json',{'BIAS_FROZEN_SHA':sha(R/'BIAS_FROZEN.json'),'epoch':time.time(),'all3_biases_sealed':True});save('INDEPENDENT_BIAS_ACCEPTANCE.json',{'status':'passed','results':independent,'formula_checks':checks,'validation_arrays_opened':False});save('FIT_COMPLETE.json',{'CPU_seconds':time.monotonic()-START,'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'scalars_fitted':3});(R/'FIT_EXIT_CODE').write_text('0\n')
if __name__=='__main__':
 try:main()
 except BaseException:(R/'FIT_FAILURE.txt').write_text(traceback.format_exc());(R/'FIT_EXIT_CODE').write_text('1\n');raise
