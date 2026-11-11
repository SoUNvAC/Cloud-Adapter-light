"""One L8 root, 18 scene weights; independent replay, then seal with reused SP bias."""
from phase235_common import *
import sys,resource,traceback,math
import numpy as np
import phase231_bias as op
START=time.monotonic()
def formula_check():
 op.handcheck();p=np.array([[.2,.3,.5],[.7,.2,.1],[.4,.5,.1]],np.float64);y=np.array([0,1,2]);s=np.log(p);d=s[:,1]-np.logaddexp(s[:,0],s[:,2]);target=(y==1).astype(float)
 def ce(b):
  q=s.copy();q[:,1]+=b;return float(np.mean(np.logaddexp.reduce(q,axis=1)-q[np.arange(3),y]))
 for b in [-5.,0.,1.,5.]:assert abs((ce(b+1e-5)-ce(b-1e-5))/2e-5-float(np.mean(op.sigmoid(d+b)-target)))<1e-9
 # Two unequal-sized chunks inside a scene must be pixel-weighted, not chunk-weighted.
 a=np.array([0.]);b=np.array([1.,1.,1.]);assert float(np.concatenate([a,b]).mean())==.75 and (a.mean()+b.mean())/2==.5
 return {'CE_vs_BCE_derivative':True,'boundary_and_root_operator':True,'unequal_chunk_pixels_within_scene':True}
def main():
 assert (R/'GPU_extract_EXIT_CODE').read_text().strip()=='0' and not (R/'BIAS_FROZEN.json').exists();check_bundle();checks=formula_check();receipts=js(R/'CACHE_RECEIPTS.json');binding=js(R/'TRAIN_NATIVE_BINDING.json');assert len(receipts)==len(binding)==1560;cacheallowed={str((R/v[k]).resolve()) for v in receipts for k in ['d3_path','label_path']}
 def audit(e,a):
  if e=='open' and a and isinstance(a[0],(str,bytes,os.PathLike)):
   p=Path(os.fsdecode(a[0]))
   if p.suffix.lower() in ['.npy','.npz','.png','.tif','.jpg'] and str(p.resolve()) not in cacheallowed:raise RuntimeError('235 fitting forbids nontraining cache '+str(p))
 sys.addaudithook(audit);scenes=sorted({v['scene'] for v in receipts});assert len(scenes)==18;chunks=[];supports=[]
 for scene in scenes:
  ds=[];ys=[];cc=np.zeros(3,np.int64);fill=0;nempty=0;rows=[v for v in receipts if v['scene']==scene]
  for v in rows:
   dp=R/v['d3_path'];yp=R/v['label_path'];assert sha(dp)==v['d3_SHA'] and sha(yp)==v['label_SHA'];d=np.load(dp,allow_pickle=False);y=np.load(yp,allow_pickle=False);assert d.dtype==np.float64 and y.dtype==np.uint8 and d.ndim==y.ndim==1 and d.shape==y.shape and len(d)==v['valid_pixels'] and np.isfinite(d).all() and np.isin(y,[0,1,2]).all();cc+=v['support'];fill+=v['Fill_pixels'];nempty+=not bool(len(d))
   if len(d):ds.append(d);ys.append((y==1).astype(np.float64))
  assert ds,scene+' has no valid pixels';chunks.append((np.concatenate(ds),np.concatenate(ys)));supports.append({'scene':scene,'inputs':len(rows),'empty_inputs_preserved':nempty,'support':cc.tolist(),'valid_pixels':int(cc.sum()),'Fill_pixels':fill});del ds,ys
  assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<16*1024**3
 save('TRAIN_SCENE_SUPPORT.json',supports);progress={'gradient_calls':0};base_gradient=op.gradient
 def tracked(chunks,b):
  assert time.monotonic()-START<14000 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<16*1024**3;v=base_gradient(chunks,b);progress['gradient_calls']+=1;save('FIT_STATUS.json',{'stage':'primary_root','PID':os.getpid(),'elapsed_CPU_stage_seconds':time.monotonic()-START,'gradient_calls':progress['gradient_calls'],'current_b':b,'g':v});return v
 op.gradient=tracked;fit=op.solve(chunks);assert fit['Lb']<=fit['L0']+1e-12;save('PRIMARY_FIT_COMPLETE.json',{'fit':fit,'CPU_seconds':time.monotonic()-START,'loss_absolute_definition':'BCE Cloud vs nonCloud, not full3classCE'})
 def g(b):
  means=[]
  for d,y in chunks:
   x=d+b;e=np.exp(-np.abs(x));p=np.where(x>=0,1/(1+e),e/(1+e));means.append((float(p.sum(dtype=np.float64))-float(y.sum(dtype=np.float64)))/len(y))
  return math.fsum(means)/18
 def loss(b):return math.fsum(float(np.mean(np.maximum(d+b,0)-(d+b)*y+np.log1p(np.exp(-np.abs(d+b))),dtype=np.float64)) for d,y in chunks)/18
 lo,hi=-5.,5.;gl,gh=g(lo),g(hi)
 if gl>=0:b=lo;boundary=True
 elif gh<=0:b=hi;boundary=True
 else:
  boundary=False
  for i in range(60):
   m=(lo+hi)/2;gm=g(m)
   if gm>0:hi=m
   else:lo=m
   save('FIT_STATUS.json',{'stage':'independent_root_replay','PID':os.getpid(),'elapsed_CPU_stage_seconds':time.monotonic()-START,'iteration':i+1});assert time.monotonic()-START<14000
  b=(lo+hi)/2
 assert abs(b-fit['b'])<1e-10 and boundary==fit['boundary'] and abs(loss(0)-fit['L0'])<1e-11 and abs(loss(b)-fit['Lb'])<1e-11 and abs(g(b)-fit['g_b'])<1e-11
 independent={'status':'passed','stored_b':fit['b'],'independent_b':b,'absolute_difference':abs(b-fit['b']),'independent_BCE0':loss(0),'independent_BCEb':loss(b),'g_b':g(b),'scene_equal_not_chunk_equal':True,'scene_count':18,'formula_checks':checks,'validation_test_arrays_opened':False};save('INDEPENDENT_BIAS_ACCEPTANCE.json',independent)
 old=js(R/'SPARCS_SOURCE_BIAS.json');assert old['bias_receipt']['b']==1.5086972783056742 and old['reuse_only'];fit['nonCloud_negative']=fit.pop('Surface_negative');fit['scene_support']=supports
 frozen={'epoch':time.time(),'biases':{'L8':{'b':fit['b'],'fit':fit,'new_fit':True,'loss_absolute_definition':'BCE not three-class CE'},'SPARCS':{'b':old['bias_receipt']['b'],'new_fit':False,'source_file_SHA':old['source_file_sha256'],'source_receipt_SHA':sha(R/'SPARCS_SOURCE_BIAS.json'),'original_fit_support':old['bias_receipt']['support_by_class'],'original_fit_scenes':60}},'cache_receipts_SHA':sha(R/'CACHE_RECEIPTS.json'),'train_manifest_SHA':sha(R/'L8_TRAIN_INPUTS.json'),'native_binding_SHA':sha(R/'TRAIN_NATIVE_BINDING.json'),'code_SHA':{p.name:sha(p) for p in (R/'code').glob('*.py')},'new_scalar_fits':1,'new_network_updates':0,'validation_test_before_bias_seal':False};save('BIAS_FROZEN.json',frozen);save('BIAS_SEAL.json',{'epoch':time.time(),'SHA':sha(R/'BIAS_FROZEN.json')});save('FIT_COMPLETE.json',{'status':'passed','CPU_seconds':time.monotonic()-START,'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'temporary_cache_bytes':bytes_cache(),'no_SP_training_pixels':True});(R/'FIT_EXIT_CODE').write_text('0\n')
if __name__=='__main__':
 try:main()
 except BaseException:(R/'FIT_FAILURE.txt').write_text(traceback.format_exc());(R/'FIT_EXIT_CODE').write_text('1\n');raise
