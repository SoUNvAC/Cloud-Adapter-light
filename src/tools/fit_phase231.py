"""Seal all three train-only bias estimates before any validation cache reads."""
from pathlib import Path
import hashlib,json,os,resource,sys,time,traceback
import numpy as np
from phase231_bias import solve,handcheck
R=Path('/home/scv/shared/phase231_20261026');START=time.monotonic()
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):
 p=R/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False));t.replace(p)
def main():
 assert (R/'GPU_EXIT_CODE').read_text().strip()=='0' and not (R/'BIAS_FROZEN.json').exists();save('FIT_STARTED.json',{'epoch':time.time(),'PID':os.getpid(),'validation_cache_accessed':False});handcheck()
 allowed={str(p.resolve()) for folder in ['d_cache','labels'] for p in (R/folder).glob('*.npy')};denied=[]
 def audit(e,a):
  if e=='open' and a and isinstance(a[0],(str,bytes,os.PathLike)):
   p=Path(os.fsdecode(a[0]))
   if p.suffix.lower() in ['.npy','.npz','.png','.tif','.jpg'] and str(p.resolve()) not in allowed:denied.append(str(p));raise RuntimeError('Bias fit forbids validation or non-calibration arrays')
 sys.addaudithook(audit);receipts=js(R/'CALIBRATION_RECEIPTS.json');results=[]
 for model in js(R/'LOCK.json')['models_to_calibrate']:
  chunks=[];support=[]
  for v in [x for x in receipts if x['model']==model]:
   dp=R/v['d_path'];yp=R/v['label_path'];assert sha(dp)==v['d_SHA'] and sha(yp)==v['label_SHA'];d=np.load(dp,allow_pickle=False);y=np.load(yp,allow_pickle=False);mask=(y==0)|(y==1);n=int(mask.sum());support.append({'scene':v['scene'],'Cloud_positive':int((y==1).sum()),'Surface_negative':int((y==0).sum()),'included':n>0})
   if n:chunks.append((d[mask].astype(np.float64),(y[mask]==1).astype(np.float64)))
  assert len(support)==60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<8*1024**3
  fit=solve(chunks);assert fit['Lb']<=fit['L0']+1e-12;results.append({'model':model,**fit,'scene_support':support});save('FIT_PARTIAL.json',results);del chunks
  assert time.monotonic()-START<3600
 assert len(results)==3 and not denied
 frozen={'biases':results,'epoch_sealed':time.time(),'input_file_SHA':sha(R/'CALIBRATION_ACTUAL_INPUTS.json'),'cache_receipts_SHA':sha(R/'CALIBRATION_RECEIPTS.json'),'fit_code_SHA':sha(Path(__file__)),'operator_code_SHA':sha(Path(__file__).with_name('phase231_bias.py')),'validation_arrays_opened_before_seal':False,'forbidden_accesses':denied,'equal_scene_weight':True,'bounds':[-5,5],'bisection_iterations':60}
 save('BIAS_FROZEN.json',frozen);save('BIAS_SEAL.json',{'BIAS_FROZEN_SHA':sha(R/'BIAS_FROZEN.json'),'epoch':time.time(),'all3_biases_sealed':True});save('FIT_COMPLETE.json',{'status':'complete','CPU_seconds':time.monotonic()-START,'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'scalars_fitted':3,'network_updates':0,'validation_cache_reads':0});(R/'FIT_EXIT_CODE').write_text('0\n')
if __name__=='__main__':
 try:main()
 except BaseException:(R/'FIT_FAILURE.txt').write_text(traceback.format_exc());(R/'FIT_EXIT_CODE').write_text('1\n');raise
