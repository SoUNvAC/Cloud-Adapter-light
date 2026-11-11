"""Source only, separate GPU extraction/evaluation stages with one cumulative ledger."""
from phase235_common import *
import argparse,sys,resource,traceback
def main():
 stage=sys.argv[1];assert stage in ['extract','evaluate'];START=time.monotonic();assert not (R/f'GPU_{stage}_STARTED.json').exists();save(f'GPU_{stage}_STARTED.json',{'PID':os.getpid(),'epoch':time.time()});check_bundle();oldseconds=prior_seconds();records=ledger();ordinal=len(records);assert [v['ordinal'] for v in records]==list(range(1,ordinal+1));counts=collections.Counter();current=None;done=threading.Event();denied=[]
 L=js(R/'LOCK.json');E=js(R/'EVAL_INPUTS.json');train=js(R/'L8_TRAIN_INPUTS.json');valrefs={v['name']:v for v in js(R/'L8_VAL_NATIVE_REFERENCES.json')};evalnative={(v['split'],v['name']):v for v in js(R/'EXPECTED_NATIVE234.json')};plans=js(R/'NATIVE_PLAN.json');source=SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth';assert sha(source)==L['Source_SHA256']
 if stage=='evaluate':
  assert js(R/'INDEPENDENT_BIAS_ACCEPTANCE.json')['status']=='passed';assert sha(R/'BIAS_FROZEN.json')==js(R/'BIAS_SEAL.json')['SHA'];save('EVALUATION_ACCESS_STARTED.json',{'epoch':time.time(),'bias_SHA':sha(R/'BIAS_FROZEN.json'),'all_biases_sealed':True});biases=js(R/'BIAS_FROZEN.json')['biases']
 rows=train if stage=='extract' else [v for pools in E.values() for vs in pools.values() for v in vs];allowed={str(Path(v[k]).resolve()) for v in rows for k in ['image_path','mask_path']};rawplans=plans['train'] if stage=='extract' else plans['test'];rawallowed={str(Path(v['path']).resolve()) for v in rawplans.values()};refallowed=set() if stage=='extract' else {str(Path(v['file']).resolve()) for v in valrefs.values()}
 def audit(e,a):
  if e=='open' and a and isinstance(a[0],(str,bytes,os.PathLike)):
   p=Path(os.fsdecode(a[0]));s=str(p.resolve())
   if p.suffix.lower() in ['.png','.jpg','.tif','.tiff','.jp2','.npz'] and s not in allowed|rawallowed|refallowed:denied.append(s);raise RuntimeError('235 forbidden image '+s)
 sys.addaudithook(audit)
 for v in rows:assert sha(v['image_path'])==v['image_sha256'] and sha(v['mask_path'])==v['mask_sha256'],v['name']
 for g,v in rawplans.items():assert Path(v['path']).is_file(),g+' native missing';assert sha(v['path'])==v['SHA'],g+' native SHA changed'
 os.chdir(SRC);sys.path[:0]=[str(SRC),str(SRC/'tools')];os.environ['XFORMERS_DISABLED']='1';os.environ['OMP_NUM_THREADS']='4'
 import numpy as np,torch
 import rasterio
 from rasterio.windows import Window
 from PIL import Image
 from export_phase12_onnx import build_wrapper
 from validate_phase12_onnxruntime import load_rgb_image
 from evaluate_phase64_source_only_parent import load_parent_target,SPARCS_TO_PARENT
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False;torch.use_deterministic_algorithms(False);log=(R/'CALL_LEDGER.jsonl').open('a',buffering=1)
 def count(kind):
  nonlocal ordinal
  assert ordinal<4800 and oldseconds+time.monotonic()-START<3600;ordinal+=1;counts[kind]+=1;log.write(json.dumps({'ordinal':ordinal,'stage':stage,'kind':kind,'current':current})+'\n')
 def status():save('GPU_STATUS.json',{'stage':stage,'current':current,'PID':os.getpid(),'cumulative_calls':ordinal,'stage_counts':dict(counts),'cumulative_GPU_stage_seconds':oldseconds+time.monotonic()-START})
 def guard():
  while not done.wait(20):
   try:
    assert oldseconds+time.monotonic()-START<3600 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<16*1024**3;assert bytes_cache()<6*1024**3 and bytes_formal()<200*1024**2
    if torch.cuda.is_initialized():assert max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())<20*1024**3
    status()
   except BaseException:(R/f'GPU_{stage}_RESOURCE_FAILURE.txt').write_text(traceback.format_exc());os._exit(71)
 threading.Thread(target=guard,daemon=True).start();depth=[0];probes=[]
 def pre(m,a):
  if depth[0]==0:count('construction_probe');probes.append(type(m).__name__)
  depth[0]+=1
 def post(m,a,o):depth[0]-=1
 h1=torch.nn.modules.module.register_module_forward_pre_hook(pre);h2=torch.nn.modules.module.register_module_forward_hook(post)
 try:wrapper,cfg=build_wrapper(argparse.Namespace(config='configs/protocol/phase64_source_parent_rgb.py',checkpoint=str(source),precision='fp32',active_block_indices=None))
 finally:h1.remove();h2.remove()
 model=wrapper.segmentor.eval()
 for p in model.parameters():p.requires_grad=False
 states=dict(list(model.named_parameters())+list(model.named_buffers()));ck=torch.load(source,map_location='cpu')['state_dict'];assert all(torch.equal(t,states[n].detach().cpu()) for n,t in ck.items());frozen={n:t.detach().cpu().clone() for n,t in states.items()};del ck
 save(f'MODEL_{stage}_ACCEPTED.json',{'Source_SHA':sha(source),'all_states_exact_Source':True,'FP32_eval':True,'versions':{'torch':torch.__version__,'numpy':np.__version__},'construction_probes':probes,'frozen_source_only':True})
 handles={};native_receipts=[];lut=np.zeros(256,np.uint8);lut[64]=1;lut[192]=2;lut[255]=3
 def label(v,domain,split):
  if domain=='SPARCS':
   y=load_parent_target(Path(v['mask_path']),SPARCS_TO_PARENT,(512,512)).astype(np.uint8);return y,y!=255
  if split=='val':
   ref=valrefs[v['name']];assert sha(ref['file'])==ref['sha256']
   with np.load(ref['file'],allow_pickle=False) as z:native=z['native'];valid=z['valid'];truth=z['truth']
   identity=ref['sha256'];rawpath=ref['file']
  else:
   g=v['scene'];entry=rawplans[g]
   if g not in handles:handles[g]=rasterio.open(entry['path'])
   x,y=int(v['x']),int(v['y']);col,row=(y,x) if plans['verified_axes']=='x_row_y_col' else (x,y);native=handles[g].read(1,window=Window(col,row,512,512),boundless=True,fill_value=0);valid=native!=0;identity=entry['SHA'];rawpath=entry['path'];truth=None
  assert native.shape==(512,512) and np.isin(native,[0,64,128,192,255]).all() and valid.dtype==bool and np.array_equal(valid,native!=0);png=np.asarray(Image.open(v['mask_path']));assert np.array_equal(lut[native],png),'Native/PNG mismatch '+v['name'];y=np.zeros(native.shape,np.uint8);y[native==64]=2;y[np.isin(native,[192,255])]=1
  if truth is not None:assert np.array_equal(truth[valid],y[valid])
  rec={'split':split,'scene':v['scene'],'name':v['name'],'raw_file':rawpath,'raw_file_SHA':identity,'native_SHA':hashlib.sha256(native.tobytes()).hexdigest(),'valid_SHA':hashlib.sha256(valid.tobytes()).hexdigest(),'support':[int(((y==k)&valid).sum()) for k in range(3)],'Fill_pixels':int((~valid).sum()),'valid_pixels':int(valid.sum()),'old_PNG_exact':True}
  if split!='train':
   ex=evalnative[split,v['name']];assert rec['native_SHA']==ex['native_array_SHA'] and rec['valid_SHA']==ex['valid_array_SHA'] and rec['raw_file_SHA']==ex['native_source_SHA']
  native_receipts.append(rec);return y,valid
 def forward(v):
  x=torch.from_numpy(load_rgb_image(Path(v['image_path']),512)).cuda();count('train_input' if stage=='extract' else 'evaluation_input')
  with torch.inference_mode():z=wrapper(x).float().clamp_min(0);p=z/z.sum(1,keepdim=True).clamp_min(1e-12)
  assert torch.isfinite(p).all();return p[0].cpu().numpy()
 if stage=='extract':
  # Validate every native crop before any training-support inference, preserving all rows.
  for i,v in enumerate(train):current=f'bind/{i+1}/1560';label(v,'L8','train')
  save('TRAIN_NATIVE_BINDING.json',native_receipts);assert len(native_receipts)==1560 and len({v['scene'] for v in native_receipts})==18
  (R/'cache').mkdir();receipts=[]
  for i,v in enumerate(train):
   current=f'train/{i+1}/1560';y,valid=label(v,'L8','train');native_receipts.pop();p=forward(v);s=np.log(np.maximum(p.astype(np.float64),1e-12));d=(s[1]-np.logaddexp(s[0],s[2]))[valid];target=y[valid];dp=R/f'cache/{i:04d}_d3.npy';yp=R/f'cache/{i:04d}_labels.npy';np.save(dp,d,allow_pickle=False);np.save(yp,target,allow_pickle=False);receipts.append({'name':v['name'],'scene':v['scene'],'d3_path':dp.relative_to(R).as_posix(),'d3_SHA':sha(dp),'label_path':yp.relative_to(R).as_posix(),'label_SHA':sha(yp),'valid_pixels':len(d),'support':[int((target==k).sum()) for k in range(3)],'Fill_pixels':int((~valid).sum()),'empty_preserved_loss_skipped':not bool(len(d))})
   if (i+1)%50==0:save('CACHE_RECEIPTS.json',receipts);status()
  save('CACHE_RECEIPTS.json',receipts);assert counts['train_input']==1560
 else:
  old=js(R/'EXPECTED_RAW234.json');b3=js(R/'EXPECTED_B3_232.json');out=(R/'EVALUATION.jsonl').open('x',buffering=1)
  for domain,pools in E.items():
   b=biases[domain]['b']
   for split,vs in pools.items():
    for i,v in enumerate(vs):
     current=f'{domain}/{split}/{i+1}/{len(vs)}';y,mask=label(v,domain,split);p=forward(v);raw=p.argmax(0).astype(np.uint8);q=p.astype(np.float64);q[1]*=np.exp(np.float64(b));pred=q.argmax(0).astype(np.uint8);zero=p.astype(np.float64);zero[1]*=np.exp(0.);assert np.array_equal(zero.argmax(0),raw)
     for mode,z,bias in [('Raw',raw,0.),('Source+B',pred,b)]:
      cm=np.bincount(3*y[mask].astype(np.int64)+z[mask],minlength=9).reshape(3,3);psha=hashlib.sha256(z.tobytes()).hexdigest();ex=old[domain][split][v['name']]
      if mode=='Raw':assert cm.tolist()==ex['VALID'] and psha==ex['prediction_SHA'],current+' Raw234 drift'
      if mode=='Source+B' and domain=='SPARCS' and split=='val':assert cm.tolist()==b3[v['scene']]['confusion'] and psha==b3[v['scene']]['prediction_SHA'],current+' B3_232 drift'
      out.write(json.dumps({'domain':domain,'split':split,'scene':v['scene'],'name':v['name'],'usgs_shadows':v.get('usgs_shadows','NA'),'decision':mode,'b':bias,'prediction_SHA':psha,'confusion':cm.tolist()},separators=(',',':'))+'\n')
     if (i+1)%50==0:status()
  out.close();save('EVAL_NATIVE_BINDING.json',native_receipts);assert counts['evaluation_input']==3150
 for h in handles.values():h.close()
 assert all(torch.equal(t,states[n].detach().cpu()) for n,t in frozen.items());assert not denied;done.set();torch.cuda.synchronize();save(f'GPU_{stage}_COMPLETE.json',{'status':'complete','counts':dict(counts),'calls_this_stage':sum(counts.values()),'cumulative_calls':ordinal,'GPU_stage_seconds':time.monotonic()-START,'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_peak_bytes':torch.cuda.max_memory_allocated(),'GPU_reserved_peak_bytes':torch.cuda.max_memory_reserved(),'temporary_cache_bytes':bytes_cache(),'formal_bytes':bytes_formal(),'frozen_all_Source_states_unchanged':True,'forbidden_accesses':denied});status();(R/f'GPU_{stage}_EXIT_CODE').write_text('0\n')
if __name__=='__main__':
 try:main()
 except BaseException:
  (R/f'GPU_{sys.argv[1]}_FAILURE.txt').write_text(traceback.format_exc());save(f'GPU_{sys.argv[1]}_FAILURE_COST.json',{'ledger_calls':len(ledger()),'completed_GPU_stage_seconds':prior_seconds(),'failed_stage_seconds_unsealed':'controller wall receipt required; never reset'});(R/f'GPU_{sys.argv[1]}_EXIT_CODE').write_text('1\n');raise
