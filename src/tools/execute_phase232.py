"""55 fixed validation forwards; preserve normalized semantic scores, not logits."""
from pathlib import Path
import argparse,collections,hashlib,json,os,resource,sys,threading,time,traceback
R=Path('/home/scv/shared/phase232_20261027');SRC=Path('/home/scv/Cloud-Adapter-light/src');START=time.monotonic();done=threading.Event();counts=collections.Counter();stage='preflight';current=None
def js(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(n,v):
 p=R/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False));t.replace(p)
def status():save('GPU_STATUS.json',{'stage':stage,'current':current,'counts':dict(counts),'calls':sum(counts.values()),'GPU_stage_seconds':time.monotonic()-START,'PID':os.getpid()})
def main():
 global stage,current
 assert not (R/'GPU_STARTED.json').exists();save('GPU_STARTED.json',{'PID':os.getpid(),'epoch':time.time()});(R/'GPU_PID').write_text(str(os.getpid()))
 for n,h in js(R/'BUNDLE_SHA.json').items():assert sha(R/n)==h,n
 for n,h in js(R/'RUNTIME_CODE_SHA.json').items():assert sha(SRC/n)==h,n
 lock=js(R/'LOCK.json');rows=js(R/'CALIBRATION_INPUTS.json');assert rows==sorted(rows,key=lambda v:(v['scene'],v['name']))
 allowed={str(Path(v[k]).resolve()) for v in rows for k in ['image_path','mask_path']};assert len(allowed)==120;denied=[]
 def audit(e,a):
  if e=='open' and a and isinstance(a[0],(str,bytes,os.PathLike)):
   p=os.path.abspath(os.fsdecode(a[0]))
   if Path(p).suffix.lower() in ['.png','.jpg','.tif','.tiff','.jp2'] and str(Path(p).resolve()) not in allowed:denied.append(p);raise RuntimeError('230 forbids non-calibration image access '+p)
 sys.addaudithook(audit)
 source=SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth';assert sha(source)==lock['Source_SHA256'];paths=js(R/'MODEL_PATHS.json')
 for v in lock['heads']:assert sha(paths[f"{v['domain']}_{v['seed']}"])==v['endpoint_head_sha256']
 actual=[{**v,'image_sha256':sha(v['image_path']),'mask_sha256':sha(v['mask_path'])} for v in rows];save('CALIBRATION_ACTUAL_INPUTS.json',actual);rows=actual;save('INPUTS_SEALED.json',{'epoch':time.time(),'actual_inputs_SHA':sha(R/'CALIBRATION_ACTUAL_INPUTS.json'),'scenes':60,'validation_cache_accessed':False})
 os.chdir(SRC);sys.path[:0]=[str(SRC),str(SRC/'tools')];os.environ['XFORMERS_DISABLED']='1'
 import numpy as np,torch
 from export_phase12_onnx import build_wrapper
 from validate_phase12_onnxruntime import load_rgb_image
 from evaluate_phase64_source_only_parent import load_parent_target,SPARCS_TO_PARENT
 torch.set_num_threads(4);torch.use_deterministic_algorithms(False);torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=False
 log=(R/'CALL_LEDGER.jsonl').open('x',buffering=1)
 def count(k):
  assert sum(counts.values())<220 and time.monotonic()-START<1800;counts[k]+=1;log.write(json.dumps({'ordinal':sum(counts.values()),'kind':k,'stage':stage,'current':current})+'\n')
 def guard():
  while not done.wait(10):
   try:
    assert time.monotonic()-START<1800 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<8*1024**3
    if torch.cuda.is_initialized():assert max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())<20*1024**3
    assert sum(p.stat().st_size for p in R.rglob('*') if p.is_file())<420_000_000;status()
   except BaseException:(R/'RESOURCE_FAILURE.txt').write_text(traceback.format_exc());(R/'GPU_EXIT_CODE').write_text('71\n');os._exit(71)
 threading.Thread(target=guard,daemon=True).start();stage='build';status();depth=[0];probes=[]
 def pre(m,a):
  if depth[0]==0:count('construction_probe');probes.append(type(m).__name__)
  depth[0]+=1
 def post(m,a,o):depth[0]-=1
 h1=torch.nn.modules.module.register_module_forward_pre_hook(pre);h2=torch.nn.modules.module.register_module_forward_hook(post)
 try:wrapper,cfg=build_wrapper(argparse.Namespace(config='configs/protocol/phase64_source_parent_rgb.py',checkpoint=str(source),precision='fp32',active_block_indices=None))
 finally:h1.remove();h2.remove()
 model=wrapper.segmentor.eval()
 for p in model.parameters():p.requires_grad=False
 all_t=dict(list(model.named_parameters())+list(model.named_buffers()));ck=torch.load(source,map_location='cpu')['state_dict'];assert all(torch.equal(t,all_t[n].detach().cpu()) for n,t in ck.items());del ck
 initial={n:t.detach().cpu().clone() for n,t in model.decode_head.state_dict().items()};frozen={n:t.detach().cpu().clone() for n,t in all_t.items() if not n.startswith('decode_head.')};heads={'Source':initial};head_receipts={}
 for v in lock['heads']:
  name=f"{v['domain']}_{v['seed']}";ep=torch.load(paths[name],map_location='cpu')['state_dict'];assert set(ep)==set(initial)
  heads[name]={n:(t.double()+.25*(ep[n].double()-t.double())).to(t.dtype) for n,t in initial.items()};digest=hashlib.sha256()
  for n,t in sorted(heads[name].items()):assert torch.isfinite(t).all();digest.update(n.encode());digest.update(t.numpy().tobytes())
  assert digest.hexdigest()==js(R/'EXPECTED_MODEL_ACCEPTED.json')['heads'][name]['interpolated_tensor_digest'];head_receipts[name]={'endpoint_SHA':v['endpoint_head_sha256'],'interpolated_tensor_digest':digest.hexdigest(),'alpha':.25}
 save('MODEL_ACCEPTED.json',{'Source_SHA':sha(source),'heads':head_receipts,'frozen_nonhead_checked_each_change':True,'versions':{'torch':torch.__version__,'numpy':np.__version__},'construction_probes':probes})
 (R/'d_cache').mkdir();OLD=Path('/home/scv/shared/phase231_20261026');old_receipts={(v['model'],v['scene']):v for v in js(OLD/'CALIBRATION_RECEIPTS.json')};assert js(R/'CALIBRATION_ACTUAL_INPUTS.json')==js(OLD/'CALIBRATION_ACTUAL_INPUTS.json')
 receipts=[];repeats=[]
 for name in lock['models']:
  stage='calibration_main';model.decode_head.load_state_dict(heads[name]);model.eval();now=dict(list(model.named_parameters())+list(model.named_buffers()));assert all(torch.equal(t,now[n].detach().cpu()) for n,t in frozen.items())
  for i,v in enumerate(rows):
   current=f'{name}/{v["scene"]}';status();x=torch.from_numpy(load_rgb_image(Path(v['image_path']),512)).cuda()
   def forward(kind):
    count(kind)
    with torch.inference_mode():
     z=wrapper(x).float().clamp_min(0);s=z.sum(1,keepdim=True);p=z/s.clamp_min(1e-12)
    assert torch.isfinite(z).all() and torch.isfinite(p).all()
    return p[0].cpu().numpy(),s[0,0].cpu().numpy()
   p,s=forward('main_complete_input');y=load_parent_target(Path(v['mask_path']),SPARCS_TO_PARENT,(512,512)).astype(np.uint8);valid=y!=255;assert not np.any((s==0)&valid)
   assert np.max(np.abs(p.sum(0)[valid]-1))<1e-6
   d=np.log(np.maximum(p[1],1e-12))-np.log(np.maximum(p[0],1e-12));assert d.dtype==np.float32 and np.isfinite(d).all()
   old=old_receipts[name,v['scene']];op=OLD/old['d_path'];yp=OLD/old['label_path'];assert sha(op)==old['d_SHA'] and sha(yp)==old['label_SHA'];assert np.array_equal(d,np.load(op,allow_pickle=False)) and np.array_equal(y,np.load(yp,allow_pickle=False)),current+' drift'
   logs=np.log(np.maximum(p.astype(np.float64),1e-12));d3=logs[1]-np.logaddexp(logs[0],logs[2]);dp=R/f'd_cache/{name}__{v["scene"]}.npy';np.save(dp,d3,allow_pickle=False)
   yy=y[valid];dv=d3[valid];ce=np.logaddexp(np.logaddexp(logs[0],logs[1]),logs[2])[valid]-np.take_along_axis(logs,y.clip(0,2)[None],axis=0)[0][valid];bce=np.logaddexp(0.,dv)-(yy==1)*dv
   K=float(np.mean(ce-bce,dtype=np.float64));assert abs(float(ce.mean())-(K+float(bce.mean())))<1e-12
   rec={'model':name,'scene':v['scene'],'d_path':str(dp.relative_to(R)),'d_SHA':sha(dp),'label_path':str(yp),'label_SHA':sha(yp),'image_SHA':v['image_sha256'],'mask_SHA':v['mask_sha256'],'support':[int((y==k).sum()) for k in range(3)],'ignore':int((y==255).sum()),'K':K,'direct_L3_zero':float(ce.mean()),'d2_bitwise_equal':True,'label_bitwise_equal':True};receipts.append(rec);save('CALIBRATION_RECEIPTS.json',receipts)
   if i==0:
    q,qs=forward('repeat_complete_input');repeats.append({'model':name,'scene':v['scene'],'max_score_abs_difference':float(np.max(np.abs(p-q))),'scores_bitwise_equal':bool(np.array_equal(p,q))});assert np.array_equal(p.argmax(0),q.argmax(0));save('REPEATS.json',repeats)
 assert counts['main_complete_input']==180 and counts['repeat_complete_input']==3 and len(receipts)==180 and not denied;done.set();torch.cuda.synchronize()
 save('GPU_COMPLETE.json',{'status':'complete','counts':dict(counts),'total_calls':sum(counts.values()),'GPU_stage_wall_seconds':time.monotonic()-START,'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_peak_bytes':torch.cuda.max_memory_allocated(),'GPU_reserved_peak_bytes':torch.cuda.max_memory_reserved(),'persistent_bytes':sum(p.stat().st_size for p in R.rglob('*') if p.is_file()),'network_parameter_updates':0,'validation_L8_source_test_pixel_reads':0,'forbidden_accesses':denied});stage='GPU_complete';status();(R/'GPU_EXIT_CODE').write_text('0\n')
if __name__=='__main__':
 try:main()
 except BaseException:
  done.set();(R/'GPU_FAILURE.txt').write_text(traceback.format_exc());save('FAILURE_COST.json',{'counts':dict(counts),'seconds':time.monotonic()-START});status();(R/'GPU_EXIT_CODE').write_text('1\n');raise
