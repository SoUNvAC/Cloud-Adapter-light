"""55 fixed validation forwards; preserve normalized semantic scores, not logits."""
from pathlib import Path
import argparse,collections,hashlib,json,os,resource,sys,threading,time,traceback
R=Path('/home/scv/shared/phase230_20261026');SRC=Path('/home/scv/Cloud-Adapter-light/src');START=time.monotonic();done=threading.Event();counts=collections.Counter();stage='preflight';current=None
def js(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(n,v):
 p=R/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False));t.replace(p)
def status():save('RUN_STATUS.json',{'stage':stage,'current':current,'counts':dict(counts),'calls':sum(counts.values()),'GPU_stage_seconds':time.monotonic()-START,'PID':os.getpid()})
def main():
 global stage,current
 assert not (R/'STARTED.json').exists();save('STARTED.json',{'PID':os.getpid(),'epoch':time.time()});(R/'PID').write_text(str(os.getpid()))
 for n,h in js(R/'BUNDLE_SHA.json').items():assert sha(R/n)==h,n
 for n,h in js(R/'RUNTIME_CODE_SHA.json').items():assert sha(SRC/n)==h,n
 lock=js(R/'LOCK.json');rows=js(R/'SPARCS_VAL_INPUTS.json');assert rows==sorted(rows,key=lambda v:(v['scene'],v['name']))
 allowed={str(Path(v[k]).resolve()) for v in rows for k in ['image_path','mask_path']};assert len(allowed)==20;denied=[]
 def audit(e,a):
  if e=='open' and a and isinstance(a[0],(str,bytes,os.PathLike)):
   p=os.path.abspath(os.fsdecode(a[0]))
   if Path(p).suffix.lower() in ['.png','.jpg','.tif','.tiff','.jp2'] and str(Path(p).resolve()) not in allowed:denied.append(p);raise RuntimeError('230 forbids non-SPARCS-validation image access '+p)
 sys.addaudithook(audit)
 source=SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth';assert sha(source)==lock['Source_SHA256'];paths=js(R/'MODEL_PATHS.json')
 for v in lock['heads']:assert sha(paths[f"{v['domain']}_{v['seed']}"])==v['endpoint_head_sha256']
 for v in rows:assert sha(v['image_path'])==v['image_sha256'] and sha(v['mask_path'])==v['mask_sha256']
 os.chdir(SRC);sys.path[:0]=[str(SRC),str(SRC/'tools')];os.environ['XFORMERS_DISABLED']='1'
 import numpy as np,torch
 from export_phase12_onnx import build_wrapper
 from validate_phase12_onnxruntime import load_rgb_image
 from evaluate_phase64_source_only_parent import load_parent_target,SPARCS_TO_PARENT
 torch.set_num_threads(4);torch.use_deterministic_algorithms(False);torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=False
 log=(R/'CALL_LEDGER.jsonl').open('x',buffering=1)
 def count(k):
  assert sum(counts.values())<80 and time.monotonic()-START<1800;counts[k]+=1;log.write(json.dumps({'ordinal':sum(counts.values()),'kind':k,'stage':stage,'current':current})+'\n')
 def guard():
  while not done.wait(10):
   try:
    assert time.monotonic()-START<1800 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<8*1024**3
    if torch.cuda.is_initialized():assert max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())<20*1024**3
    assert sum(p.stat().st_size for p in R.rglob('*') if p.is_file())<175_000_000;status()
   except BaseException:(R/'RESOURCE_FAILURE.txt').write_text(traceback.format_exc());(R/'EXIT_CODE').write_text('71\n');os._exit(71)
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
 for d in ['scores','labels']:(R/d).mkdir()
 receipts=[];repeats=[]
 for name in lock['models']:
  stage='main';model.decode_head.load_state_dict(heads[name]);model.eval();now=dict(list(model.named_parameters())+list(model.named_buffers()));assert all(torch.equal(t,now[n].detach().cpu()) for n,t in frozen.items())
  expected={v['name']:v for v in js(R/f'expected/{name}.json')['per_input']}
  for i,v in enumerate(rows):
   current=f'{name}/{v["scene"]}';status();x=torch.from_numpy(load_rgb_image(Path(v['image_path']),512)).cuda()
   def forward(kind):
    count(kind)
    with torch.inference_mode():
     z=wrapper(x).float().clamp_min(0);s=z.sum(1,keepdim=True);p=z/s.clamp_min(1e-12)
    assert torch.isfinite(z).all() and torch.isfinite(p).all()
    return p[0].cpu().numpy(),s[0,0].cpu().numpy()
   p,s=forward('main_complete_input');assert p.dtype==np.float32 and p.shape==(3,512,512);pred=p.argmax(0).astype(np.uint8);y=load_parent_target(Path(v['mask_path']),SPARCS_TO_PARENT,pred.shape).astype(np.uint8);valid=y!=255
   c=np.bincount(3*y[valid].astype(np.int64)+pred[valid],minlength=9).reshape(3,3);ph=hashlib.sha256(pred.tobytes()).hexdigest();old=expected[v['name']];assert c.tolist()==old['confusion'] and ph==old['prediction_SHA'],current
   zero=s==0;assert not np.any(zero&valid),'zero-sum semantic output on valid reference support'
   nonzero=s>1e-12;normerr=float(np.max(np.abs(p.sum(0)[nonzero]-1))) if nonzero.any() else None;assert normerr is None or normerr<1e-6
   assert not np.any((s>0)&(s<1e-12)&valid),'positive sub-epsilon denominator on valid support; diagnostic stopped'
   pp=R/f'scores/{name}__{v["scene"]}.npy';np.save(pp,p,allow_pickle=False);yp=R/f'labels/{v["scene"]}.npy'
   if name=='Source':np.save(yp,y,allow_pickle=False)
   else:assert np.array_equal(y,np.load(yp,allow_pickle=False))
   rec={'model':name,'scene':v['scene'],'name':v['name'],'score_path':str(pp.relative_to(R)),'score_SHA':sha(pp),'label_path':str(yp.relative_to(R)),'label_SHA':sha(yp),'image_SHA':v['image_sha256'],'original_mask_SHA':v['mask_sha256'],'prediction_SHA':ph,'confusion':c.tolist(),'valid_pixels':int(valid.sum()),'ignore_pixels':int((~valid).sum()),'reference_support':c.sum(1).tolist(),'zero_sum_all_pixels':int(zero.sum()),'zero_sum_valid_pixels':int((zero&valid).sum()),'denominator_clamp_all_pixels':int((s<1e-12).sum()),'normalization_max_abs_error_nonclamped':normerr,'score_log_clamp_valid_channel_counts':[int((p[k][valid]<1e-12).sum()) for k in range(3)],'score_log_clamp_all_channel_counts':[int((p[k]<1e-12).sum()) for k in range(3)],'argmax_and_confusion_exact_229':True}
   receipts.append(rec);save('SCORE_RECEIPTS.json',receipts)
   if i==0:
    stage='repeat';q,qs=forward('repeat_complete_input');diff=float(np.max(np.abs(p-q)));assert np.array_equal(q.argmax(0).astype(np.uint8),pred);repeats.append({'model':name,'scene':v['scene'],'maximum_score_abs_difference':diff,'argmax_equal':True,'scores_bitwise_equal':bool(np.array_equal(p,q))});save('REPEATS.json',repeats);stage='main'
 assert counts['main_complete_input']==50 and counts['repeat_complete_input']==5 and len(receipts)==50 and not denied;done.set();torch.cuda.synchronize()
 save('COMPLETE.json',{'status':'complete','counts':dict(counts),'total_calls':sum(counts.values()),'GPU_stage_wall_seconds':time.monotonic()-START,'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_peak_bytes':torch.cuda.max_memory_allocated(),'GPU_reserved_peak_bytes':torch.cuda.max_memory_reserved(),'persistent_bytes':sum(p.stat().st_size for p in R.rglob('*') if p.is_file()),'training_updates':0,'fits':0,'threshold_search':False,'test_L8_source_pixel_reads':0,'forbidden_accesses':denied,'normalized_semantic_scores_not_calibrated_probability':True});stage='complete';status();(R/'EXIT_CODE').write_text('0\n')
if __name__=='__main__':
 try:main()
 except BaseException:
  done.set();(R/'FAILURE.txt').write_text(traceback.format_exc());save('FAILURE_COST.json',{'counts':dict(counts),'seconds':time.monotonic()-START});status();(R/'EXIT_CODE').write_text('1\n');raise
