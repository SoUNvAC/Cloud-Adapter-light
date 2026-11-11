"""Fixed 5x2 Landsat validation, source non-head states unchanged."""
import argparse,collections,hashlib,json,os,resource,sys,time,traceback,threading
from pathlib import Path
R=Path('/home/scv/shared/phase229_20261026');SRC=Path('/home/scv/Cloud-Adapter-light/src')
START=time.monotonic();COUNTS=collections.Counter();stage='preflight';current=None;done=threading.Event()
def js(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(n,v):
 p=R/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False));t.replace(p)
def status():save('RUN_STATUS.json',{'stage':stage,'current':current,'counts':dict(COUNTS),'total_calls':sum(COUNTS.values()),'seconds':time.monotonic()-START,'PID':os.getpid()})
def main():
 global stage,current
 assert not (R/'STARTED.json').exists();save('STARTED.json',{'PID':os.getpid(),'epoch':time.time()});(R/'PID').write_text(str(os.getpid()))
 for n,h in js(R/'BUNDLE_SHA.json').items():assert sha(R/n)==h,n
 for n,h in js(R/'RUNTIME_CODE_SHA.json').items():assert sha(SRC/n)==h,n
 L=js(R/'LOCK.json');support=js(R/'INPUT_SUPPORT.json');paths=js(R/'MODEL_PATHS.json');source=SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth'
 assert sha(source)==L['Source_SHA256']
 for v in L['heads']:assert sha(paths[f"{v['domain']}_{v['seed']}"])==v['endpoint_head_sha256']
 allowed={str(Path(v[k]).resolve()) for rows in support.values() for v in rows for k in ['image_path','mask_path']}
 denied=[]
 def audit(event,args):
  if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=os.path.abspath(os.fsdecode(args[0]));ext=Path(p).suffix.lower()
  if ext in ['.png','.jpg','.jpeg','.tif','.tiff','.jp2'] and str(Path(p).resolve()) not in allowed:denied.append(p);raise RuntimeError('229 forbids non-validation image access: '+p)
 sys.addaudithook(audit)
 for rows in support.values():
  for v in rows:
   assert sha(v['image_path'])==v['image_sha256'] and sha(v['mask_path'])==v['mask_sha256'],v['name']
 assert len(allowed)==1946
 os.chdir(SRC);sys.path[:0]=[str(SRC),str(SRC/'tools')];os.environ['XFORMERS_DISABLED']='1';os.environ['OMP_NUM_THREADS']='4'
 import numpy as np,torch
 from export_phase12_onnx import build_wrapper
 from validate_phase12_onnxruntime import load_rgb_image
 from evaluate_phase64_source_only_parent import load_parent_target,L8_TO_PARENT,SPARCS_TO_PARENT
 torch.set_num_threads(4);torch.use_deterministic_algorithms(False);torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=False
 ledger=(R/'CALL_LEDGER.jsonl').open('x',buffering=1)
 def count(kind):
  assert sum(COUNTS.values())<5200 and time.monotonic()-START<3600
  COUNTS[kind]+=1;ledger.write(json.dumps({'ordinal':sum(COUNTS.values()),'kind':kind,'stage':stage,'current':current})+'\n')
 def guard():
  while not done.wait(15):
   try:
    assert time.monotonic()-START<3600 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<16*1024**3
    if torch.cuda.is_initialized():assert max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())<20*1024**3
    assert sum(p.stat().st_size for p in R.rglob('*') if p.is_file())<200_000_000
    status()
   except BaseException:
    (R/'RESOURCE_FAILURE.txt').write_text(traceback.format_exc());(R/'EXIT_CODE').write_text('71\n');os._exit(71)
 threading.Thread(target=guard,daemon=True).start()
 stage='build';status()
 # Count top-level module forwards during construction separately, including probes.
 depth=[0];construction=[]
 def pre(m,a):
  if depth[0]==0:count('construction_probe');construction.append(type(m).__name__)
  depth[0]+=1
 def post(m,a,out):depth[0]-=1
 h1=torch.nn.modules.module.register_module_forward_pre_hook(pre);h2=torch.nn.modules.module.register_module_forward_hook(post)
 try:wrapper,cfg=build_wrapper(argparse.Namespace(config='configs/protocol/phase64_source_parent_rgb.py',checkpoint=str(source),precision='fp32',active_block_indices=None))
 finally:h1.remove();h2.remove()
 model=wrapper.segmentor.eval()
 for p in model.parameters():p.requires_grad=False
 all_t=dict(list(model.named_parameters())+list(model.named_buffers()));ck=torch.load(source,map_location='cpu')['state_dict'];assert all(torch.equal(t,all_t[n].detach().cpu()) for n,t in ck.items());del ck
 initial={n:t.detach().cpu().clone() for n,t in model.decode_head.state_dict().items()};frozen={n:t.detach().cpu().clone() for n,t in all_t.items() if not n.startswith('decode_head.')}
 heads={'Source':initial};receipts={}
 for v in L['heads']:
  name=f"{v['domain']}_{v['seed']}";ep=torch.load(paths[name],map_location='cpu')['state_dict'];assert set(ep)==set(initial)
  heads[name]={n:(t.double()+.25*(ep[n].double()-t.double())).to(t.dtype) for n,t in initial.items()}
  assert all(torch.isfinite(t).all() for t in heads[name].values())
  digest=hashlib.sha256()
  for n,t in sorted(heads[name].items()):digest.update(n.encode());digest.update(t.numpy().tobytes())
  receipts[name]={'endpoint_SHA':v['endpoint_head_sha256'],'interpolated_tensor_digest':digest.hexdigest(),'alpha':.25,'operator':'CPU float64 then original dtype','parameters':sum(t.numel() for t in heads[name].values())}
 save('MODEL_ACCEPTED.json',{'Source_SHA':L['Source_SHA256'],'heads':receipts,'all_Source_state_loaded_exact':True,'construction_probes':construction,'runtime_versions':{'torch':torch.__version__,'numpy':np.__version__},'settings':{'precision':'FP32','size':512,'argmax':'clamp nonnegative, sum normalized, historical semantic','parent_order':['Surface','Cloud','Shadow'],'cudnn_benchmark':False,'cudnn_deterministic':False}})
 def sethead(name):
  model.decode_head.load_state_dict(heads[name]);model.eval();now=dict(list(model.named_parameters())+list(model.named_buffers()));assert all(torch.equal(t,now[n].detach().cpu()) for n,t in frozen.items())
 def one(v,domain,kind):
  x=torch.from_numpy(load_rgb_image(Path(v['image_path']),512)).cuda();count(kind)
  with torch.inference_mode():
   z=wrapper(x).float().clamp_min(0);pred=(z/z.sum(1,keepdim=True).clamp_min(1e-12)).argmax(1)[0].cpu().numpy().astype(np.uint8)
  y=load_parent_target(Path(v['mask_path']),L8_TO_PARENT if domain=='L8' else SPARCS_TO_PARENT,pred.shape);valid=y!=255;c=np.bincount(3*y[valid]+pred[valid],minlength=9).reshape(3,3)
  return {'name':v['name'],'scene':v['scene'],'confusion':c.tolist(),'valid_pixels':int(c.sum()),'reference_support':c.sum(1).tolist(),'prediction_SHA':hashlib.sha256(pred.tobytes()).hexdigest(),'image_SHA':v['image_sha256'],'mask_SHA':v['mask_sha256']}
 parity=[]
 # Gate BOTH seeds and Source on both pools before any full matrix evaluation.
 for domain in ['L8','SPARCS']:
  for name,label in [('Source','Source'),(f'{domain}_21701','21701_C_a0.25'),(f'{domain}_21702','21702_C_a0.25')]:
   stage='parity';current=f'{domain}/{name}';status();sethead(name);v=support[domain][0];row=one(v,domain,'parity_complete_input');old=next(x for x in js(R/f'expected/{domain}_{label}.json')['per_input'] if x['name']==v['name']);assert row['confusion']==old['confusion'],current
   if 'prediction_SHA' in old:assert row['prediction_SHA']==old['prediction_SHA'],current
   parity.append({'pool':domain,'head':name,'input':v['name'],'confusion_equal':True,'prediction_SHA_equal':'prediction_SHA' in old})
   save('PARITY_ACCEPTED.json',parity)
 (R/'evaluation').mkdir()
 for domain in ['L8','SPARCS']:
  own=f'{domain}_21701';reference=None
  for name in ['Source','L8_21701','L8_21702','SPARCS_21701','SPARCS_21702']:
   stage='matrix';sethead(name);per=[];groups=collections.defaultdict(lambda:np.zeros((3,3),dtype=np.int64))
   for i,v in enumerate(support[domain],1):
    current=f'{domain}/{name}/{i}';row=one(v,domain,'matrix_complete_input');per.append(row);groups[v['scene']]+=np.array(row['confusion'],dtype=np.int64)
   refs=[v['reference_support'] for v in per]
   if reference is None:reference=refs
   else:assert refs==reference
   save(f'evaluation/{domain}_{name}.json',{'domain':domain,'head':name,'alpha':0 if name=='Source' else .25,'per_input':per,'per_group':{g:c.tolist() for g,c in sorted(groups.items())},'confusion':sum(groups.values()).tolist(),'cache_reused':False});status()
 assert COUNTS['matrix_complete_input']==4865 and COUNTS['parity_complete_input']==6
 assert not denied and sha(source)==L['Source_SHA256'];done.set();torch.cuda.synchronize()
 save('COMPLETE.json',{'status':'complete','counts':dict(COUNTS),'complete_input_forwards':4871,'construction_probes':construction,'total_calls':sum(COUNTS.values()),'GPU_stage_seconds':time.monotonic()-START,'GPU_peak_bytes':torch.cuda.max_memory_allocated(),'GPU_reserved_peak_bytes':torch.cuda.max_memory_reserved(),'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'training_updates':0,'test_pixels_read':0,'forbidden_accesses':denied,'state_checked_at_every_head_change':True,'old_cache_reused':False});stage='complete';status();(R/'EXIT_CODE').write_text('0\n')
if __name__=='__main__':
 try:main()
 except BaseException:
  done.set();(R/'FAILURE.txt').write_text(traceback.format_exc());save('FAILURE_COST.json',{'counts':dict(COUNTS),'seconds':time.monotonic()-START});status();(R/'EXIT_CODE').write_text('1\n');raise
