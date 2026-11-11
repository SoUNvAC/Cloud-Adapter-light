"""Fixed 5x2 Landsat validation, source non-head states unchanged."""
import argparse,collections,hashlib,json,os,resource,sys,time,traceback,threading
from pathlib import Path
R=Path('/home/scv/shared/phase233_20261027');SRC=Path('/home/scv/Cloud-Adapter-light/src')
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
 assert len(allowed)==1926
 import numpy as np
 refs=js(R/'L8_NATIVE_REFERENCES.json');refmap={v['name']:v for v in refs};assert len(refmap)==963
 oldmap={model:{v['name']:v for v in js(R/f'expected/{model}.json')['per_input']} for model in L['models']}
 assert all(len(v)==963 and all('prediction_SHA' in x for x in v.values()) for v in oldmap.values())
 ref_allowed={str(Path(v['file']).resolve()) for v in refs}
 def arrays_audit(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
   p=Path(os.fsdecode(args[0]))
   if p.suffix.lower() in ['.npz','.npy'] and str(p.resolve()) not in ref_allowed:raise RuntimeError('233 forbids nonreference arrays: '+str(p))
 sys.addaudithook(arrays_audit)
 def native_reference(v):
  entry=refmap[v['name']];p=Path(entry['file']);assert sha(p)==entry['sha256']
  with np.load(p,allow_pickle=False) as z:native=z['native'];valid=z['valid'];truth=z['truth']
  assert native.shape==valid.shape==truth.shape==(512,512) and valid.dtype==bool and np.isin(native,[0,64,128,192,255]).all();assert np.array_equal(valid,native!=0)
  parent=np.zeros(native.shape,np.uint8);parent[native==64]=2;parent[np.isin(native,[192,255])]=1
  assert np.array_equal(truth[valid],parent[valid]) and np.all(truth[~valid]==255);assert int((~valid).sum())==entry['invalid'] and int(valid.sum())==entry['support']
  return parent,valid
 stage='reference_preflight';status();fill=0;total=0
 from PIL import Image
 lut=np.array([0,2,1,1],np.uint8)
 for v in support['L8']:
  parent,valid=native_reference(v);png=np.asarray(Image.open(v['mask_path']));assert png.shape==(512,512) and np.isin(png,[0,1,2,3]).all();assert np.array_equal(parent,lut[png]);fill+=int((~valid).sum());total+=parent.size
 assert fill==33961651 and total==252444672;save('REFERENCE_ACCEPTED.json',{'inputs':963,'scenes':8,'Fill_pixels':fill,'OLD_pixels':total,'VALID_pixels':total-fill,'native_valid_truth_and_oldPNG_exact':True,'reference_source':'Phase317 only native/valid/truth, no models or predictions','reference_manifest_SHA':sha(R/'L8_NATIVE_REFERENCES.json')})
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
  assert digest.hexdigest()==js(R/'EXPECTED_MODEL_ACCEPTED.json')['heads'][name]['interpolated_tensor_digest']
  receipts[name]={'endpoint_SHA':v['endpoint_head_sha256'],'interpolated_tensor_digest':digest.hexdigest(),'alpha':.25,'operator':'CPU float64 then original dtype','parameters':sum(t.numel() for t in heads[name].values())}
 save('MODEL_ACCEPTED.json',{'Source_SHA':L['Source_SHA256'],'heads':receipts,'all_Source_state_loaded_exact':True,'construction_probes':construction,'runtime_versions':{'torch':torch.__version__,'numpy':np.__version__},'settings':{'precision':'FP32','size':512,'argmax':'clamp nonnegative, sum normalized, historical semantic','parent_order':['Surface','Cloud','Shadow'],'cudnn_benchmark':False,'cudnn_deterministic':False}})
 def sethead(name):
  model.decode_head.load_state_dict(heads[name]);model.eval();now=dict(list(model.named_parameters())+list(model.named_buffers()));assert all(torch.equal(t,now[n].detach().cpu()) for n,t in frozen.items())
 (R/'evaluation').mkdir()
 for name in L['models']:
  stage='matrix';sethead(name);per=[];oldgroups=collections.defaultdict(lambda:np.zeros((3,3),np.int64));newgroups=collections.defaultdict(lambda:np.zeros((3,3),np.int64));histgroups=collections.defaultdict(lambda:np.zeros(3,np.int64))
  for i,v in enumerate(support['L8'],1):
   current=f'{name}/{i}/963';parent,valid=native_reference(v);x=torch.from_numpy(load_rgb_image(Path(v['image_path']),512)).cuda();count('matrix_complete_input')
   with torch.inference_mode():
    z=wrapper(x).float().clamp_min(0);pred=(z/z.sum(1,keepdim=True).clamp_min(1e-12)).argmax(1)[0].cpu().numpy().astype(np.uint8)
   old=np.bincount(3*parent.ravel()+pred.ravel(),minlength=9).reshape(3,3);new=np.bincount(3*parent[valid]+pred[valid],minlength=9).reshape(3,3);hist=np.bincount(pred[~valid],minlength=3);prediction_SHA=hashlib.sha256(pred.tobytes()).hexdigest();expected=oldmap[name][v['name']]
   assert old.tolist()==expected['confusion'] and prediction_SHA==expected['prediction_SHA'],current+' old prediction drift';delta=old-new;assert np.array_equal(delta[0],hist) and not delta[1:].any();assert expected['image_SHA']==v['image_sha256'] and expected['mask_SHA']==v['mask_sha256']
   row={'name':v['name'],'scene':v['scene'],'OLD':old.tolist(),'VALID':new.tolist(),'Fill_prediction_histogram':hist.tolist(),'Fill_pixels':int(hist.sum()),'prediction_SHA':prediction_SHA,'old_prediction_SHA_exact':True,'old_confusion_exact':True,'reference_SHA':refmap[v['name']]['sha256'],'image_SHA':v['image_sha256'],'mask_SHA':v['mask_sha256']};per.append(row);oldgroups[v['scene']]+=old;newgroups[v['scene']]+=new;histgroups[v['scene']]+=hist
   if i%100==0:status()
  save(f'evaluation/{name}.json',{'model':name,'per_input':per,'per_scene':{g:{'OLD':oldgroups[g].tolist(),'VALID':newgroups[g].tolist(),'Fill_prediction_histogram':histgroups[g].tolist()} for g in sorted(oldgroups)}});status()
 assert COUNTS['matrix_complete_input']==4815 and sum(COUNTS.values())<=5200 and not denied;done.set();torch.cuda.synchronize()
 save('COMPLETE.json',{'status':'complete','counts':dict(COUNTS),'total_calls':sum(COUNTS.values()),'GPU_stage_seconds':time.monotonic()-START,'GPU_peak_bytes':torch.cuda.max_memory_allocated(),'GPU_reserved_peak_bytes':torch.cuda.max_memory_reserved(),'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'training_updates':0,'test_pixels_read':0,'forbidden_accesses':denied,'all_OLD_predictions_confusions_exact229':True,'no_spatial_prediction_cache':'229 persisted only per-input SHA and confusion; no spatial predictions; no317 model predictions used'});stage='complete';status();(R/'EXIT_CODE').write_text('0\n')
if __name__=='__main__':
 try:main()
 except BaseException:
  done.set();(R/'FAILURE.txt').write_text(traceback.format_exc());save('FAILURE_COST.json',{'counts':dict(COUNTS),'seconds':time.monotonic()-START});status();(R/'EXIT_CODE').write_text('1\n');raise
