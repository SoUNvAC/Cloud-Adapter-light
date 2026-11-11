"""Online fixed weight/output mixing; val selection sealed before test decoding."""
from pathlib import Path
import argparse,collections,hashlib,json,os,resource,sys,time,threading,traceback
R=Path('/home/scv/shared/phase234_20261027');SRC=Path('/home/scv/Cloud-Adapter-light/src');START=time.monotonic();COUNTS=collections.Counter();done=threading.Event();stage='preflight';current=None
def js(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(n,v):
 p=R/n;t=p.with_name(p.name+'.tmp_'+str(threading.get_ident()));t.write_text(json.dumps(v,indent=2,allow_nan=False));t.replace(p)
def status():save('RUN_STATUS.json',{'stage':stage,'current':current,'counts':dict(COUNTS),'total_calls':sum(COUNTS.values()),'seconds':time.monotonic()-START,'PID':os.getpid()})
def main():
 global stage,current
 assert not (R/'STARTED.json').exists();save('STARTED.json',{'PID':os.getpid(),'epoch':time.time()});(R/'PID').write_text(str(os.getpid()))
 for n,h in js(R/'BUNDLE_SHA.json').items():assert sha(R/n)==h,n
 for n,h in js(R/'RUNTIME_CODE_SHA.json').items():assert sha(SRC/n)==h,n
 L=js(R/'LOCK.json');E=js(R/'EVAL_INPUTS.json');idx=js(R/'EXPECTED_INDEX.json');paths=js(R/'MODEL_PATHS.json');plan=js(R/'TEST_NATIVE_PLAN.json');source=SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth';assert sha(source)==L['Source_SHA256']
 for v in L['heads']:assert sha(paths[f"{v['domain']}_{v['seed']}"])==v['endpoint_head_sha256']
 refs={v['name']:v for v in js(R/'L8_VAL_NATIVE_REFERENCES.json')};allowed={str(Path(v[k]).resolve()) for pools in E.values() for rows in pools.values() for v in rows for k in ['image_path','mask_path']};rawallowed={str(Path(v['path']).resolve()) for v in plan['scenes'].values()};refallowed={str(Path(v['file']).resolve()) for v in refs.values()};denied=[];testdecoded=False
 def audit(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
   p=Path(os.fsdecode(args[0]));s=str(p.resolve())
   if p.suffix.lower() in ['.png','.jpg','.tif','.tiff','.jp2','.npz','.npy'] and s not in allowed|rawallowed|refallowed:denied.append(s);raise RuntimeError('234 forbidden pixel path '+s)
 sys.addaudithook(audit)
 # Read bytes for identity only, no test pixel decoding or performance analysis here.
 metadata=[]
 for domain,pools in E.items():
  for split,rows in pools.items():
   for v in rows:
    assert sha(v['image_path'])==v['image_sha256'] and sha(v['mask_path'])==v['mask_sha256'],v['name']
   metadata.append({'domain':domain,'split':split,'inputs':len(rows),'identity_SHA_pass':True})
 for g,v in plan['scenes'].items():v['exists']=Path(v['path']).is_file();v['bytes']=Path(v['path']).stat().st_size if v['exists'] else None;v['SHA']=sha(v['path']) if v['exists'] else None
 save('INPUT_IDENTITY_ACCEPTED.json',{'pools':metadata,'test_raw_metadata_only':plan,'test_labels_decoded':False});os.chdir(SRC);sys.path[:0]=[str(SRC),str(SRC/'tools')];os.environ['XFORMERS_DISABLED']='1';os.environ['OMP_NUM_THREADS']='4'
 import numpy as np,torch
 from PIL import Image
 from export_phase12_onnx import build_wrapper
 from validate_phase12_onnxruntime import load_rgb_image
 from evaluate_phase64_source_only_parent import load_parent_target,L8_TO_PARENT,SPARCS_TO_PARENT
 torch.set_num_threads(4);torch.use_deterministic_algorithms(False);torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=False;ledger=(R/'CALL_LEDGER.jsonl').open('x',buffering=1)
 def count(k):
  assert sum(COUNTS.values())<30000 and time.monotonic()-START<7200;COUNTS[k]+=1;ledger.write(json.dumps({'ordinal':sum(COUNTS.values()),'kind':k,'current':current})+'\n')
 def guard():
  while not done.wait(15):
   try:
    assert time.monotonic()-START<7200 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<16*1024**3
    if torch.cuda.is_initialized():assert max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())<20*1024**3
    assert sum(p.stat().st_size for p in R.rglob('*') if p.is_file())<450_000_000;status()
   except BaseException:(R/'RESOURCE_FAILURE.txt').write_text(traceback.format_exc());(R/'EXIT_CODE').write_text('71\n');os._exit(71)
 threading.Thread(target=guard,daemon=True).start();stage='build';depth=[0];probes=[]
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
 initial={n:t.detach().cpu().clone() for n,t in model.decode_head.state_dict().items()};frozen={n:t.detach().cpu().clone() for n,t in all_t.items() if not n.startswith('decode_head.')};heads={};head_receipts={}
 for v in L['heads']:
  name=f"{v['domain']}_{v['seed']}";ep=torch.load(paths[name],map_location='cpu')['state_dict'];assert set(ep)==set(initial)
  for alpha in [.25,.5,.75,1.]:
   key=(v['domain'],v['seed'],alpha);heads[key]={n:(t.double()+alpha*(ep[n].double()-t.double())).to(t.dtype) for n,t in initial.items()};h=hashlib.sha256()
   for n,t in sorted(heads[key].items()):assert torch.isfinite(t).all();h.update(n.encode());h.update(t.numpy().tobytes())
   head_receipts[str(key)]={'endpoint_SHA':v['endpoint_head_sha256'],'tensor_SHA':h.hexdigest(),'alpha':alpha}
 save('MODEL_ACCEPTED.json',{'Source_SHA':sha(source),'heads':head_receipts,'FP32':True,'head_interpolation':'CPUfloat64 then originaldtype','parent_order':['Surface','Cloud','Shadow'],'frozen_state_checks':'before and after each pool; parameters requires_grad false, eval, no backward or optimizer','construction_probes':probes,'versions':{'torch':torch.__version__,'numpy':np.__version__},'backbone_sharing':False,'count_units':'one full wrapper forward per model/input; no shared feature shortcut'})
 def frozencheck():
  now=dict(list(model.named_parameters())+list(model.named_buffers()));assert all(torch.equal(t,now[n].detach().cpu()) for n,t in frozen.items())
 def cm(y,pred,mask):return np.bincount(3*y[mask].astype(np.int64)+pred[mask],minlength=9).reshape(3,3)
 def miou(c):
  d=np.diag(c);u=c.sum(0)+c.sum(1)-d;return float(np.mean(d[u>0]/u[u>0])*100)
 native_lut=np.zeros(256,np.uint8);native_lut[64]=1;native_lut[192]=2;native_lut[255]=3;native_records=[]
 def parentnative(n):
  assert n.shape==(512,512) and np.isin(n,[0,64,128,192,255]).all();y=np.zeros(n.shape,np.uint8);y[n==64]=2;y[np.isin(n,[192,255])]=1;return y,n!=0
 rawhandles={}
 def labels(v,domain,split):
  if domain=='SPARCS':y=load_parent_target(Path(v['mask_path']),SPARCS_TO_PARENT,(512,512));return y,y!=255,None
  if split=='val':
   ref=refs[v['name']];assert sha(ref['file'])==ref['sha256']
   with np.load(ref['file'],allow_pickle=False) as z:n=z['native'];valid=z['valid'];truth=z['truth']
   y,mask=parentnative(n);assert np.array_equal(mask,valid) and np.array_equal(y[mask],truth[mask]);assert int((~mask).sum())==ref['invalid'];identity=ref['sha256']
  else:
   assert (R/'VAL_SELECTION_SEAL.json').exists() and testdecoded
   import rasterio
   from rasterio.windows import Window
   g=v['scene'];entry=plan['scenes'][g];assert entry['exists'],'Missing native '+g
   if g not in rawhandles:assert sha(entry['path'])==entry['SHA'];rawhandles[g]=rasterio.open(entry['path'])
   x,z=int(v['x']),int(v['y']);col,row=(z,x) if plan['verified_axes']=='x_row_y_col' else (x,z);n=rawhandles[g].read(1,window=Window(col,row,512,512),boundless=True,fill_value=0);y,mask=parentnative(n);identity=entry['SHA']
  png=np.asarray(Image.open(v['mask_path']));assert np.array_equal(native_lut[n],png),'Native/PNG mismatch '+v['name'];native_records.append({'split':split,'scene':v['scene'],'name':v['name'],'native_source_SHA':identity,'native_array_SHA':hashlib.sha256(n.tobytes()).hexdigest(),'valid_array_SHA':hashlib.sha256(mask.tobytes()).hexdigest(),'Fill_pixels':int((~mask).sum()),'valid_pixels':int(mask.sum()),'old_PNG_exact':True});return y,mask,n
 (R/'evaluation').mkdir();pooled={};gaps=[];pool_failures=[];old_sha_unavailable=0;overlap={m:{x['name']:x for x in js(R/f'expected/VALID233_{m}.json')['per_input']} for m in ['Source','L8_21701','L8_21702']}
 def evaluate(domain,split):
  global current,stage
  nonlocal old_sha_unavailable
  stage=domain+'_'+split;frozencheck();expected={k.rsplit('/',1)[1]:{v['name']:v for v in js(R/p)['per_input']} for k,p in idx.items() if k.startswith(domain+'/'+split+'/')};acc={};expectedchecks=collections.Counter();f=(R/f'evaluation/{domain}_{split}.jsonl').open('x',buffering=1)
  for i,v in enumerate(E[domain][split]):
   current=f'{domain}/{split}/{i+1}/{len(E[domain][split])}';y,valid,native=labels(v,domain,split);oldmask=y!=255;mask=valid if domain=='L8' else oldmask;x=torch.from_numpy(load_rgb_image(Path(v['image_path']),512)).cuda();scores={}
   for seed,alpha in [(0,0.),(21701,1.),(21702,1.),(21701,.25),(21701,.5),(21701,.75),(21702,.25),(21702,.5),(21702,.75)]:
    model.decode_head.load_state_dict(initial if seed==0 else heads[domain,seed,alpha]);model.eval();count('main_complete_input')
    with torch.inference_mode():
     z=wrapper(x).float().clamp_min(0);p=(z/z.sum(1,keepdim=True).clamp_min(1e-12))[0].cpu().numpy()
    assert np.isfinite(p).all();scores[seed,alpha]=p;pred=p.argmax(0).astype(np.uint8);old=cm(y,pred,oldmask);key='Source' if seed==0 else f'{seed}_{alpha:g}';ex=expected[key][v['name']];assert old.tolist()==ex['confusion'],current+'/'+key+' cm drift';psha=hashlib.sha256(pred.tobytes()).hexdigest();h=ex.get('prediction_SHA',ex.get('prediction_sha256'))
    if h:assert psha==h,current+'/'+key+' SHA drift';expectedchecks['prediction_SHA']+=1
    else:old_sha_unavailable+=1;expectedchecks['SHA_unavailable_confusion_only']+=1
    if domain=='L8' and split=='val' and alpha in [0.,.25]:
     m='Source' if seed==0 else f'L8_{seed}';ex233=overlap[m][v['name']];assert cm(y,pred,valid).tolist()==ex233['VALID'] and psha==ex233['prediction_SHA']
   for seed in [21701,21702]:
    for alpha in L['alphas']:
     w=scores[0,0.] if alpha==0 else scores[seed,float(alpha)];p=scores[0,0.] if alpha==0 else scores[seed,1.] if alpha==1 else (np.float32(1-alpha)*scores[0,0.]+np.float32(alpha)*scores[seed,1.]);wp=w.argmax(0).astype(np.uint8);pp=p.argmax(0).astype(np.uint8)
     if alpha in [0,1]:assert np.array_equal(w,p) and np.array_equal(wp,pp)
     for strategy,score,pred in [('W',w,wp),('P',p,pp)]:
      old=cm(y,pred,oldmask);val=cm(y,pred,mask);hist=np.bincount(pred[~mask],minlength=3) if domain=='L8' else np.zeros(3,np.int64)
      if domain=='L8':delta=old-val;assert np.array_equal(delta[0],hist) and not delta[1:].any()
      row={'domain':domain,'split':split,'name':v['name'],'scene':v['scene'],'usgs_shadows':v.get('usgs_shadows','NA'),'seed':seed,'strategy':strategy,'alpha':alpha,'prediction_SHA':hashlib.sha256(pred.tobytes()).hexdigest(),'OLD':old.tolist(),'VALID':val.tolist(),'Fill_prediction_histogram':hist.tolist()};f.write(json.dumps(row,separators=(',',':'))+'\n');key=(seed,strategy,float(alpha));acc.setdefault(key,{'OLD':np.zeros((3,3),np.int64),'VALID':np.zeros((3,3),np.int64)});acc[key]['OLD']+=old;acc[key]['VALID']+=val
     diff=np.abs(w[:,mask].astype(np.float64)-p[:,mask].astype(np.float64));gaps.append({'domain':domain,'split':split,'scene':v['scene'],'name':v['name'],'seed':seed,'alpha':alpha,'valid_pixels':int(mask.sum()),'score_abs_sum_by_class':diff.sum(1).tolist(),'score_abs_max_by_class':diff.max(1).tolist() if mask.any() else [None]*3,'argmax_disagree_pixels':int(np.count_nonzero((wp!=pp)&mask))})
   if (i+1)%20==0:status()
  f.close();frozencheck();res=[{'domain':domain,'split':split,'seed':seed,'strategy':st,'alpha':a,'OLD':v['OLD'].tolist(),'VALID':v['VALID'].tolist(),'selection_mIoU':miou(v['VALID'])} for (seed,st,a),v in acc.items()];pooled[domain,split]=res;save(f'{domain}_{split}_SUMMARY.json',{'rows':res,'expected_identity_checks':dict(expectedchecks),'per_input_SHA':sha(R/f'evaluation/{domain}_{split}.jsonl')});save('NATIVE_REFERENCES.json',native_records);save('SCORE_DIFFERENCES.json',gaps);status()
 for domain in ['L8','SPARCS']:evaluate(domain,'val')
 selections=[]
 for domain in ['L8','SPARCS']:
  for seed in L['seeds']:
   for strategy in ['W','P']:
    candidates=[v for v in pooled[domain,'val'] if v['seed']==seed and v['strategy']==strategy];best=sorted(candidates,key=lambda v:(-v['selection_mIoU'],v['alpha']))[0];selections.append({'domain':domain,'seed':seed,'strategy':strategy,'selected_alpha':best['alpha'],'val_mIoU':best['selection_mIoU'],'candidates':candidates})
 save('VAL_SELECTION_LOCK.json',{'epoch':time.time(),'selections':selections,'input_SHA':sha(R/'EVAL_INPUTS.json'),'code_SHA':sha(Path(__file__)),'historical_test_exposed':True,'test_native_decoded_before_seal':False,'test_performance_results_read_before_seal':False,'rule':'maximum pooled val mIoU, exact ties smaller alpha, L8VALID'});save('VAL_SELECTION_SEAL.json',{'epoch':time.time(),'SHA':sha(R/'VAL_SELECTION_LOCK.json')});testdecoded=True;save('TEST_ACCESS_STARTED.json',{'epoch':time.time(),'selection_SHA':sha(R/'VAL_SELECTION_LOCK.json')})
 for domain in ['L8','SPARCS']:
  try:evaluate(domain,'test')
  except (AssertionError,FileNotFoundError) as e:pool_failures.append({'domain':domain,'split':'test','reason':str(e),'traceback':traceback.format_exc()});save('POOL_FAILURES.json',pool_failures)
 for h in rawhandles.values():h.close()
 done.set();torch.cuda.synchronize();assert not denied;save('COMPLETE.json',{'status':'partial' if pool_failures else 'complete','pool_failures':pool_failures,'counts':dict(COUNTS),'total_calls':sum(COUNTS.values()),'GPU_stage_seconds':time.monotonic()-START,'GPU_peak_bytes':torch.cuda.max_memory_allocated(),'GPU_reserved_peak_bytes':torch.cuda.max_memory_reserved(),'CPU_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'old_SHA_unavailable_count':old_sha_unavailable,'source_once_per_input':True,'backbone_sharing':False,'new_training_updates':0,'test_after_val_seal':True,'forbidden_accesses':denied});stage='complete';status();(R/'EXIT_CODE').write_text('0\n' if not pool_failures else '2\n')
if __name__=='__main__':
 try:main()
 except BaseException:done.set();(R/'FAILURE.txt').write_text(traceback.format_exc());save('FAILURE_COST.json',{'counts':dict(COUNTS),'seconds':time.monotonic()-START});status();(R/'EXIT_CODE').write_text('1\n');raise
