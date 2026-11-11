"""One frozen V4 R4 pass on all53 calibration representatives; zero label semantics."""
import csv,gc,hashlib,json,os,resource,sys,threading,time,traceback,zipfile
from pathlib import Path
sys.dont_write_bytecode=True
REPO=Path('/home/scv/Cloud-Adapter-light');ROOT=Path('/home/scv/shared/phase209_20261020');BASE=Path('/home/scv/shared/phase111_ocm_runtime')
sys.path.insert(0,str(REPO/'src/tools'))
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def write(p,d):
 p=Path(p);tmp=p.with_name(p.name+'.writing');tmp.write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf8');tmp.replace(p)
def digest(m):
 h=hashlib.sha256()
 for n,v in sorted(m.state_dict().items()):h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def main():
 from phase200_common import setup
 setup();lock=load(ROOT/'EXECUTION_LOCK.json')
 for n,h in lock['files'].items():assert sha(ROOT/n)==h
 for n,h in lock['sources'].items():assert sha(REPO/'src/tools'/n)==h
 assert sha(Path(__file__))==lock['new_code_sha256']
 assert not (ROOT/'RUN_STATUS.json').exists() and not (ROOT/'CALL_LEDGER.jsonl').exists()
 (ROOT/'PID').write_text(str(os.getpid()));start=time.monotonic();S=dict(status='constructing',calls=0,PID=os.getpid(),completed=[],fits=0,training=0);stop=threading.Event();mutex=threading.Lock();ledger=[]
 import numpy as np,torch
 from install_phase111_runtime import host_free
 from phase119_common import catalog_image,plan_indexes
 from omnicloudmask import model_utils as mu,cloud_mask as cm
 assert torch.__version__=='2.11.0+cu128';torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
 def guard(host=True):
  with mutex:
   S.update(seconds=time.monotonic()-start,cpu_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,gpu_peak_bytes=torch.cuda.max_memory_reserved())
   assert S['calls']<=4000 and S['seconds']<=7200 and S['cpu_peak_bytes']<=16*1024**3 and S['gpu_peak_bytes']<=20*1024**3
   if host:
    free=host_free();S['minimum_host_free_bytes']=min(free,S.get('minimum_host_free_bytes',free));assert free>=4000000000
    S['new_bytes']=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file());assert S['new_bytes']<=4000000000
   write(ROOT/'RUN_STATUS.json',S)
 guard()
 def monitor():
  try:
   while not stop.wait(20):guard()
  except BaseException:
   (ROOT/'MONITOR_FAILURE.txt').write_text(traceback.format_exc());(ROOT/'EXIT_CODE').write_text('1');os._exit(1)
 thread=threading.Thread(target=monitor,daemon=True);thread.start();depth=[0];kind=['constructor']
 def pre(m,args):
  if depth[0]==0:
   assert S['calls']<4000;S['calls']+=1;record=dict(call=S['calls'],kind=kind[0],product=S.get('product'),epoch=time.time());ledger.append(record)
   with (ROOT/'CALL_LEDGER.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  depth[0]+=1
 def post(m,args,result):depth[0]-=1
 hooks=[torch.nn.modules.module.register_module_forward_pre_hook(pre),torch.nn.modules.module.register_module_forward_hook(post,always_call=True)]
 try:
  w=next(w for w in load(ROOT/'V4_WEIGHT_LOCK.json')['weights'] if 'regnet' in w['filename']);wp=Path('/home/scv/shared/phase200_20261017/assets')/w['filename'];assert sha(wp)==w['sha256']
  assert sha(BASE/'assets/vendor/omnicloudmask/model_utils.py')=='f749ceced40f8b0899f56e95e79b9f43ea77b74898376175407c21c26796a1dd'
  assert sha(BASE/'assets/vendor/omnicloudmask/cloud_mask.py')=='9130116e5daa130ceaed28de222484f141abf365c614c0b77aaf7b4f8cf7c33c'
  row=next(r for r in csv.DictReader((BASE/'assets/vendor/omnicloudmask/model_download_links.csv').open()) if r['version']=='4' and r['file_name']==w['filename'])
  model=mu.load_model_from_weights(model_name=row['timm_model_name'],weights_path=wp,model_library='smp',device=torch.device('cuda'),dtype=torch.float32,in_chans=3,n_out=4,compile_models=False,patch_size=1000,batch_size=1)
  model.eval()
  for p in model.parameters():p.requires_grad_(False)
  before=digest(model);S['constructor_calls']=S['calls'];assert S['constructor_calls']<=20;records=[]
  (ROOT/'scores').mkdir(exist_ok=False)
  for q in load(ROOT/'CALIBRATION_PLAN.json')['rows']:
   pid=q['product_id'];S.update(status='scoring',product=pid);guard();arr,valid,_=catalog_image(q['group'],read_truth=False);assert plan_indexes(arr)==q['actual_patch_coordinates'];pre_calls=S['calls'];kind[0]='calibration_R4'
   with torch.no_grad():official=cm.predict_from_array(arr,patch_size=1000,patch_overlap=300,batch_size=1,inference_device='cuda',mosaic_device='cpu',inference_dtype=torch.float32,export_confidence=True,softmax_output=True,custom_models=[model],model_version=4,pred_classes=4,compile_models=False,no_data_value=0,apply_no_data_mask=True)
   scores=official[3].astype(np.float32);flat=np.flatnonzero(valid);assert np.isfinite(scores.reshape(-1)[flat]).all();assert S['calls']-pre_calls==len(q['actual_patch_coordinates']);assert np.any(arr[:,valid]!=0,axis=0).all()
   path=ROOT/'scores'/(pid+'.npz');np.savez_compressed(path,scores=scores.reshape(-1)[flat],flat_indices=flat,valid=np.ones(len(flat),bool));records.append(dict(product_id=pid,group_id=q['group']['group_id'],sha256=sha(path),bytes=path.stat().st_size,count=len(flat),image_member_sha256=q['group']['image_member_sha256'],mask_member_sha256=q['group']['mask_member_sha256'],network_calls=S['calls']-pre_calls));S['completed'].append(pid);write(ROOT/'SCORES_PROGRESS.json',dict(rows=records));print(len(records),pid,S['calls'],flush=True);del arr,valid,official,scores,flat;gc.collect();guard()
  assert len(records)==53 and sum(q['network_calls'] for q in records)==212 and digest(model)==before and all(not m.training for m in model.modules())
  write(ROOT/'SCORES_SEALED.json',dict(rows=records,model_weight_sha256=w['sha256'],model_state_before=before,model_state_after=digest(model),all_parameters_buffers_unchanged=True,all_eval=True,reference_semantics_accessed=False,full53_support=True,total_calls=S['calls'],constructor_calls=S['constructor_calls'],postprocessing='officialV4softmaxplus.001clip.001to.999',phase208_same_model_and_postprocessing=True))
  S['status']='complete';guard();stop.set();thread.join();(ROOT/'EXIT_CODE').write_text('0')
  names=['SCORES_SEALED.json','CALIBRATION_PLAN.json','V4_WEIGHT_LOCK.json','EXECUTION_LOCK.json','CACHE_IDENTITY_AUDIT.json','CALL_LEDGER.jsonl','RUN_STATUS.json','EXIT_CODE','code/execute_phase209.py']+[str(p.relative_to(ROOT)) for p in (ROOT/'scores').glob('*.npz')]
  manifest={n:sha(ROOT/n) for n in names};write(ROOT/'DELIVERY_MANIFEST.json',dict(files=manifest,reference_included=False,weights_included=False))
  with zipfile.ZipFile(ROOT/'phase209_calibration.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
   for n in names+['DELIVERY_MANIFEST.json']:z.write(ROOT/n,n)
  path=ROOT/'phase209_calibration.zip';write(ROOT/'DELIVERY_RECEIPT.json',dict(bytes=path.stat().st_size,sha256=sha(path),remote_new_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),total_calls=S['calls'],GPU_stage_seconds=S['seconds'],no_new_fits=True))
 except BaseException:
  stop.set();S.update(status='failed',failure=traceback.format_exc());write(ROOT/'RUN_STATUS.json',S);(ROOT/'EXIT_CODE').write_text('1');raise
 finally:
  stop.set()
  for h in hooks:h.remove()
if __name__=='__main__':main()
