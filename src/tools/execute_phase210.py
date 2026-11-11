"""Recover only missing conditional classes after full original R4 mosaics."""
import csv,gc,json,os,resource,sys,threading,time,traceback,zipfile,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
REPO=Path('/home/scv/Cloud-Adapter-light');ROOT=Path('/home/scv/shared/phase210_20261020');BASE=Path('/home/scv/shared/phase111_ocm_runtime');sys.path.insert(0,str(REPO/'src/tools'))
from execute_phase209 import load,sha,write,digest
def main():
 from phase200_common import setup
 setup();lock=load(ROOT/'EXECUTION_LOCK.json')
 assert sha(Path(__file__))==lock['code_sha256'] and sha(Path(__file__).with_name('execute_phase209.py'))==lock['common_helper_sha256']
 for n,h in lock['files'].items():assert sha(ROOT/n)==h,n
 for n,h in lock['sources'].items():assert sha(REPO/'src/tools'/n)==h,n
 assert not (ROOT/'RUN_STATUS.json').exists() and not (ROOT/'CALL_LEDGER.jsonl').exists();(ROOT/'PID').write_text(str(os.getpid()))
 start=time.monotonic();S=dict(status='initializing',PID=os.getpid(),calls=0,completed=[],training=0,fits=0);stop=threading.Event();mutex=threading.Lock()
 import numpy as np,torch
 from install_phase111_runtime import host_free
 from phase119_common import catalog_image,plan_indexes
 from omnicloudmask import model_utils as mu,cloud_mask as cm
 assert torch.__version__=='2.11.0+cu128';torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats();plan=load(ROOT/'MISSING_PLAN.json');t=load(ROOT/'THRESHOLD_REFERENCE.json')['threshold'];assert t==plan['threshold']
 def guard(host=True):
  with mutex:
   S.update(seconds=time.monotonic()-start,cpu_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,gpu_peak_bytes=torch.cuda.max_memory_reserved());assert S['calls']<=512 and S['seconds']<=3600 and S['cpu_peak_bytes']<=16*1024**3 and S['gpu_peak_bytes']<=20*1024**3
   if host:
    free=host_free();S['minimum_host_free_bytes']=min(free,S.get('minimum_host_free_bytes',free));assert free>=4000000000;S['new_bytes']=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file());assert S['new_bytes']<=2000000000
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
   assert S['calls']<512;S['calls']+=1
   with (ROOT/'CALL_LEDGER.jsonl').open('a') as f:f.write(json.dumps(dict(call=S['calls'],kind=kind[0],product=S.get('product'),epoch=time.time()))+'\n')
  depth[0]+=1
 def post(m,args,result):depth[0]-=1
 hooks=[torch.nn.modules.module.register_module_forward_pre_hook(pre),torch.nn.modules.module.register_module_forward_hook(post,always_call=True)]
 try:
  for n,h in [('model_utils.py','f749ceced40f8b0899f56e95e79b9f43ea77b74898376175407c21c26796a1dd'),('cloud_mask.py','9130116e5daa130ceaed28de222484f141abf365c614c0b77aaf7b4f8cf7c33c')]:assert sha(BASE/'assets/vendor/omnicloudmask'/n)==h
  w=next(w for w in load(ROOT/'V4_WEIGHT_LOCK.json')['weights'] if 'regnet' in w['filename']);wp=Path('/home/scv/shared/phase200_20261017/assets')/w['filename'];assert sha(wp)==w['sha256'];row=next(r for r in csv.DictReader((BASE/'assets/vendor/omnicloudmask/model_download_links.csv').open()) if r['version']=='4' and r['file_name']==w['filename'])
  m=mu.load_model_from_weights(model_name=row['timm_model_name'],weights_path=wp,model_library='smp',device=torch.device('cuda'),dtype=torch.float32,in_chans=3,n_out=4,compile_models=False,patch_size=1000,batch_size=1);m.eval()
  for p in m.parameters():p.requires_grad_(False)
  before=digest(m);S['constructor_calls']=S['calls'];assert S['constructor_calls']<=20;receipts=[];(ROOT/'conditional').mkdir(exist_ok=False)
  for q in plan['rows']:
   if not q['missing_positions']:continue
   pid=q['product_id'];S.update(status='recovering',product=pid);guard();old=Path(q['cache_remote_path']);assert sha(old)==q['cache_sha256']
   with np.load(old,allow_pickle=False) as z:flat=z['flat_indices'];joint=z['valid'];oldraw=z['raw_class'];oldscore=z['scores'];oldparent=z['parent_class']
   missing=ROOT/q['missing_file'];assert sha(missing)==q['missing_file_sha256']
   with np.load(missing,allow_pickle=False) as z:commonpositions=z['common_positions'];slots=z['original_slots'];neededflat=z['flat_indices']
   assert np.array_equal(flat[slots],neededflat);assert (oldparent[slots]==2).all() and (oldscore[slots].astype(np.float64)<t).all()
   arr,valid,_=catalog_image(q['group'],read_truth=False);assert np.array_equal(np.flatnonzero(valid),flat) and plan_indexes(arr)==q['actual_patch_coordinates'];yy,xx=np.unravel_index(flat,valid.shape);cpu=np.zeros((4,len(flat)),np.float32);grad=np.zeros(len(flat),np.float32);latest={};actual=[];oldinfer=mu.infer_batch;oldstore=mu.store_results;startcalls=S['calls'];kind[0]='R4_conditional_recovery'
   def single(ms,batch):
    with torch.no_grad():raw=m(batch)
    latest['raw']=raw.detach().cpu().numpy().copy();return raw
   def store(pred_batch,index_batch,pred_tracker,gradient,grad_tracker=None):
    g=gradient.detach().cpu().numpy();axis=np.minimum(np.minimum(np.arange(1,1001),np.arange(1000,0,-1)),300).astype(np.float32)*np.float32(1/300);assert np.max(np.abs(g-axis[:,None]*axis[None,:]))<=1e-7
    for i,(top,bottom,left,right) in enumerate(index_batch):
     inside=(yy>=top)&(yy<bottom)&(xx>=left)&(xx<right);dy=yy[inside]-top;dx=xx[inside]-left;ww=g[dy,dx];cpu[:,inside]+=latest['raw'][i,:,dy,dx].T*ww[None];grad[inside]+=ww;actual.append([top,bottom,left,right])
    latest['author_grad']=grad_tracker;return oldstore(pred_batch,index_batch,pred_tracker,gradient,grad_tracker)
   mu.infer_batch=single;mu.store_results=store
   try:official=cm.predict_from_array(arr,patch_size=1000,patch_overlap=300,batch_size=1,inference_device='cuda',mosaic_device='cpu',inference_dtype=torch.float32,export_confidence=True,softmax_output=True,custom_models=[m],model_version=4,pred_classes=4,compile_models=False,no_data_value=0,apply_no_data_mask=True)
   finally:mu.infer_batch=oldinfer;mu.store_results=oldstore
   assert actual==q['actual_patch_coordinates'] and S['calls']-startcalls==len(actual);assert np.array_equal(grad,latest['author_grad'].numpy()[yy,xx]) and (grad[joint]>0).all()
   logits=cpu/np.maximum(grad[None],np.finfo(np.float32).tiny);assert np.isfinite(logits[:,joint]).all();rawclass=np.argmax(logits,axis=0).astype(np.uint8);classchanges=int((rawclass[joint]!=oldraw[joint]).sum());assert classchanges==0,('raw4 mismatch',pid,classchanges)
   soft=np.exp(logits-logits.max(0,keepdims=True));soft/=soft.sum(0,keepdims=True);prob=np.clip(soft+np.float32(.001),.001,.999)*np.any(arr[:,yy,xx]!=0,axis=0)[None];error=float(np.max(np.abs(prob[3,joint]-oldscore[joint])));officialerr=float(np.max(np.abs(prob[:,joint]-official[:,yy[joint],xx[joint]])));assert error<=1e-5 and officialerr<=1e-5
   condraw=np.argmax(logits[:3,slots],axis=0).astype(np.uint8);condparent=np.array([0,1,1],np.uint8)[condraw];p=ROOT/'conditional'/(pid+'.npz');np.savez_compressed(p,common_positions=commonpositions,original_slots=slots,flat_indices=neededflat,conditional_raw=condraw,conditional_parent=condparent,cached_shadow_scores=oldscore[slots],rerun_shadow_scores=prob[3,slots]);receipts.append(dict(product_id=pid,group_id=q['group']['group_id'],positions=len(slots),file='conditional/'+pid+'.npz',sha256=sha(p),network_calls=S['calls']-startcalls,max_shadow_score_error=error,official_confidence_error=officialerr,default_raw4_mismatch=classchanges,common_support_exact=True,full_original_mosaic_before_conditional_argmax=True,excluded_raw_index=3,remaining_raw_order=[0,1,2],conditional_parent_mapping=[0,1,1],retained_original_raw4_order=['Surface','ThickCloud','ThinCloud','Shadow'],cache_sha256=q['cache_sha256']));S['completed'].append(pid);write(ROOT/'RECOVERY_PROGRESS.json',dict(rows=receipts));print(len(receipts),pid,S['calls'],error,flush=True)
   del arr,valid,official,cpu,grad,latest,logits,soft,prob;gc.collect();guard()
  assert len(receipts)==sum(q['missing_positions']>0 for q in plan['rows']) and sum(q['network_calls'] for q in receipts)==plan['main_calls'] and sum(q['positions'] for q in receipts)==172518
  assert digest(m)==before and all(not x.training for x in m.modules());write(ROOT/'RECOVERY_SEALED.json',dict(rows=receipts,model_sha256=w['sha256'],before=before,after=digest(m),parameters_buffers_unchanged=True,all_eval=True,threshold=t,total_calls=S['calls'],no_ref_semantics=True,only_missing_positions_persisted=True,rule='raw logits original float32 mosaic; argmax over indices0,1,2; map0toSurface,1and2toCloud'))
  S['status']='complete';guard();stop.set();thread.join();(ROOT/'EXIT_CODE').write_text('0')
  names=['RECOVERY_SEALED.json','MISSING_PLAN.json','THRESHOLD_REFERENCE.json','V4_WEIGHT_LOCK.json','EXECUTION_LOCK.json','CACHE_AUDIT.json','CALL_LEDGER.jsonl','RUN_STATUS.json','EXIT_CODE','code/execute_phase210.py','code/execute_phase209.py']+[str(p.relative_to(ROOT)) for p in (ROOT/'conditional').glob('*.npz')];write(ROOT/'DELIVERY_MANIFEST.json',dict(files={n:sha(ROOT/n) for n in names},weights_or_references_included=False))
  archive=ROOT/'phase210_conditional.zip'
  with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
   for n in names+['DELIVERY_MANIFEST.json']:z.write(ROOT/n,n)
  write(ROOT/'DELIVERY_RECEIPT.json',dict(bytes=archive.stat().st_size,sha256=sha(archive),remote_new_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),total_calls=S['calls'],GPU_stage_seconds=S['seconds'],fits=0,training=0))
 except BaseException:
  stop.set();S.update(status='failed',failure=traceback.format_exc());write(ROOT/'RUN_STATUS.json',S);(ROOT/'EXIT_CODE').write_text('1');raise
 finally:
  stop.set()
  for h in hooks:h.remove()
if __name__=='__main__':main()
