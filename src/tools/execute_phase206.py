"""Phase206 fixed W/BN swaps; no training or fitting; immutable Phase205 reuse."""
import csv,datetime,gc,hashlib,inspect,json,os,resource,subprocess,sys,threading,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
REPO=Path('/home/scv/Cloud-Adapter-light');ROOT=Path('/home/scv/shared/phase206_20261019');BASE=Path('/home/scv/shared/phase111_ocm_runtime');sys.path.insert(0,str(REPO/'src/tools'))
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.writing.'+str(threading.get_ident()));t.write_text(json.dumps(dict(v) if isinstance(v,dict) else v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def digest(state):
 h=hashlib.sha256()
 for n,v in sorted(state.items()):h.update(n.encode());h.update(str(v.dtype).encode());h.update(str(list(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
S={};START=time.monotonic();STOP=threading.Event();OUT=None;PRIOR=0;PRESECONDS=0
def setup(stage):
 global OUT,PRIOR,PRESECONDS,S
 from phase200_common import setup as init
 init();cfg=load(ROOT/'INPUT_LOCK.json')
 for n,h in cfg['files'].items():assert sha(ROOT/n)==h,n
 for n,h in cfg['source_sha256'].items():assert sha(REPO/n)==h,n
 for n,h in load(ROOT/'CODE_LOCK.json')['files'].items():assert sha(ROOT/'code'/n)==h,n
 assert sha(BASE/'assets/vendor/omnicloudmask/model_utils.py')=='f749ceced40f8b0899f56e95e79b9f43ea77b74898376175407c21c26796a1dd'
 assert sha(BASE/'assets/vendor/omnicloudmask/cloud_mask.py')=='9130116e5daa130ceaed28de222484f141abf365c614c0b77aaf7b4f8cf7c33c'
 from phase200_common import ROOT as old200
 for w in load(ROOT/'V4_WEIGHT_LOCK.json')['weights']:assert sha(old200/'assets'/w['filename'])==w['sha256']
 OUT=ROOT/stage;OUT.mkdir(exist_ok=False);(OUT/'PID').write_text(str(os.getpid()))
 PRIOR=sum(1 for _ in (ROOT/'CALL_LEDGER.jsonl').open()) if (ROOT/'CALL_LEDGER.jsonl').exists() else 0
 PRESECONDS=load(ROOT/'CONTROLLER_STATUS.json').get('stage_seconds_total',0) if (ROOT/'CONTROLLER_STATUS.json').exists() else 0
 S=dict(status='initializing',stage=stage,PID=os.getpid(),calls=0,prior_calls=PRIOR,kind='constructor',step=0)
 import torch
 torch.set_num_threads(4);assert torch.__version__=='2.11.0+cu128' and torch.cuda.is_available() and not torch.is_autocast_enabled()
 torch.cuda.reset_peak_memory_stats()
 from install_phase111_runtime import host_free
 def guard(host=False):
  S.update(stage_seconds=time.monotonic()-START,cumulative_stage_seconds=PRESECONDS+time.monotonic()-START,cumulative_calls=PRIOR+S['calls'],rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,gpu_peak_reserved=torch.cuda.max_memory_reserved())
  assert S['cumulative_calls']<=8000 and S['cumulative_stage_seconds']<=14400
  assert S['rss_peak_bytes']<=16*1024**3 and S['gpu_peak_reserved']<=20*1024**3
  if host:
   free=host_free();S['host_min_free_bytes']=min(S.get('host_min_free_bytes',free),free);assert free>=4000000000
   size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file());S['new_bytes']=size;assert size<=6000000000
  write(OUT/'RUN_STATUS.json',S)
 guard(True)
 def monitor():
  try:
   while not STOP.wait(20):guard(True)
  except BaseException:
   S.update(status='failed_resource',failure=traceback.format_exc());write(OUT/'RUN_STATUS.json',S);(OUT/'EXIT_CODE').write_text('1\n');os._exit(1)
 thread=threading.Thread(target=monitor,daemon=True);thread.start();depth=[0]
 def pre(m,args):
  if depth[0]==0:
   assert PRIOR+S['calls']<8000;S['calls']+=1;kind=S['kind'];S[kind+'_calls']=S.get(kind+'_calls',0)+1
   with (ROOT/'CALL_LEDGER.jsonl').open('a') as f:f.write(json.dumps(dict(global_call=PRIOR+S['calls'],stage=stage,kind=kind,step=S['step'],epoch=time.time()))+'\n')
  depth[0]+=1
 def post(m,args,result):depth[0]-=1
 hooks=[torch.nn.modules.module.register_module_forward_pre_hook(pre),torch.nn.modules.module.register_module_forward_hook(post,always_call=True)]
 return guard,thread,hooks
def build(member):
 import torch
 from omnicloudmask import model_utils as mu
 from phase200_common import ROOT as old200
 index=list(csv.DictReader((BASE/'assets/vendor/omnicloudmask/model_download_links.csv').open()));weights=load(ROOT/'V4_WEIGHT_LOCK.json')['weights'];w=next(w for w in weights if ('regnet' in w['filename'])==(member=='R'));row=next(r for r in index if r['version']=='4' and r['file_name']==w['filename']);assert row['model_library']=='smp'
 mu.load_model_from_weights.cache_clear();S['kind']='constructor'
 return mu.load_model_from_weights(model_name=row['timm_model_name'],weights_path=old200/'assets'/w['filename'],model_library='smp',device=torch.device('cuda'),dtype=torch.float32,in_chans=3,n_out=4,compile_models=False,patch_size=1000,batch_size=1),row
def split(m):
 assert set(dict(m.named_children()))=={'encoder','decoder','segmentation_head'}
 dec=[n for n in m.state_dict() if n.startswith(('decoder.','segmentation_head.'))];train=[n for n,p in m.named_parameters() if n.startswith(('decoder.','segmentation_head.'))]
 assert len(dec)+len([n for n in m.state_dict() if n.startswith('encoder.')])==len(m.state_dict())
 for n,p in m.named_parameters():p.requires_grad_(n in train)
 return train,dec
def score(models,names,member_indices,plan,guard):
 import numpy as np,torch
 from phase119_common import catalog_image
 from phase118_common import image,coordinates
 from omnicloudmask import model_utils as mu,cloud_mask as cm
 for m in models:
  m.eval()
  for p in m.parameters():p.requires_grad_(False)
 initial=[digest(m.state_dict()) for m in models];coord=coordinates();seals={k:{} for k in names};receipts=[];support=[];S['kind']='inference';S['status']='scoring'
 expected={r['product_id']:r for r in load(ROOT/'EXPECTED_SUPPORT.json')['rows']}
 for row in plan:
  guard(True);pool,pid=row['pool'],row['product_id'];S.update(pool=pool,product_id=pid,patch_completed=0);actual=[]
  if pool=='Catalogue':
   arr,valid,_=catalog_image(row['group']);flat=np.flatnonzero(valid);yy,xx=np.unravel_index(flat,valid.shape);joint=np.ones(len(flat),bool);points=None;assert hashlib.sha256(flat.tobytes()).hexdigest()==expected[pid]['flat_indices_sha256']
  else:
   points=[q for q in coord if q['product_id']==pid];arr,v,bindings=image(row['product'],'C20',points);assert bindings==row['input_bindings'];yy=np.array([int(q['row20']) for q in points]);xx=np.array([int(q['col20']) for q in points]);joint=v[yy,xx].copy();del v
  count=len(yy);cpu=np.zeros((len(names),4,count),np.float32);grad=np.zeros(count,np.float32);latest={};meanerrors=[];gradcheck=[];oldinfer=mu.infer_batch;oldstore=mu.store_results;before=S['calls']
  def multi(ms,batch):
   with torch.no_grad():raw=[m(batch) for m in models]
   vals=[raw[i] for i in member_indices]+[torch.mean(torch.stack([raw[i],raw[1]]),0) for i in member_indices]
   npraw=[v.detach().cpu().numpy() for v in raw];arrays=[v.detach().cpu().numpy().copy() for v in vals]
   for k,i in enumerate(member_indices,len(member_indices)):
    err=float(np.max(np.abs(np.mean(np.stack([npraw[i],npraw[1]]),axis=0,dtype=np.float32)-arrays[k])));assert err<=1e-5;meanerrors.append(err)
   latest['raw']=arrays;return torch.mean(torch.stack([raw[0]]),0)
  def store(pred_batch,index_batch,pred_tracker,gradient,grad_tracker=None):
   g=gradient.detach().cpu().numpy();axis=np.minimum(np.minimum(np.arange(1,1001),np.arange(1000,0,-1)),300).astype(np.float32)*np.float32(1/300);ind=axis[:,None]*axis[None,:];assert np.max(np.abs(g-ind))<=1e-7
   for i,(t,b,l,rr) in enumerate(index_batch):
    inside=(yy>=t)&(yy<b)&(xx>=l)&(xx<rr);dy=yy[inside]-t;dx=xx[inside]-l;w=g[dy,dx]
    for j,raw in enumerate(latest['raw']):cpu[j][:,inside]+=raw[i,:,dy,dx].T*w[None]
    grad[inside]+=w;actual.append([t,b,l,rr]);S['patch_completed']=len(actual)
   latest['author_grad']=grad_tracker;return oldstore(pred_batch,index_batch,pred_tracker,gradient,grad_tracker)
  mu.infer_batch=multi;mu.store_results=store
  try:official=cm.predict_from_array(arr,patch_size=1000,patch_overlap=300,batch_size=1,inference_device='cuda',mosaic_device='cpu',inference_dtype=torch.float32,export_confidence=True,softmax_output=True,custom_models=[models[0]],model_version=4,pred_classes=4,compile_models=False,no_data_value=0,apply_no_data_mask=True)
  finally:mu.infer_batch=oldinfer;mu.store_results=oldstore
  assert actual==row['actual_patch_coordinates'] and S['calls']-before==len(models)*len(actual);assert np.array_equal(grad,latest['author_grad'].numpy()[yy,xx]);author=np.any(arr[:,yy,xx]!=0,axis=0);joint &= author&(grad>0);logits=cpu/np.maximum(grad[None,None],np.finfo(np.float32).tiny);joint &= np.isfinite(logits).all((0,1));rawpred=np.argmax(logits,axis=1).astype(np.uint8);parent=np.array([0,1,1,2],np.uint8)[rawpred];soft=np.exp(logits-logits.max(1,keepdims=True));soft/=soft.sum(1,keepdims=True);probs=np.clip(soft+np.float32(.001),.001,.999)*author[None,None];error=float(np.max(np.abs(probs[0][:,joint]-official[:,yy[joint],xx[joint]])));assert error<=1e-5
  # Recompute all policies' point postprocessing/argmax with official functions, zero network calls.
  posterrors=[]
  for j,name in enumerate(names):
   ten=torch.from_numpy(logits[j]);conf=cm.postprocess_preds(ten,True,True,class_dim=0).numpy();cl=cm.postprocess_preds(ten,False,False,class_dim=0).numpy().reshape(-1);e=float(np.max(np.abs(conf[:,joint]-probs[j][:,joint])));assert e<=1e-5 and np.array_equal(cl[joint],rawpred[j,joint]);posterrors.append(e)
   target=OUT/name/'scores'/pool;target.mkdir(parents=True,exist_ok=True)
   if pool=='Catalogue':
    path=target/(pid+'.npz');np.savez_compressed(path,scores=probs[j,3],valid=joint,flat_indices=flat,raw_class=rawpred[j],parent_class=parent[j])
   else:
    payload=[]
    for i,q in enumerate(points):v={k:q[k] for k in ('point_id','product_id','group_id','row10','col10','row20','col20')};v.update(valid=bool(joint[i]),shadow_confidence=float(probs[j,3,i]) if joint[i] else None,raw_class=int(rawpred[j,i]) if joint[i] else None,parent_class=int(parent[j,i]) if joint[i] else None);payload.append(v)
    path=target/pid/'UNLABELLED_POINT_SCORES.json';write(path,dict(policy=name,points=payload,labels_accessed=False))
   seals[name][pool+'/'+pid]=sha(path)
  if pool=='PixBox':assert [q['point_id'] for i,q in enumerate(points) if joint[i]]==expected[pid]['common_image_valid_point_ids']
  else:assert int(joint.sum())==expected[pid]['count']
  support.append(dict(pool=pool,product_id=pid,planned=count,common_valid=int(joint.sum()),loss=int((~joint).sum()),same_all_states=True));receipts.append(dict(pool=pool,product_id=pid,patches=len(actual),member_calls=S['calls']-before,official_first_member_confidence_error=error,raw_ensemble_mean_error=max(meanerrors),all_postprocessing_error=max(posterrors),coverage_gradient_exact=True));write(OUT/'SCENE_RECEIPTS.json',receipts);write(OUT/'SUPPORT_LOCK.json',dict(rows=support,before_local_reference_join=True));guard(True);print(pool,pid,S['calls'],flush=True)
  del arr,cpu,grad,latest,official,logits,soft,probs,ten,conf,rawpred,parent;gc.collect();torch.cuda.empty_cache()
 assert [digest(m.state_dict()) for m in models]==initial
 write(OUT/'SCORES_SEALED.json',dict(scores=seals,all_model_parameters_buffers_unchanged=True,before_local_reference_join=True))
OLD=ROOT.parent/'phase205_20261019'
def actual_audit():
 import numpy as np
 from safetensors.numpy import load_file
 from phase200_common import ROOT as old200
 white=load(ROOT/'SMP_WHITELIST.json');params=set(white['trainable']);buffers=set(white['decoder_state'])-params
 assert len(params)==32 and len(buffers)==30 and len({k.rsplit('.',1)[0] for k in buffers})==10
 assert all(k.startswith(('decoder.','segmentation_head.')) for k in params|buffers)
 assert all(k.endswith(('.running_mean','.running_var','.num_batches_tracked')) for k in buffers)
 original=load_file(str(next(old200/'assets'/w['filename'] for w in load(ROOT/'V4_WEIGHT_LOCK.json')['weights'] if 'regnet' in w['filename'])))
 assert len([k for k in original if k.startswith('encoder.')])==382
 rows=[]
 for seed,expected in [(12201,'655f4a3da0eb1cc198d61af91acbcc844559dc441391cd9c766cc582c02514f0'),(12301,'77ae35fc7b9023c115d3cbd36de036cb6f11581bb2bd07b33040f50cf12a992e')]:
  a=OLD/f'train_{seed}/decoder_initial.safetensors';b=OLD/f'train_{seed}/decoder_final.safetensors'
  assert sha(a)=='b29430c54233a5bcfa6583e2eaeff05fa07e293aea6ecd5371f8974ec27a65c3' and sha(b)==expected
  initial,final=load_file(str(a)),load_file(str(b));assert set(initial)==set(final)==params|buffers
  assert all(np.array_equal(initial[k],original[k]) for k in initial)
  assert all(final[k].dtype==initial[k].dtype and final[k].shape==initial[k].shape and np.isfinite(final[k]).all() for k in initial)
  rows.append(dict(seed=seed,initial_path=str(a),initial_sha256=sha(a),final_path=str(b),final_sha256=sha(b),parameter_tensors=len(params),BN_buffers=len(buffers),BN_modules=10,encoder_tensors=382,initial_equals_official=True,changed_parameters=sum(not np.array_equal(initial[k],final[k]) for k in params),changed_buffers=sum(not np.array_equal(initial[k],final[k]) for k in buffers)))
 write(ROOT/'ACTUAL_WEIGHT_ACCEPTANCE.json',dict(status='passed',rows=rows,no_fit_inputs_read=True,neural_calls_before_build=S['calls']))
 return params,buffers

def state_model(seed,state,params,buffers):
 import torch
 from safetensors.torch import load_file
 m,_=build('R');original={k:v.detach().clone() for k,v in m.state_dict().items()};adapt=load_file(str(OLD/f'train_{seed}/decoder_final.safetensors'),device='cuda');source={}
 for k in m.state_dict():
  changed=(k in params and state[0]=='1') or (k in buffers and state[1]=='1')
  source[k]='seed'+str(seed)+'_endpoint' if changed else 'official_R4'
  if changed:original[k]=adapt[k]
 m.load_state_dict(original,strict=True);m.eval()
 assert all(torch.equal(v,original[k]) for k,v in m.state_dict().items())
 write(OUT/f'STATE_{seed}_{state}.json',dict(seed=seed,state=state,tensor_sources=source,state_sha256=digest(m.state_dict()),encoder_sha256=digest(m.encoder.state_dict()),W_includes_BN_affine=True,B_only_running_statistics_and_counter=True,counter_not_used_in_eval=True))
 return m

def compare_reconstruction():
 import numpy as np
 sealed=load(OUT/'SCORES_SEALED.json')['scores'];checks=[]
 for name,index in sealed.items():
  for key,h in index.items():
   pool,pid=key.split('/',1);tail=Path(name)/'scores'/pool/((pid+'.npz') if pool=='Catalogue' else pid+'/UNLABELLED_POINT_SCORES.json');new=OUT/tail;prior=OLD/'infer'/tail;assert sha(new)==h and sha(prior)==load(OLD/'infer/SCORES_SEALED.json')['scores'][name][key]
   if pool=='Catalogue':
    with np.load(new,allow_pickle=False) as a,np.load(prior,allow_pickle=False) as b:
     assert all(np.array_equal(a[k],b[k]) for k in ('valid','flat_indices','raw_class','parent_class'));error=float(np.max(np.abs(a['scores']-b['scores'])))
   else:
    a,b=load(new)['points'],load(prior)['points'];assert len(a)==len(b);error=0.
    for x,y in zip(a,b):
     assert {k:v for k,v in x.items() if k!='shadow_confidence'}=={k:v for k,v in y.items() if k!='shadow_confidence'}
     if x['valid']:error=max(error,abs(x['shadow_confidence']-y['shadow_confidence']))
   assert error<=1e-5;checks.append(dict(strategy=name,pool=pool,product_id=pid,max_score_error=error,decisions_support_exact=True))
 assert len(checks)==12
 write(ROOT/'RECONSTRUCTION_ACCEPTANCE.json',dict(status='passed',rows=checks,original_all_group_acceptance_sha256=sha(ROOT/'BASELINE_ACCEPTANCE.json')))

def worker(stage):
 guard,thread,hooks=setup(stage)
 try:
  params,buffers=actual_audit();plan=load(ROOT/'EVALUATION_INPUT_PLAN.json')['rows']
  n,_=build('N')
  if stage=='reconstruction':
   r,_=build('R');models=[r,n]+[state_model(s,'11',params,buffers) for s in (12201,12301)];names=['R4','C4-12201','C4-12301','E4','E4-12201','E4-12301'];indices=[0,2,3]
   plan=[r for r in plan if ('_T32TML_' in r['product_id'] or '_T19FDV_' in r['product_id'])];assert len(plan)==2
  elif stage=='infer':
   assert load(ROOT/'RECONSTRUCTION_ACCEPTANCE.json')['status']=='passed'
   members=[state_model(s,t,params,buffers) for s in (12201,12301) for t in ('10','01')];models=[members[0],n,*members[1:]];indices=[0,2,3,4];members_names=[str(s)+'_'+t for s in (12201,12301) for t in ('10','01')];names=['M_'+v for v in members_names]+['E_'+v for v in members_names]
  else:raise ValueError(stage)
  score(models,names,indices,plan,guard)
  if stage=='reconstruction':compare_reconstruction()
  guard(True);S['status']='completed';write(OUT/'RUN_STATUS.json',S);(OUT/'EXIT_CODE').write_text('0\n')
 finally:
  STOP.set();thread.join(timeout=30)
  for h in hooks:h.remove()

def run():
 assert load(ROOT/'BASELINE_ACCEPTANCE.json')['status']=='passed';assert not (ROOT/'LAUNCH.json').exists();write(ROOT/'LAUNCH.json',dict(PID=os.getpid(),epoch=time.time()));state=dict(status='running',stage_seconds_total=0,completed=[])
 for stage in ('reconstruction','infer'):
  state['current']=stage;write(ROOT/'CONTROLLER_STATUS.json',state);start=time.monotonic()
  with (ROOT/(stage+'.console.log')).open('xb') as out:
   q=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'worker',stage],stdout=out,stderr=subprocess.STDOUT);state['worker_PID']=q.pid;write(ROOT/'CONTROLLER_STATUS.json',state);code=q.wait()
  state['stage_seconds_total']+=time.monotonic()-start;state['total_calls']=sum(1 for _ in (ROOT/'CALL_LEDGER.jsonl').open());write(ROOT/'CONTROLLER_STATUS.json',state);assert code==0,(stage,code);assert state['total_calls']<=8000 and state['stage_seconds_total']<=14400;state['completed'].append(stage)
 state['status']='completed_pending_controlled_delivery';write(ROOT/'CONTROLLER_STATUS.json',state);(ROOT/'EXIT_CODE').write_text('0\n')
if __name__=='__main__':
 try:
  if sys.argv[1]=='worker':worker(sys.argv[2])
  elif sys.argv[1]=='run':run()
  else:raise ValueError(sys.argv)
 except BaseException:
  target=OUT if OUT is not None else ROOT;target.mkdir(parents=True,exist_ok=True);S.update(status='failed',failure=traceback.format_exc(),seconds=time.monotonic()-START,cumulative_calls=PRIOR+S.get('calls',0),cumulative_stage_seconds=PRESECONDS+time.monotonic()-START,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
  if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():S['gpu_peak_reserved']=sys.modules['torch'].cuda.max_memory_reserved()
  write(target/'FAILURE.json',S);(target/'EXIT_CODE').write_text('1\n');print(S['failure'],file=sys.stderr,flush=True)
  if len(sys.argv)>1 and sys.argv[1]=='worker':os._exit(1)
  raise
