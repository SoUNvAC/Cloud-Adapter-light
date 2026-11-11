"""Frozen V4 SMP gates, two endpoint fits, and raw-member-reused six-policy scoring."""
import csv,datetime,gc,hashlib,inspect,json,os,resource,subprocess,sys,threading,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
REPO=Path('/home/scv/Cloud-Adapter-light');ROOT=Path('/home/scv/shared/phase208_20261020');BASE=Path('/home/scv/shared/phase111_ocm_runtime');sys.path.insert(0,str(REPO/'src/tools'));sys.path.insert(0,str(ROOT/'code'))
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
PENDING=[];LEDGER_LOCK=threading.Lock()
S={};START=time.monotonic();STOP=threading.Event();OUT=None;PRIOR=0;PRESECONDS=0
def flush_ledger():
 with LEDGER_LOCK:
  if PENDING:
   with (ROOT/'CALL_LEDGER.jsonl').open('a') as f:
    for record in PENDING:f.write(json.dumps(record)+'\n')
   PENDING.clear()

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
  if S.get('timing_active',False):return
  flush_ledger()
  S.update(stage_seconds=time.monotonic()-START,cumulative_stage_seconds=PRESECONDS+time.monotonic()-START,cumulative_calls=PRIOR+S['calls'],rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,gpu_peak_reserved=max(S.get('gpu_peak_reserved',0),torch.cuda.max_memory_reserved()))
  assert S['cumulative_calls']<=5000 and S['cumulative_stage_seconds']<=14400
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
   assert PRIOR+S['calls']<5000;S['calls']+=1;kind=S['kind'];S[kind+'_calls']=S.get(kind+'_calls',0)+1
   with LEDGER_LOCK:PENDING.append(dict(global_call=PRIOR+S['calls'],stage=stage,kind=kind,step=S['step'],epoch=time.time()))
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

def modules():
 from phase208_modules import paired_modules
 return paired_modules(REPO/'src/cloud_adapter/models/backbones/target_msre_dinov2.py')

def make_candidate(arm):
 import torch
 from phase207_msre_interface import FrozenR4WithParallelMsRE
 r,_=build('R');m,b,_=modules();adapter=(m if arm=='M' else b).cuda();candidate=FrozenR4WithParallelMsRE(r,adapter);candidate.eval();return candidate

def parent_loss(logits,target):
 import torch
 return torch.nn.functional.cross_entropy(torch.stack((logits[:,0],torch.logsumexp(logits[:,1:3],1),logits[:,3]),1),target,ignore_index=255)

def fixed_crop():
 import torch,numpy as np
 from phase119_common import catalog_image
 from omnicloudmask import model_utils as mu
 cfg=load(ROOT/'GATE_INPUT.json');arr,v,truth=catalog_image(cfg['group'],read_truth=True);full=np.full(v.shape,255,np.uint8);full[v]=truth;y,x=cfg['top'],cfg['left'];a=mu.channel_norm(arr[:,y:y+512,x:x+512].copy(),0);target=full[y:y+512,x:x+512];assert np.isfinite(a).all() and (target!=255).any()
 return torch.from_numpy(a[None].copy()).cuda(),torch.from_numpy(target[None].astype(np.int64)).cuda(),dict(product=cfg['group']['product'],group_id=cfg['group']['group_id'],top=y,left=x,input_sha256=hashlib.sha256(a.tobytes()).hexdigest(),target_sha256=hashlib.sha256(target.tobytes()).hexdigest())

def prepare_benchmark():
 import numpy as np
 from phase118_common import image,coordinates
 from omnicloudmask import model_utils as mu
 coord=coordinates();locks=[];(ROOT/'benchmark_inputs').mkdir(exist_ok=False)
 for row in load(ROOT/'BENCHMARK_PLAN.json')['rows']:
  pid=row['product_id'];points=[q for q in coord if q['product_id']==pid];arr,v,bindings=image(row['product'],'C20',points);assert bindings==row['input_bindings'];chosen=None
  for i,(t,b,l,r) in enumerate(row['actual_patch_coordinates']):
   if v[t:b,l:r].any():chosen=(i,t,b,l,r);break
  assert chosen is not None;index,t,b,l,r=chosen;a=mu.channel_norm(arr[:,t:b,l:r].copy(),0);assert a.shape==(3,1000,1000) and np.isfinite(a).all();path=ROOT/'benchmark_inputs'/(pid+'.npy');np.save(path,a,allow_pickle=False);locks.append(dict(product_id=pid,window_index=index,coordinates=[t,b,l,r],shape=list(a.shape),array_sha256=hashlib.sha256(a.tobytes()).hexdigest(),file_sha256=sha(path),valid_pixels=int(v[t:b,l:r].sum()),first_image_valid_window=True,label_read=False));del arr,v,a
 write(ROOT/'BENCHMARK_INPUT_LOCK.json',dict(rows=locks,before_any_training=True,pre_normalized=True))

def gate(guard):
 import torch,numpy as np
 from safetensors.torch import save_file
 prepare_benchmark();r,_=build('R');r.eval()
 for p in r.parameters():p.requires_grad_(False)
 original_digest=digest(r.state_dict());x,y,input_record=fixed_crop();S['kind']='gate_R4';S['status']='gate';guard()
 with torch.no_grad():original=r(x)
 M,B,shared=modules();assert len(shared)==18;write(OUT/'COMMON_INITIALIZATION.json',dict(seed=20801,copy_order=load(ROOT/'INITIALIZATION_RULE.json')['order'],shared_tensors=shared,exact_initial_values_equal=True,all_cross_arm_storage_independent=True,M_new_parameters=970369,B_new_parameters=944641,parameter_difference=25728))
 rows=[]
 for arm in ('M','B'):
  m=make_candidate(arm);adapter=m.adapter;expected=M if arm=='M' else B;assert digest(adapter.state_dict())==digest(expected.state_dict());del expected
  initial={k:v.detach().clone() for k,v in adapter.state_dict().items()};before=digest(m.base.state_dict());assert before==original_digest;assert not list(adapter.named_buffers());assert sum(p.numel() for p in adapter.parameters())==(970369 if arm=='M' else 944641)
  save_file({k:v.detach().cpu().contiguous() for k,v in adapter.state_dict().items()},str(OUT/f'{arm}_initial.safetensors'));adapter.enabled=False;S['kind']='gate_'+arm+'_disabled'
  with torch.no_grad():disabled=m(x)
  closed=float((disabled-original).abs().max());assert closed<=1e-5;adapter.enabled=True;S['kind']='gate_'+arm+'_enabled_backward';enabled=m(x);opened=float((enabled.detach()-original).abs().max());loss=parent_loss(enabled,y);loss.backward();gr=[]
  for name,p in adapter.named_parameters():
   g=p.grad;gr.append(dict(name=name,shape=list(p.shape),numel=p.numel(),has_gradient=g is not None,finite=bool(torch.isfinite(g).all()) if g is not None else None,nonzero=bool(torch.count_nonzero(g)) if g is not None else None,L2=float(g.double().norm()) if g is not None else None))
  assert any(q['nonzero'] for q in gr) and all(q['finite'] for q in gr if q['has_gradient']);assert all(p.grad is None and not p.requires_grad for p in m.base.parameters());assert all(not q.training for q in m.base.modules());assert digest(m.base.state_dict())==before
  adapter.load_state_dict(initial,strict=True);adapter.zero_grad(set_to_none=True);assert all(p.grad is None for p in adapter.parameters()) and digest(adapter.state_dict())==digest(initial)
  rows.append(dict(arm=arm,closed_max_error=closed,enabled_initial_max_error=opened,gradient_rows=gr,gate_loss=float(loss.detach()),initial_sha256=sha(OUT/f'{arm}_initial.safetensors'),initial_state_digest=digest(initial),initial_restored=True,gradients_cleared=True,original_parameters_buffers_unchanged=True,original_whitelist=[dict(name=k,shape=list(p.shape),requires_grad=p.requires_grad) for k,p in m.base.named_parameters()],original_buffer_names=list(dict(m.base.named_buffers()))))
  m.handle.remove();del m,adapter,initial,enabled,disabled,loss;gc.collect();torch.cuda.empty_cache();guard(True)
 write(OUT/'GATE_ACCEPTANCE.json',dict(status='passed',rows=rows,input=input_record,seed=20801,optimizer_updates=0,backward_per_arm=1,shared_count=len(shared),initialization_order_sha256=sha(ROOT/'INITIALIZATION_RULE.json'),original_R4_digest=original_digest,benchmark_lock_sha256=sha(ROOT/'BENCHMARK_INPUT_LOCK.json')))

def train(arm,guard):
 import numpy as np,torch
 from safetensors.torch import load_file,save_file
 from phase119_common import catalog_image
 from omnicloudmask import model_utils as mu
 accepted=load(ROOT/'gate/GATE_ACCEPTANCE.json');assert accepted['status']=='passed';m=make_candidate(arm);initial_file=ROOT/'gate'/f'{arm}_initial.safetensors';m.adapter.load_state_dict(load_file(str(initial_file),device='cuda'),strict=True);m.adapter.zero_grad(set_to_none=True);assert digest(m.adapter.state_dict())==next(r['initial_state_digest'] for r in accepted['rows'] if r['arm']==arm);original=digest(m.base.state_dict());buffers=digest(dict(m.base.named_buffers()));fit=load(ROOT/'12201_FIT_INPUT_LOCK.json')['bindings'];schedule=load(ROOT/'12201_PAIRED_CROP_SCHEDULE.json')['rows'];assert len(fit)==100 and len(schedule)==1000;assert sha(ROOT/'12201_PAIRED_CROP_SCHEDULE.json')=='3baae5c5e13c3a3dbc515410b537d2071db87e5485953ef87ef5a783b10c0aba'
 cache=[]
 for group in fit:
  a,v,y=catalog_image(group,read_truth=True);full=np.full(v.shape,255,np.uint8);full[v]=y;cache.append((a,full))
 torch.manual_seed(20801);np.random.seed(20801);opt=torch.optim.AdamW(list(m.adapter.parameters()),lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8);assert {id(p) for q in opt.param_groups for p in q['params']}=={id(p) for p in m.adapter.parameters()};m.train();assert all(not q.training for q in m.base.modules());losses=[];S['kind']='training_'+arm;S['status']='training'
 for step,rec in enumerate(schedule,1):
  S['step']=step;guard();a,y=cache[rec['fit_index']];t,l=rec['top'],rec['left'];a=a[:,t:t+512,l:l+512];y=y[t:t+512,l:l+512]
  if rec['horizontal_flip']:a=a[:,:,::-1];y=y[:,::-1]
  a=mu.channel_norm(a,0);xx=torch.from_numpy(a[None].copy()).cuda();yy=torch.from_numpy(y[None].astype(np.int64)).cuda();opt.zero_grad(set_to_none=True);z=m(xx);loss=parent_loss(z,yy);assert torch.isfinite(loss);loss.backward();assert all(p.grad is None for p in m.base.parameters());assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.adapter.parameters());opt.step();losses.append(dict(step=step,CE=float(loss.detach())))
  if step%25==0:print(arm,step,losses[-1],flush=True);assert not m.base.training;guard(True)
  del xx,yy,z,loss
 assert digest(m.base.state_dict())==original and digest(dict(m.base.named_buffers()))==buffers;m.eval();save_file({k:v.detach().cpu().contiguous() for k,v in m.adapter.state_dict().items()},str(OUT/'module_final.safetensors'))
 write(OUT/'TRAINING_RECEIPT.json',dict(arm=arm,steps=1000,optimizer_steps=1000,module_seed=20801,losses=losses,initial_path=str(initial_file),initial_sha256=sha(initial_file),final_sha256=sha(OUT/'module_final.safetensors'),original_parameters_buffers_unchanged=True,original_all_eval=True,schedule_sha256=sha(ROOT/'12201_PAIRED_CROP_SCHEDULE.json'),fit_sha256=sha(ROOT/'12201_FIT_INPUT_LOCK.json'),trainable_names=[k for k,p in m.adapter.named_parameters()],optimizer='AdamW1e-4 wd.01 betas.9,.999 eps1e-8 constant',selection='1000endpoint; noevalaccess; nogateupdates',no_calendar_cutoff_in_current_protocol=True))

def benchmark(guard):
 import torch,numpy as np
 from safetensors.torch import load_file
 r,_=build('R');n,_=build('N');candidates={a:make_candidate(a) for a in ('M','B')}
 for a,m in candidates.items():m.adapter.load_state_dict(load_file(str(ROOT/f'train_{a}/module_final.safetensors'),device='cuda'),strict=True);m.eval()
 models=[r,n,*candidates.values()]
 for m in models:
  m.eval()
  for p in m.parameters():p.requires_grad_(False)
 before=[digest(m.state_dict()) for m in models];receipts=[];lock=load(ROOT/'BENCHMARK_INPUT_LOCK.json');S['status']='benchmarking'
 for index,row in enumerate(lock['rows']):
  S['product_id']=row['product_id'];path=ROOT/'benchmark_inputs'/(row['product_id']+'.npy');assert sha(path)==row['file_sha256'];a=np.load(path,allow_pickle=False);assert hashlib.sha256(a.tobytes()).hexdigest()==row['array_sha256'];x=torch.from_numpy(a[None].copy()).cuda();order=['R4','E4','M','B'];order=order[index:]+order[:index];records=[]
  def forward(strategy):
   if strategy=='R4':return r(x)
   if strategy=='E4':return torch.mean(torch.stack([r(x),n(x)]),dim=0)
   return candidates[strategy](x)
  for strategy in order:
   S['kind']='benchmark_'+strategy+'_warmup'
   with torch.no_grad():warm=forward(strategy)
   torch.cuda.synchronize();del warm;guard()
   for repeat in range(2):
    S['kind']='benchmark_'+strategy+'_timed';S['timing_active']=True;start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin=S['calls'];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start.record()
    with torch.no_grad():result=forward(strategy)
    end.record();torch.cuda.synchronize();ms=float(start.elapsed_time(end));S['timing_active']=False;peak=torch.cuda.max_memory_reserved();records.append(dict(strategy=strategy,repeat=repeat,GPU_milliseconds=ms,network_calls=S['calls']-begin,process_reserved_peak_bytes=peak,shared_process_peak_not_independent_deployment=True));del result;guard(True)
  receipts.append(dict(product_id=row['product_id'],coordinates=row['coordinates'],input_sha256=row['array_sha256'],order=order,warmup_per_strategy=1,timed_repeats=2,rows=records));write(OUT/'BENCHMARK_TIMINGS.json',dict(rows=receipts,scope='sameprenormalizedwindowpurenetworkCUDAevents; notsceneIOormosaic',ensemble_includes_two_members_and_raw_mean=True))
 assert [digest(m.state_dict()) for m in models]==before
 write(OUT/'BENCHMARK_ACCEPTANCE.json',dict(status='passed',model_states_unchanged=True,all_repeats_retained=True,local_peak_reset_only_for_benchmark='perwindowstrategy; earlierinfer/trainpeaksinownstage',no_whole_scene_speed=True))

def infer(guard):
 import numpy as np,torch
 from phase119_common import catalog_image
 from phase118_common import image,coordinates
 from omnicloudmask import model_utils as mu,cloud_mask as cm
 from safetensors.torch import load_file
 seal=load(ROOT/'TRAINING_SEAL.json');models=[]
 for arm in ('M','B'):
  m=make_candidate(arm);path=ROOT/f'train_{arm}/module_final.safetensors';assert sha(path)==seal[arm];m.adapter.load_state_dict(load_file(str(path),device='cuda'),strict=True);m.eval()
  for p in m.parameters():p.requires_grad_(False)
  models.append(m)
 initial=[digest(m.state_dict()) for m in models];plan=load(ROOT/'EVALUATION_INPUT_PLAN.json')['rows'];coord=coordinates();names=['M','B'];seals={k:{} for k in names};receipts=[];support=[];S['kind']='inference';S['status']='scoring'
 expected={r['product_id']:r for r in load(ROOT/'EXPECTED_SUPPORT.json')['rows']}
 for row in plan:
  guard(True);pool,pid=row['pool'],row['product_id'];S.update(pool=pool,product_id=pid,patch_completed=0);actual=[]
  if pool=='Catalogue':
   arr,valid,_=catalog_image(row['group']);flat=np.flatnonzero(valid);yy,xx=np.unravel_index(flat,valid.shape);joint=np.ones(len(flat),bool);points=None;assert hashlib.sha256(flat.tobytes()).hexdigest()==expected[pid]['flat_indices_sha256']
  else:
   points=[q for q in coord if q['product_id']==pid];arr,v,bindings=image(row['product'],'C20',points);assert bindings==row['input_bindings'];yy=np.array([int(q['row20']) for q in points]);xx=np.array([int(q['col20']) for q in points]);joint=v[yy,xx].copy();del v
  count=len(yy);cpu=np.zeros((2,4,count),np.float32);grad=np.zeros(count,np.float32);latest={};meanerrors=[];gradcheck=[];oldinfer=mu.infer_batch;oldstore=mu.store_results;before=S['calls']
  def multi(ms,batch):
   with torch.no_grad():raw=[m(batch) for m in models]
   vals=raw
   npraw=[v.detach().cpu().numpy() for v in raw];arrays=[v.detach().cpu().numpy().copy() for v in vals]
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
  assert actual==row['actual_patch_coordinates'] and S['calls']-before==2*len(actual);assert np.array_equal(grad,latest['author_grad'].numpy()[yy,xx]);author=np.any(arr[:,yy,xx]!=0,axis=0);joint &= author&(grad>0);logits=cpu/np.maximum(grad[None,None],np.finfo(np.float32).tiny);joint &= np.isfinite(logits).all((0,1));rawpred=np.argmax(logits,axis=1).astype(np.uint8);parent=np.array([0,1,1,2],np.uint8)[rawpred];soft=np.exp(logits-logits.max(1,keepdims=True));soft/=soft.sum(1,keepdims=True);probs=np.clip(soft+np.float32(.001),.001,.999)*author[None,None];error=float(np.max(np.abs(probs[0][:,joint]-official[:,yy[joint],xx[joint]])));assert error<=1e-5
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
  support.append(dict(pool=pool,product_id=pid,planned=count,common_valid=int(joint.sum()),loss=int((~joint).sum()),same_all_two=True));receipts.append(dict(pool=pool,product_id=pid,patches=len(actual),member_calls=S['calls']-before,official_M_confidence_error=error,raw_ensemble_not_applicable=True,all_postprocessing_error=max(posterrors),coverage_gradient_exact=True));write(OUT/'SCENE_RECEIPTS.json',receipts);write(OUT/'SUPPORT_LOCK.json',dict(rows=support,before_local_reference_join=True));guard(True);print(pool,pid,S['calls'],flush=True)
  del arr,cpu,grad,latest,official,logits,soft,probs,ten,conf,rawpred,parent;gc.collect();torch.cuda.empty_cache()
 assert [digest(m.state_dict()) for m in models]==initial
 write(OUT/'SCORES_SEALED.json',dict(scores=seals,all_model_parameters_buffers_unchanged=True,before_local_reference_join=True))

def worker(stage):
 guard,thread,hooks=setup(stage)
 try:
  if stage=='gate':gate(guard)
  elif stage in ('train_M','train_B'):train(stage[-1],guard)
  elif stage=='infer':infer(guard)
  elif stage=='benchmark':benchmark(guard)
  else:raise ValueError(stage)
  guard(True);S['status']='completed';write(OUT/'RUN_STATUS.json',S);(OUT/'EXIT_CODE').write_text('0\n')
 finally:
  STOP.set();thread.join(timeout=30)
  for h in hooks:h.remove()
def pipeline():
 assert not (ROOT/'LAUNCH.json').exists();write(ROOT/'LAUNCH.json',dict(PID=os.getpid(),epoch=time.time()));state=dict(status='running',stage_seconds_total=0,completed=[])
 for stage in ('gate','train_M','train_B','infer','benchmark'):
  if stage.startswith('train'):assert load(ROOT/'gate/GATE_ACCEPTANCE.json')['status']=='passed' and (ROOT/'BENCHMARK_INPUT_LOCK.json').exists()
  if stage=='infer':write(ROOT/'TRAINING_SEAL.json',{a:sha(ROOT/f'train_{a}/module_final.safetensors') for a in ('M','B')})
  state['current']=stage;write(ROOT/'CONTROLLER_STATUS.json',state);start=time.monotonic()
  with (ROOT/(stage+'.console.log')).open('xb') as out:
   q=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'worker',stage],stdout=out,stderr=subprocess.STDOUT);state['worker_PID']=q.pid;write(ROOT/'CONTROLLER_STATUS.json',state);code=q.wait()
  state['stage_seconds_total']+=time.monotonic()-start;state['total_calls']=sum(1 for _ in (ROOT/'CALL_LEDGER.jsonl').open()) if (ROOT/'CALL_LEDGER.jsonl').exists() else 0;write(ROOT/'CONTROLLER_STATUS.json',state);assert code==0,(stage,code);state['completed'].append(stage);assert state['total_calls']<=5000 and state['stage_seconds_total']<=14400
 state['status']='completed_pending_controlled_delivery';write(ROOT/'CONTROLLER_STATUS.json',state);(ROOT/'EXIT_CODE').write_text('0\n')
if __name__=='__main__':
 try:
  if sys.argv[1]=='worker':worker(sys.argv[2])
  elif sys.argv[1]=='run':pipeline()
  else:raise ValueError(sys.argv)
 except BaseException:
  S['timing_active']=False;flush_ledger()
  target=OUT if OUT is not None else ROOT;target.mkdir(parents=True,exist_ok=True);S.update(status='failed',failure=traceback.format_exc(),seconds=time.monotonic()-START,cumulative_calls=PRIOR+S.get('calls',0),cumulative_stage_seconds=PRESECONDS+time.monotonic()-START,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
  if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():S['gpu_peak_reserved']=sys.modules['torch'].cuda.max_memory_reserved()
  write(target/'FAILURE.json',S);(target/'EXIT_CODE').write_text('1\n');print(S['failure'],file=sys.stderr,flush=True)
  if len(sys.argv)>1 and sys.argv[1]=='worker':os._exit(1)
  raise
