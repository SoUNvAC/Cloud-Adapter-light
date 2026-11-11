"""Frozen V4 SMP gates, two endpoint fits, and raw-member-reused six-policy scoring."""
import csv,datetime,gc,hashlib,inspect,json,os,resource,subprocess,sys,threading,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
REPO=Path('/home/scv/Cloud-Adapter-light');ROOT=Path('/home/scv/shared/phase205_20261019');BASE=Path('/home/scv/shared/phase111_ocm_runtime');sys.path.insert(0,str(REPO/'src/tools'))
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
def crop(group,record):
 import numpy as np,torch
 from omnicloudmask import model_utils as mu
 from phase119_common import catalog_image
 arr,valid,truth=catalog_image(group,read_truth=True);target=np.full(valid.shape,255,np.uint8);target[valid]=truth;y,x=record['top'],record['left'];a=arr[:,y:y+512,x:x+512];b=target[y:y+512,x:x+512]
 if record['horizontal_flip']:a=a[:,:,::-1];b=b[:,::-1]
 a=mu.channel_norm(a,0);assert np.isfinite(a).all() and (b!=255).any()
 return torch.from_numpy(a[None].copy()).cuda(),torch.from_numpy(b[None].astype(np.int64)).cuda()
def gate(guard):
 import numpy as np,torch,segmentation_models_pytorch as smp,timm
 from safetensors.torch import load_file
 from phase200_common import ROOT as old200
 from omnicloudmask import model_utils as mu,cloud_mask as cm
 r,row=build('R');n,nrow=build('N');S['kind']='constructor';direct=smp.Unet(encoder_name=row['timm_model_name'],encoder_weights=None,in_channels=3,classes=4).cuda().eval();w=next(w for w in load(ROOT/'V4_WEIGHT_LOCK.json')['weights'] if 'regnet' in w['filename']);direct.load_state_dict(load_file(str(old200/'assets'/w['filename']),device='cuda'),strict=True)
 assert digest(r.state_dict())==digest(direct.state_dict());train,dec=split(direct);initial={k:v.detach().clone() for k,v in direct.state_dict().items()};eh=digest(r.encoder.state_dict())
 white=dict(encoder='encoder',trainable=train,decoder_state=dec,total_parameters=sum(p.numel() for p in r.parameters()),trainable_parameters=sum(p.numel() for p in direct.parameters() if p.requires_grad),parameters=[dict(name=k,numel=p.numel(),shape=list(p.shape),trainable=k in train) for k,p in direct.named_parameters()],buffers=[dict(name=k,shape=list(v.shape),frozen=k.startswith('encoder.')) for k,v in direct.named_buffers()],BN=[dict(name=k,type=type(v).__name__,affine=v.affine,track_running_stats=v.track_running_stats) for k,v in direct.named_modules() if isinstance(v,torch.nn.modules.batchnorm._BatchNorm)])
 white['trainable_fraction']=white['trainable_parameters']/white['total_parameters'];write(OUT/'SMP_WHITELIST.json',white)
 fit=load(ROOT/'12201_FIT_INPUT_LOCK.json')['bindings'];schedule=load(ROOT/'12201_PAIRED_CROP_SCHEDULE.json')['rows'];x,y=crop(fit[schedule[0]['fit_index']],schedule[0]);S['kind']='gate'
 with torch.no_grad():a=r(x);b=direct(x)
 assert a.shape==(1,4,512,512);err=float((a-b).abs().max());assert err<=1e-5
 direct.train();direct.encoder.eval();from phase120_common import objectives
 z=direct(x);loss,_=objectives(z,y);loss.backward();assert all(p.grad is None for p in direct.encoder.parameters());assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in direct.parameters() if p.requires_grad)
 for part in (direct.decoder,direct.segmentation_head):assert any(p.grad.abs().max()>0 for p in part.parameters() if p.requires_grad)
 assert digest(direct.encoder.state_dict())==eh;direct.load_state_dict(initial,strict=True);direct.zero_grad(set_to_none=True);direct.eval();assert digest(direct.state_dict())==digest(r.state_dict())
 del direct,initial,x,y,a,b,z,loss;gc.collect();torch.cuda.empty_cache()
 # Official E4 on a fixed fit image; independent raw-mean, weighted mosaic and postprocess.
 from phase119_common import catalog_image
 arr,valid,_=catalog_image(fit[0]);cpu=np.zeros((4,1022,1022),np.float32);grad=np.zeros((1022,1022),np.float32);axis=np.minimum(np.minimum(np.arange(1,1001),np.arange(1000,0,-1)),300).astype(np.float32)*np.float32(1/300);g=axis[:,None]*axis[None,:];latest={};rawerrors=[];oldinfer=mu.infer_batch;oldstore=mu.store_results
 def capture(ms,batch):
  outputs=[];hs=[m.register_forward_hook(lambda m,a,o:outputs.append(o.detach().cpu().numpy().copy())) for m in ms]
  try:result=oldinfer(ms,batch)
  finally:
   for h in hs:h.remove()
  independent=np.mean(np.stack(outputs),axis=0,dtype=np.float32);e=float(np.max(np.abs(independent-result.detach().cpu().numpy())));assert e<=1e-5;rawerrors.append(e);latest['mean']=independent;return result
 def store(pred_batch,index_batch,pred_tracker,gradient,grad_tracker=None):
  assert np.max(np.abs(g-gradient.detach().cpu().numpy()))<=1e-7
  for i,(t,b,l,rr) in enumerate(index_batch):cpu[:,t:b,l:rr]+=latest['mean'][i]*g;grad[t:b,l:rr]+=g
  return oldstore(pred_batch,index_batch,pred_tracker,gradient,grad_tracker)
 mu.infer_batch=capture;mu.store_results=store;S['kind']='gate'
 try:out=cm.predict_from_array(arr,patch_size=1000,patch_overlap=300,batch_size=1,inference_device='cuda',mosaic_device='cpu',inference_dtype=torch.float32,export_confidence=True,softmax_output=True,custom_models=[r,n],model_version=4,pred_classes=4,compile_models=False,no_data_value=0,apply_no_data_mask=True)
 finally:mu.infer_batch=oldinfer;mu.store_results=oldstore
 logits=cpu/np.maximum(grad[None],np.finfo(np.float32).tiny);soft=np.exp(logits-logits.max(0,keepdims=True));soft/=soft.sum(0,keepdims=True);ind=np.clip(soft+np.float32(.001),.001,.999)*np.any(arr!=0,axis=0)[None];e=float(np.max(np.abs(ind-out)));assert e<=1e-5
 guard(True);write(OUT/'GATE_ACCEPTANCE.json',dict(status='passed',model_version=4,library='smp',torch=torch.__version__,smp=smp.__version__,timm=timm.__version__,R4_official_vs_direct_max_error=err,E4_official_vs_independent_max_error=e,raw_mean_max_error=max(rawerrors),classes=['Clear','ThickCloud','ThinCloud','Shadow'],decoder_head_finite_nonzero_gradients=True,encoder_no_grad_and_unchanged=True,post_gate_initial_state_restored=True,whitelist_sha256=sha(OUT/'SMP_WHITELIST.json'),schedule_SHA={str(seed):sha(ROOT/f'{seed}_PAIRED_CROP_SCHEDULE.json') for seed in (12201,12301)},no_training_started=True))
def train(seed,guard):
 import numpy as np,torch
 from safetensors.torch import save_file
 from phase119_common import catalog_image
 from phase120_common import objectives
 from omnicloudmask import model_utils as mu
 assert load(ROOT/'gate/GATE_ACCEPTANCE.json')['status']=='passed' and (ROOT/'GATE_ACCEPTED_LOCAL.json').exists()
 torch.manual_seed(seed);np.random.seed(seed);m,_=build('R');names,dec=split(m);white=load(ROOT/'gate/SMP_WHITELIST.json');assert names==white['trainable'] and dec==white['decoder_state'];eh=digest(m.encoder.state_dict())
 def save(p):save_file({k:v.detach().cpu().contiguous() for k,v in m.state_dict().items() if k in dec},str(p))
 save(OUT/'decoder_initial.safetensors');fit=load(ROOT/f'{seed}_FIT_INPUT_LOCK.json')['bindings'];sch=load(ROOT/f'{seed}_PAIRED_CROP_SCHEDULE.json')['rows'];assert len(fit)==100 and len(sch)==1000
 cache=[]
 for g in fit:
  a,v,y=catalog_image(g,read_truth=True);b=np.full(v.shape,255,np.uint8);b[v]=y;cache.append((a,b))
 torch.manual_seed(seed);opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8);m.train();m.encoder.eval();losses=[];S['kind']='training';S['status']='training'
 for step,rec in enumerate(sch,1):
  guard();S['step']=step;a,b=cache[rec['fit_index']];y,x=rec['top'],rec['left'];a=a[:,y:y+512,x:x+512];b=b[y:y+512,x:x+512]
  if rec['horizontal_flip']:a=a[:,:,::-1];b=b[:,::-1]
  a=mu.channel_norm(a,0);assert np.isfinite(a).all() and (b!=255).any();xx=torch.from_numpy(a[None].copy()).cuda();yy=torch.from_numpy(b[None].astype(np.int64)).cuda();opt.zero_grad(set_to_none=True);z=m(xx);ce,_=objectives(z,yy);assert torch.isfinite(ce);ce.backward();assert all(p.grad is None for p in m.encoder.parameters());assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad);opt.step();losses.append(dict(step=step,CE=float(ce.detach())))
  if step%25==0:print(seed,step,losses[-1],flush=True)
  del xx,yy,z,ce
 assert digest(m.encoder.state_dict())==eh;save(OUT/'decoder_final.safetensors');write(OUT/'TRAINING_RECEIPT.json',dict(seed=seed,steps=1000,losses=losses,encoder_unchanged=True,initial_sha256=sha(OUT/'decoder_initial.safetensors'),final_sha256=sha(OUT/'decoder_final.safetensors'),schedule_sha256=sha(ROOT/f'{seed}_PAIRED_CROP_SCHEDULE.json'),fit_sha256=sha(ROOT/f'{seed}_FIT_INPUT_LOCK.json'),selection='fixed endpoint; no evaluation access',trainable_names=names));guard(True)
def infer(guard):
 import numpy as np,torch
 from safetensors.torch import load_file
 from phase119_common import catalog_image
 from phase118_common import image,coordinates
 from omnicloudmask import model_utils as mu,cloud_mask as cm
 sealed=load(ROOT/'TRAINING_SEAL.json')
 for seed in (12201,12301):assert sha(ROOT/f'train_{seed}/decoder_final.safetensors')==sealed[str(seed)]
 r,_=build('R');n,_=build('N');models=[r,n]
 for seed in (12201,12301):
  m,_=build('R');m.load_state_dict(load_file(str(ROOT/f'train_{seed}/decoder_final.safetensors'),device='cuda'),strict=False);models.append(m)
 for m in models:
  m.eval()
  for p in m.parameters():p.requires_grad_(False)
 initial=[digest(m.state_dict()) for m in models];plan=load(ROOT/'EVALUATION_INPUT_PLAN.json')['rows'];coord=coordinates();names=['R4','C4-12201','C4-12301','E4','E4-12201','E4-12301'];seals={k:{} for k in names};receipts=[];support=[];S['kind']='inference';S['status']='scoring'
 expected={r['product_id']:r for r in load(ROOT/'EXPECTED_SUPPORT.json')['rows']}
 for row in plan:
  guard(True);pool,pid=row['pool'],row['product_id'];S.update(pool=pool,product_id=pid,patch_completed=0);actual=[]
  if pool=='Catalogue':
   arr,valid,_=catalog_image(row['group']);flat=np.flatnonzero(valid);yy,xx=np.unravel_index(flat,valid.shape);joint=np.ones(len(flat),bool);points=None;assert hashlib.sha256(flat.tobytes()).hexdigest()==expected[pid]['flat_indices_sha256']
  else:
   points=[q for q in coord if q['product_id']==pid];arr,v,bindings=image(row['product'],'C20',points);assert bindings==row['input_bindings'];yy=np.array([int(q['row20']) for q in points]);xx=np.array([int(q['col20']) for q in points]);joint=v[yy,xx].copy();del v
  count=len(yy);cpu=np.zeros((6,4,count),np.float32);grad=np.zeros(count,np.float32);latest={};meanerrors=[];gradcheck=[];oldinfer=mu.infer_batch;oldstore=mu.store_results;before=S['calls']
  def multi(ms,batch):
   with torch.no_grad():raw=[m(batch) for m in models]
   vals=[raw[0],raw[2],raw[3]]+[torch.mean(torch.stack([raw[i],raw[1]]),0) for i in (0,2,3)]
   npraw=[v.detach().cpu().numpy() for v in raw];arrays=[v.detach().cpu().numpy().copy() for v in vals]
   for k,i in enumerate((0,2,3),3):
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
  try:official=cm.predict_from_array(arr,patch_size=1000,patch_overlap=300,batch_size=1,inference_device='cuda',mosaic_device='cpu',inference_dtype=torch.float32,export_confidence=True,softmax_output=True,custom_models=[r],model_version=4,pred_classes=4,compile_models=False,no_data_value=0,apply_no_data_mask=True)
  finally:mu.infer_batch=oldinfer;mu.store_results=oldstore
  assert actual==row['actual_patch_coordinates'] and S['calls']-before==4*len(actual);assert np.array_equal(grad,latest['author_grad'].numpy()[yy,xx]);author=np.any(arr[:,yy,xx]!=0,axis=0);joint &= author&(grad>0);logits=cpu/np.maximum(grad[None,None],np.finfo(np.float32).tiny);joint &= np.isfinite(logits).all((0,1));rawpred=np.argmax(logits,axis=1).astype(np.uint8);parent=np.array([0,1,1,2],np.uint8)[rawpred];soft=np.exp(logits-logits.max(1,keepdims=True));soft/=soft.sum(1,keepdims=True);probs=np.clip(soft+np.float32(.001),.001,.999)*author[None,None];error=float(np.max(np.abs(probs[0][:,joint]-official[:,yy[joint],xx[joint]])));assert error<=1e-5
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
  support.append(dict(pool=pool,product_id=pid,planned=count,common_valid=int(joint.sum()),loss=int((~joint).sum()),same_all_six=True));receipts.append(dict(pool=pool,product_id=pid,patches=len(actual),member_calls=S['calls']-before,official_R4_confidence_error=error,raw_ensemble_mean_error=max(meanerrors),all_postprocessing_error=max(posterrors),coverage_gradient_exact=True));write(OUT/'SCENE_RECEIPTS.json',receipts);write(OUT/'SUPPORT_LOCK.json',dict(rows=support,before_local_reference_join=True));guard(True);print(pool,pid,S['calls'],flush=True)
  del arr,cpu,grad,latest,official,logits,soft,probs,ten,conf,rawpred,parent;gc.collect();torch.cuda.empty_cache()
 assert [digest(m.state_dict()) for m in models]==initial
 write(OUT/'SCORES_SEALED.json',dict(scores=seals,all_model_parameters_buffers_unchanged=True,before_local_reference_join=True))
def worker(stage):
 guard,thread,hooks=setup(stage)
 try:
  if stage=='gate':gate(guard)
  elif stage.startswith('train_'):train(int(stage.split('_')[1]),guard)
  elif stage=='infer':infer(guard)
  else:raise ValueError(stage)
  guard(True);S['status']='completed';write(OUT/'RUN_STATUS.json',S);(OUT/'EXIT_CODE').write_text('0\n')
 finally:
  STOP.set();thread.join(timeout=30)
  for h in hooks:h.remove()
def pipeline():
 assert (ROOT/'GATE_ACCEPTED_LOCAL.json').exists() and load(ROOT/'gate/GATE_ACCEPTANCE.json')['status']=='passed';assert not (ROOT/'LAUNCH.json').exists();write(ROOT/'LAUNCH.json',dict(PID=os.getpid(),epoch=time.time()));g=load(ROOT/'gate/RUN_STATUS.json');state=dict(status='running',stage_seconds_total=g['cumulative_stage_seconds'],completed=['gate']);write(ROOT/'CONTROLLER_STATUS.json',state)
 for stage in ('train_12201','train_12301','infer'):
  if stage=='infer':write(ROOT/'TRAINING_SEAL.json',{str(s):sha(ROOT/f'train_{s}/decoder_final.safetensors') for s in (12201,12301)})
  state['current']=stage;write(ROOT/'CONTROLLER_STATUS.json',state);start=time.monotonic()
  with (ROOT/(stage+'.console.log')).open('xb') as out:
   q=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'worker',stage],stdout=out,stderr=subprocess.STDOUT);state['worker_PID']=q.pid;write(ROOT/'CONTROLLER_STATUS.json',state);code=q.wait()
  state['stage_seconds_total']+=time.monotonic()-start;state['total_calls']=sum(1 for _ in (ROOT/'CALL_LEDGER.jsonl').open());write(ROOT/'CONTROLLER_STATUS.json',state);assert code==0,(stage,code);state['completed'].append(stage);assert state['total_calls']<=8000 and state['stage_seconds_total']<=14400
 state['status']='completed_pending_controlled_delivery_local_reference_join';write(ROOT/'CONTROLLER_STATUS.json',state);(ROOT/'EXIT_CODE').write_text('0\n')
if __name__=='__main__':
 try:
  if sys.argv[1]=='worker':worker(sys.argv[2])
  elif sys.argv[1]=='run':pipeline()
  else:raise ValueError(sys.argv)
 except BaseException:
  target=OUT if OUT is not None else ROOT;target.mkdir(parents=True,exist_ok=True);S.update(status='failed',failure=traceback.format_exc(),seconds=time.monotonic()-START,cumulative_calls=PRIOR+S.get('calls',0),cumulative_stage_seconds=PRESECONDS+time.monotonic()-START,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
  if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():S['gpu_peak_reserved']=sys.modules['torch'].cuda.max_memory_reserved()
  write(target/'FAILURE.json',S);(target/'EXIT_CODE').write_text('1\n');print(S['failure'],file=sys.stderr,flush=True)
  if len(sys.argv)>1 and sys.argv[1]=='worker':os._exit(1)
  raise
