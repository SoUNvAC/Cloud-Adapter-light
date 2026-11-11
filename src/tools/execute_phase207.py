"""Phase207 bounded 3-forward/one-backward implementation audit; zero optimizer steps."""
import ast,csv,hashlib,inspect,json,os,resource,sys,threading,time,traceback
from pathlib import Path
ROOT=Path('/home/scv/shared/phase207_20261020');REPO=Path('/home/scv/Cloud-Adapter-light');BASE=Path('/home/scv/shared/phase111_ocm_runtime');sys.dont_write_bytecode=True;sys.path.insert(0,str(REPO/'src/tools'));sys.path.insert(0,str(ROOT/'code'))
PRIOR_SECONDS=load_prior=0.0
if (ROOT/'RECOVERY_LOCK.json').exists():PRIOR_SECONDS=json.loads((ROOT/'RECOVERY_LOCK.json').read_text())['prior_stage_seconds']
START=time.monotonic();S=dict(PID=os.getpid(),status='initializing',calls=0,kind='constructor');STOP=threading.Event()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(n,v):
 p=ROOT/n;t=p.with_name(p.name+'.writing.'+str(threading.get_ident()));t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf8');t.replace(p)
def digest(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(list(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def main():
 assert not (ROOT/'LAUNCH.json').exists();write('LAUNCH.json',dict(PID=os.getpid(),epoch=time.time()));cfg=load(ROOT/'PRE_EXECUTION_LOCK.json')
 for n,h in cfg['source_sha256'].items():assert sha(REPO/n)==h,n
 for n,h in load(ROOT/'CODE_LOCK.json')['files'].items():assert sha(ROOT/'code'/n)==h
 from phase200_common import setup,ROOT as old200
 setup();import torch,numpy as np,segmentation_models_pytorch as smp,timm
 from omnicloudmask import model_utils as mu
 from install_phase111_runtime import host_free
 torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats();assert torch.__version__=='2.11.0+cu128' and smp.__version__=='0.5.0' and timm.__version__=='1.0.30'
 def guard():
  S.update(stage_seconds=PRIOR_SECONDS+time.monotonic()-START,CPU_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved());assert S['calls']<=24 and S['stage_seconds']<=900 and S['CPU_peak_bytes']<=16*1024**3 and S['GPU_peak_reserved_bytes']<=20*1024**3
  assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<=1000000000
  free=host_free();S['host_min_free_bytes']=min(S.get('host_min_free_bytes',free),free);assert free>=4000000000;write('RUN_STATUS.json',S)
 guard();depth=[0]
 def before(m,args):
  if depth[0]==0:
   assert S['calls']<24;S['calls']+=1
   with (ROOT/'CALL_LEDGER.jsonl').open('a') as f:f.write(json.dumps(dict(call=S['calls'],kind=S['kind'],epoch=time.time()))+'\n')
  depth[0]+=1
 def after(m,args,out):depth[0]-=1
 hooks=[torch.nn.modules.module.register_module_forward_pre_hook(before),torch.nn.modules.module.register_module_forward_hook(after,always_call=True)]
 def watch():
  try:
   while not STOP.wait(15):guard()
  except BaseException:
   write('RESOURCE_FAILURE.json',dict(failure=traceback.format_exc(),state=S));(ROOT/'EXIT_CODE').write_text('1\n');os._exit(1)
 thread=threading.Thread(target=watch,daemon=True);thread.start()
 try:
  weights=load(ROOT/'V4_WEIGHT_LOCK.json');models=[];identities=[];index=list(csv.DictReader((BASE/'assets/vendor/omnicloudmask/model_download_links.csv').open()))
  for isR in (True,False):
   w=next(w for w in weights['weights'] if ('regnet' in w['filename'])==isR);p=old200/'assets'/w['filename'];assert sha(p)==w['sha256'];row=next(r for r in index if r['version']=='4' and r['file_name']==w['filename']);assert row['model_library']=='smp';mu.load_model_from_weights.cache_clear();m=mu.load_model_from_weights(model_name=row['timm_model_name'],weights_path=p,model_library='smp',device=torch.device('cuda'),dtype=torch.float32,in_chans=3,n_out=4,compile_models=False,patch_size=512,batch_size=1);m.eval()
   for q in m.parameters():q.requires_grad_(False)
   models.append(m);identities.append(dict(member='R4' if isR else 'N4',path=str(p),sha256=sha(p),state_sha256=digest(m.state_dict()),parameters=sum(v.numel() for v in m.parameters()),buffers=len(list(m.named_buffers()))))
  r,n=models;frozen=[digest(m.state_dict()) for m in models];buffer_before=[digest(dict(m.named_buffers())) for m in models];assert r.classification_head is None
  from phase119_common import catalog_image
  arr,valid,truth=catalog_image(cfg['group'],read_truth=True);a=arr[:,:512,:512].copy();full_target=np.full(valid.shape,255,np.uint8);full_target[valid]=truth;target=full_target[:512,:512].copy();vv=valid[:512,:512];assert vv.any();a=mu.channel_norm(a,0);assert np.isfinite(a).all()
  write('INPUT_RECEIPT.json',dict(group_id=cfg['group']['group_id'],product=cfg['group']['product'],shape=list(a.shape),top=0,left=0,normalized_input_sha256=hashlib.sha256(a.tobytes()).hexdigest(),target_sha256=hashlib.sha256(target.tobytes()).hexdigest(),valid_pixels=int(vv.sum()),prelock_sha256=sha(ROOT/'PRE_EXECUTION_LOCK.json'),selection_not_based_on_target_classes=True,development_or_PixBox_reference_read=False))
  x=torch.from_numpy(a[None].copy()).cuda();y=torch.from_numpy(target[None].astype(np.int64)).cuda();features=[];capt=r.encoder.register_forward_hook(lambda m,args,out:features.extend([dict(index=i,shape=list(v.shape),channels=int(v.shape[1])) for i,v in enumerate(out)]));S['kind']='original_R4';S['status']='original_forward';guard()
  with torch.no_grad():original=r(x)
  capt.remove();assert [v['channels'] for v in features[-4:]]==cfg['expected_stage_channels'];write('ARCHITECTURE.json',dict(encoder_features=features,decoder_forward_source=inspect.getsource(r.decoder.forward),model_forward_source=inspect.getsource(type(r).forward),encoder_forward_source=inspect.getsource(r.encoder.forward),decoder_input_indices=[v['index'] for v in features[-4:]],positions_selected_by_architecture_order=True))
  from phase207_msre_interface import original_core,ParallelMsREInterface,FrozenR4WithParallelMsRE
  torch.manual_seed(cfg['module_seed']);core_type=original_core(REPO/'src/cloud_adapter/models/backbones/target_msre_dinov2.py');adapter=ParallelMsREInterface(cfg['expected_stage_channels'],core_type).cuda();assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in adapter.modules());assert len(list(adapter.named_buffers()))==0
  candidate=FrozenR4WithParallelMsRE(r,adapter);candidate.eval();adapter.enabled=False;init=digest(adapter.state_dict());S['kind']='disabled_path';guard()
  with torch.no_grad():disabled=candidate(x)
  closed_error=float((original-disabled).abs().max());assert closed_error<=1e-5
  adapter.enabled=True;S['kind']='enabled_path_one_loss_forward';guard();enabled=candidate(x);assert enabled.requires_grad and enabled.shape==original.shape;opened_error=float((enabled.detach()-original).abs().max());opened_mean=float((enabled.detach()-original).abs().mean());parent=torch.stack((enabled[:,0],torch.logsumexp(enabled[:,1:3],1),enabled[:,3]),1);loss=torch.nn.functional.cross_entropy(parent,y,ignore_index=255);assert torch.isfinite(loss);loss.backward()
  gradients=[]
  for name,p in adapter.named_parameters():
   g=p.grad;gradients.append(dict(name=name,shape=list(p.shape),parameters=p.numel(),has_gradient=g is not None,finite=bool(torch.isfinite(g).all()) if g is not None else None,nonzero=bool(torch.count_nonzero(g)) if g is not None else None,gradient_L2=float(g.double().norm()) if g is not None else None,gradient_max_abs=float(g.abs().max()) if g is not None else None))
  assert any(q['nonzero'] for q in gradients) and all(q['finite'] for q in gradients if q['has_gradient']);assert all(p.grad is None for m in models for p in m.parameters());assert all(not m.training for m in r.modules()) and all(not m.training for m in n.modules());assert [digest(m.state_dict()) for m in models]==frozen and [digest(dict(m.named_buffers())) for m in models]==buffer_before;assert digest(adapter.state_dict())==init
  from safetensors.torch import save_file
  save_file({k:v.detach().cpu().contiguous() for k,v in adapter.state_dict().items()},str(ROOT/'adapter_initial_only.safetensors'))
  write('GRADIENT_RECEIPT.json',dict(rows=gradients,original_parameters_all_no_grad=True,original_R4_N4_all_eval=True,original_buffers_and_parameters_unchanged=True,new_module_has_BN=False,new_buffer_count=0,optimizer_steps=0,backward_calls=1,loss_kind='fixedparentCE; notanaccuracyestimate',loss_value=float(loss.detach())))
  write('FROZEN_ORIGINALS.json',dict(identities=identities,parameters=[dict(member='R4' if i==0 else 'N4',name=k,shape=list(v.shape),requires_grad=v.requires_grad) for i,m in enumerate(models) for k,v in m.named_parameters()],buffers=[dict(member='R4' if i==0 else 'N4',name=k,shape=list(v.shape)) for i,m in enumerate(models) for k,v in m.named_buffers()]))
  guard();write('IMPLEMENTATION_GATE.json',dict(status='passed',name='MsRE-inspired parallel CNN-interface adapter',not_original_MsRE_unchanged_backbone_transfer=True,core_class_source_sha256=cfg['source_sha256']['src/cloud_adapter/models/backbones/target_msre_dinov2.py'],core_class_AST_sha256=hashlib.sha256(ast.dump(next(q for q in ast.parse((REPO/'src/cloud_adapter/models/backbones/target_msre_dinov2.py').read_text()).body if isinstance(q,ast.ClassDef) and q.name=='SparseMsREResidual'),include_attributes=False).encode()).hexdigest(),core_parameters=sum(p.numel() for p in adapter.core.parameters()),input_projection_parameters=sum(p.numel() for p in adapter.input_projections.parameters()),output_projection_parameters=sum(p.numel() for p in adapter.output_projections.parameters()),new_parameters=sum(p.numel() for p in adapter.parameters()),original_R4_parameters=identities[0]['parameters'],new_parameter_fraction_of_R4=sum(p.numel() for p in adapter.parameters())/identities[0]['parameters'],disabled_output_max_error=closed_error,enabled_initial_output_max_error=opened_error,enabled_initial_output_mean_abs_error=opened_mean,strict_zero_output_init=False,scale_init=.001,all_originals_unchanged=True,new_initial_state_sha256=init,new_state_file_sha256=sha(ROOT/'adapter_initial_only.safetensors'),backward_calls=1,optimizer_steps=0,neural_calls=S['calls'],development_PixBox_access=False,architecture_file_sha256=sha(ROOT/'ARCHITECTURE.json'),elapsed_GPU_stage_seconds=PRIOR_SECONDS+time.monotonic()-START,scope='engineeringfeasibilityonly; noaccuracyclaim'))
  S['status']='completed';guard();(ROOT/'EXIT_CODE').write_text('0\n')
 finally:
  STOP.set();thread.join(timeout=20)
  for h in hooks:h.remove()
if __name__=='__main__':
 try:main()
 except BaseException:
  S.update(status='failed',failure=traceback.format_exc(),stage_seconds=PRIOR_SECONDS+time.monotonic()-START);write('FAILURE.json',S);(ROOT/'EXIT_CODE').write_text('1\n');print(S['failure'],flush=True);os._exit(1)
