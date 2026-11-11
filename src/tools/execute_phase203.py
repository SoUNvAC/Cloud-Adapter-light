"""Fixed decoder W/BN exchange, accepted author inference adapter; no training."""
import argparse,csv,gc,hashlib,json,os,subprocess,sys,time,traceback,types,zipfile
from pathlib import Path
R=Path('/home/scv/Cloud-Adapter-light');S=Path('/home/scv/shared');ROOT=S/'phase203_20261018';B=S/'phase111_ocm_runtime'
STATES=('shared_R_00','12201_11','12301_11','12201_10','12201_01','12301_10','12301_01')
def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for a in iter(lambda:f.read(4*1024**2),b''):h.update(a)
 return h.hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.writing');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def verify_code():
 cfg=load(ROOT/'EXECUTION_LOCK.json')
 for p,h in cfg['source_sha256'].items():assert sha(ROOT/p)==h,p
 for p,h in cfg['accepted_source_sha256'].items():assert sha(R/p)==h,p
 return cfg
def old_path(state,pool,pid):
 if state=='shared_R_00':base=S/'phase122_20261015' if pool=='Catalogue' else S/'phase119_20261015';stage='R_Catalogue_reuse' if pool=='Catalogue' else 'R'
 else:base=S/('phase122_20261015' if state.startswith('12201') else 'phase123_20261016');stage='C_'+pool
 path=base/stage/'scores'/pool/(pid+'.npz' if pool=='Catalogue' else pid+'/UNLABELLED_POINT_SCORES.json')
 seal=base/stage/'SCORES_SEALED.json';expected=load(seal)['scores_sha256'][pool+'/'+pid];assert sha(path)==expected
 return path,expected
def preflight():
 cfg=verify_code();sys.path.insert(0,str(R/'src/tools'));from install_phase111_runtime import host_free
 assert host_free()>=4_000_000_000
 for p,h in cfg['model_source_sha256'].items():assert sha(p)==h,p
 plan=load(ROOT/'EVALUATION_INPUT_PLAN.json')['rows'];assert len(plan)==48 and sum(len(r['actual_patch_coordinates']) for r in plan)==1146
 records=[]
 for state in STATES[:3]:
  for r in plan:
   p,h=old_path(state,r['pool'],r['product_id']);records.append(dict(state=state,pool=r['pool'],product_id=r['product_id'],path=str(p),sha256=h,bytes=p.stat().st_size))
 write(ROOT/'OLD_SCORE_INDEX.json',dict(rows=records,labels_accessed=False));write(ROOT/'PREFLIGHT.json',dict(status='passed',host_free=host_free(),old_score_files=len(records),neural_calls=0,plan_groups=48))
def context(state):
 import numpy as np,torch
 from safetensors.torch import load_file
 sys.path.insert(0,str(R/'src/tools'))
 import phase119_common as c
 from phase120_common import decoder_names,digest_state
 from omnicloudmask import model_utils as mu
 cfg=verify_code();w=load(ROOT/'WHITELIST.json');dest=ROOT/state
 x=types.ModuleType('phase203_context');x.ROOT=dest;x.BASE=B;x.REPO=R;x.LOCK=ROOT/'EXECUTION_LOCK.json';x.OLD=c.OLD
 for n in ('load','write','sha','ram','coordinates','image','catalog_image','verify_native_and_cache','original_pix_score'):setattr(x,n,getattr(c,n))
 def coords():
  # Accepted helper binds raster-only stat hashes after each asset verification.
  # Restore its original content hash before the next pool's coordinate check.
  c.p118.sha=c.sha
  return c.coordinates()
 x.coordinates=coords
 def setup():
  verify_code()
  for k in ('HF_HOME','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR','CUDA_CACHE_PATH','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_EXTENSIONS_DIR','TMPDIR'):os.environ[k]=str(B/'controlled_caches'/k.lower())
  os.environ.update(HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',GDAL_NUM_THREADS='1',OPJ_NUM_THREADS='1')
  sys.path.insert(0,str(B/'assets/vendor'))
  rows=load(dest/'INPUT_PATCH_PLAN.json')['rows'];return dict(benchmark_products=[r['product_id'] for r in rows if r['pool']=='PixBox'][:1])
 x.setup=setup;cache={};original_loader=mu.load_model_from_weights
 def cached_loader(**kw):
  if 'model' not in cache:
   model=original_loader(**kw);full={n:v.detach().cpu().clone() for n,v in model.state_dict().items()};cache.update(model=model,full=full)
  return cache['model']
 mu.load_model_from_weights=cached_loader
 def configure(model):
  assert model is cache['model'];full=cache['full'];enc,params,dec=decoder_names(model)
  assert set(params)==set(w['trainable_names']) and set(dec)==set(w['decoder_state_names'])
  assert [(n,type(v).__module__+'.'+type(v).__name__) for n,v in model.named_modules()]==[tuple(v) for v in w['graph_signature']['modules']]
  assert {n:list(v.shape) for n,v in model.state_dict().items()}==w['graph_signature']['state']
  buffers=set(dec)-set(params);assert len(params)==46 and len(buffers)==15
  namedbuf=dict(model.named_buffers());namedparam=dict(model.named_parameters());assert set(params)<=set(namedparam) and buffers<=set(namedbuf)
  assert all(n.endswith(('.running_mean','.running_var','.num_batches_tracked')) for n in buffers)
  for n in buffers:assert isinstance(dict(model.named_modules())[n.rsplit('.',1)[0]],torch.nn.modules.batchnorm._BatchNorm)
  assert digest_state({n:full[n] for n in w['encoder_names']})==w['initial_encoder_sha256']
  initial=load_file(str(S/'phase122_20261015/train_C/decoder_initial.safetensors'),device='cpu')
  assert set(initial)==set(dec) and digest_state(initial)==w['initial_decoder_sha256']
  assert all(torch.equal(initial[n],full[n]) and initial[n].numpy().tobytes()==full[n].numpy().tobytes() for n in dec)
  final=initial if state=='shared_R_00' else load_file(str(S/('phase122_20261015' if state.startswith('12201') else 'phase123_20261016')/'C_decoder_final.safetensors'),device='cpu')
  assert set(final)==set(dec);mode=state.rsplit('_',1)[1];selected={n:(final[n] if mode[0 if n in params else 1]=='1' else initial[n]) for n in dec}
  merged=dict(full);merged.update(selected);result=model.load_state_dict(merged,strict=True);assert not result.missing_keys and not result.unexpected_keys
  model.eval()
  for p in model.parameters():p.requires_grad_(False)
  actual={n:v.detach().cpu() for n,v in model.state_dict().items()}
  assert all(actual[n].numpy().tobytes()==full[n].numpy().tobytes() for n in w['encoder_names'])
  assert all(actual[n].numpy().tobytes()==selected[n].numpy().tobytes() for n in dec)
  receipt=dict(state=state,encoder_sha256=digest_state({n:actual[n] for n in w['encoder_names']}),decoder_sha256=digest_state({n:actual[n] for n in dec}),full_sha256=digest_state(actual),parameters=sorted(params),BN_buffers=sorted(buffers),tensor_source={n:('adapted' if mode[0 if n in params else 1]=='1' else 'original') for n in dec},eval=True,trainable_parameters=0)
  old=dest/'STATE_IDENTITY.json'
  if old.exists():assert load(old)==receipt
  else:write(old,receipt)
  cache['receipt']=receipt
 x.configure_model=configure
 def verify_after():
  m=cache['model'];assert all(not v.training for v in m.modules());assert digest_state(m.state_dict())==cache['receipt']['full_sha256']
 x.verify_after=verify_after
 sys.modules['phase203_context']=x;return x
def adapted_source(source):
 source=source.replace('from phase122_common import ', 'from phase203_context import ')
 source=source.replace("    if stage!='Calibration':assert (ROOT/'THRESHOLD_LOCK.json').exists()","    assert stage in ('Catalogue','PixBox')")
 start=source.index("    if policy in ('C','D'):\n");end=source.index('    model_info=',start)
 source=source[:start]+"    from phase203_context import configure_model\n    configure_model(models[0])\n"+source[end:]
 source=source.replace('3500','5000').replace('size<=4000000000','size<=6000000000')
 source=source.replace('    threading.Thread(target=monitor,daemon=True).start()', '    monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()')
 before="guard();S['status']='completed_scores_sealed';ended.set();write(out/'RUN_STATUS.json',S)"
 after="ended.set();monitor_thread.join(timeout=60);assert not monitor_thread.is_alive();guard();S['status']='completed_scores_sealed';write(out/'RUN_STATUS.json',S)"
 assert before in source
 source=source.replace(before,after)
 assert 'load_file(str(checkpoint)' not in source and 'THRESHOLD_LOCK' not in source
 return source
def spot_verify(state,pool):
 import numpy as np
 rows=load(ROOT/state/'INPUT_PATCH_PLAN.json')['rows'];receipts=[]
 for row in [r for r in rows if r['pool']==pool]:
  pid=row['product_id'];old,h=old_path(state,pool,pid);new=ROOT/state/('R_'+pool)/'scores'/pool/(pid+'.npz' if pool=='Catalogue' else pid+'/UNLABELLED_POINT_SCORES.json')
  if pool=='Catalogue':
   with np.load(old,allow_pickle=False) as a,np.load(new,allow_pickle=False) as b:
    assert np.array_equal(a['flat_indices'],b['flat_indices']) and np.array_equal(a['valid'],b['valid']);v=a['valid'];error=float(np.max(np.abs(a['scores'][v]-b['scores'][v])))
  else:
   a={r['point_id']:r for r in load(old)['points']};b={r['point_id']:r for r in load(new)['points']};assert set(a)==set(b);errors=[]
   for k,r in a.items():
    assert r['valid']==b[k]['valid']
    if r['valid']:errors.append(abs(r['shadow_confidence']-b[k]['shadow_confidence']))
   error=max(errors,default=0.)
  assert error<=1e-5,(state,pool,error);receipts.append(dict(state=state,pool=pool,product_id=pid,max_Shadow_confidence_error=error,old_sha256=h,new_sha256=sha(new),same_support=True))
 write(ROOT/state/(pool+'_REPRODUCTION.json'),dict(rows=receipts,passed=True))
def worker(state):
 verify_code();dest=ROOT/state;assert state in STATES
 sys.path.insert(0,str(B/'assets/vendor'));sys.path.insert(0,str(R/'src/tools'))
 # Cache variables are redirected before importing Torch/vendor.
 for k in ('HF_HOME','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR','CUDA_CACHE_PATH','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TORCH_EXTENSIONS_DIR','TMPDIR'):os.environ[k]=str(B/'controlled_caches'/k.lower())
 os.environ.update(HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
 ctx=context(state);source=adapted_source((R/'src/tools/infer_phase122.py').read_text());write(dest/'ADAPTER_SOURCE.json',dict(source_sha256=hashlib.sha256(source.encode()).hexdigest()))
 ns={'__name__':'phase203_accepted_adapter','__file__':str(R/'src/tools/infer_phase122.py')};exec(compile(source,ns['__file__'],'exec'),ns)
 for pool in ('Catalogue','PixBox'):
  accepted=dest/('R_'+pool)
  if accepted.exists():
   assert (accepted/'EXIT_CODE').read_text().strip()=='0' and load(accepted/'RUN_STATUS.json')['status']=='completed_scores_sealed'
   for key,h in load(accepted/'SCORES_SEALED.json')['scores_sha256'].items():
    a,pid=key.split('/',1);path=accepted/'scores'/a/(pid+'.npz' if a=='Catalogue' else pid+'/UNLABELLED_POINT_SCORES.json');assert sha(path)==h
   if state in STATES[:3]:assert load(dest/(pool+'_REPRODUCTION.json'))['passed']
   continue
  ns['S']={'status':'initializing','construction_calls':0,'main_calls':0,'warmup_calls':0};ns['START']=time.monotonic()
  try:
   ns['run_worker']('R',pool);ctx.verify_after()
   if state in STATES[:3]:spot_verify(state,pool)
   ctrl=load(dest/'CONTROLLER_STATUS.json');ctrl['calls_completed']+=load(dest/('R_'+pool)/'RUN_STATUS.json')['all_calls'];write(dest/'CONTROLLER_STATUS.json',ctrl)
  except BaseException:
   write(dest/'FAILURE.json',dict(error=traceback.format_exc(),state=state,pool=pool,worker_counts=ns.get('S',{})));raise
 write(dest/'COMPLETE.json',dict(status='completed',all_calls=sum(load(dest/('R_'+p)/'RUN_STATUS.json')['all_calls'] for p in ('Catalogue','PixBox')),state_unchanged_in_eval=True))
def controller():
 verify_code();preflight();recovery=load(ROOT/'RECOVERY_LOCK.json') if (ROOT/'RECOVERY_LOCK.json').exists() else {};started=(time.time()-recovery['prior_active_stage_seconds']) if 'prior_active_stage_seconds' in recovery else recovery.get('original_started_epoch',time.time());status=dict(status='running',started_epoch=started,original_started_epoch=recovery.get('original_started_epoch',started),prior_active_stage_seconds=recovery.get('prior_active_stage_seconds',0),calls=recovery.get('prior_consumed_calls',0),completed=list(recovery.get('completed_states',[])),PID=os.getpid());write(ROOT/'RUN_STATUS.json',status)
 from install_phase111_runtime import host_free
 try:
  for state in STATES:
   if state in status['completed']:
    assert load(ROOT/state/'COMPLETE.json')['status']=='completed';continue
   if state not in STATES[:3]:assert all((ROOT/s/'COMPLETE.json').exists() for s in STATES[:3])
   dest=ROOT/state
   if dest.exists():assert recovery.get('resume_state')==state and not (dest/'COMPLETE.json').exists()
   else:dest.mkdir()
   previous_counts=sum(load(p).get('all_calls',0) for p in dest.glob('R_*/RUN_STATUS.json'))
   rows=load(ROOT/'EVALUATION_INPUT_PLAN.json')['rows']
   if state in STATES[:3]:rows=[r for r in rows if r['product_id'] in load(ROOT/'EXECUTION_LOCK.json')['spot_products']]
   write(dest/'INPUT_PATCH_PLAN.json',dict(rows=rows));write(dest/'CONTROLLER_STATUS.json',dict(calls_completed=status['calls'],started_epoch=started));status.update(current=state);write(ROOT/'RUN_STATUS.json',status)
   with (dest/'console.log').open('xb') as f:
    q=subprocess.Popen([sys.executable,'-B',str(ROOT/'code/execute_phase203.py'),'--worker',state],stdout=f,stderr=subprocess.STDOUT,cwd=R);status['worker_PID']=q.pid;write(ROOT/'RUN_STATUS.json',status)
    while q.poll() is None:
     if time.time()-started>7200 or host_free()<4_000_000_000 or sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())>6_000_000_000:q.terminate();q.wait();raise AssertionError('global resource limit')
     time.sleep(15)
   counts={p:load(dest/('R_'+p)/'RUN_STATUS.json').get('all_calls',0) if (dest/('R_'+p)/'RUN_STATUS.json').exists() else 0 for p in ('Catalogue','PixBox')}
   if (dest/'FAILURE.json').exists():
    failure=load(dest/'FAILURE.json');v=failure['worker_counts'];counts[failure['pool']]=max(counts[failure['pool']],sum(v.get(k,0) for k in ('construction_calls','main_calls','warmup_calls')))
   consumed=sum(counts.values())-previous_counts;assert consumed>=0;status['calls']+=consumed;status['elapsed_seconds']=time.time()-started;write(ROOT/'RUN_STATUS.json',status)
   assert q.returncode==0,state;assert status['calls']<=5000;assert (dest/'COMPLETE.json').exists();status['completed'].append(state);write(ROOT/'RUN_STATUS.json',status)
  status.update(status='completed_pending_local_delivery',elapsed_seconds=time.time()-started);write(ROOT/'RUN_STATUS.json',status);(ROOT/'EXIT_CODE').write_text('0\n')
 except BaseException:
  status.update(status='failed',error=traceback.format_exc(),elapsed_seconds=time.time()-started);write(ROOT/'RUN_STATUS.json',status);(ROOT/'EXIT_CODE').write_text('1\n');raise
def main():
 p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--worker',choices=STATES);p.add_argument('--preflight',action='store_true');a=p.parse_args()
 sys.path.insert(0,str(R/'src/tools'))
 if a.worker:worker(a.worker)
 elif a.preflight:preflight();print(json.dumps(load(ROOT/'PREFLIGHT.json')))
 elif a.launch:
  verify_code();assert not (ROOT/'LAUNCH.json').exists()
  with (ROOT/'console.log').open('xb') as f:q=subprocess.Popen([sys.executable,'-B',str(ROOT/'code/execute_phase203.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=R,close_fds=True)
  write(ROOT/'LAUNCH.json',dict(PID=q.pid,epoch=time.time()));print(json.dumps(load(ROOT/'LAUNCH.json')))
 else:controller()
if __name__=='__main__':main()
