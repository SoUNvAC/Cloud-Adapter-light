"""Locked eight-arm cloud/shadow endpoint screen; independent fresh 4000-step promotions."""
from pathlib import Path
import os,sys,json,hashlib,time,subprocess,traceback,shutil,ast
R=Path('/home/scv/shared/phase417_20261101')
OLD=Path('/home/scv/shared/phase400_20261031')
REF=Path('/home/scv/shared/phase411_20261101/layerwise')
PY='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
START=time.monotonic()
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)
def run(mode,steps,count):
 sr=R/(mode if steps==1000 else mode+'_full');sr.mkdir();(sr/'code').mkdir();(sr/'scratch').mkdir()
 for n in ['INPUT_LOCK.json','PREPARED_MANIFEST.json']:shutil.copy2(OLD/n,sr/n)
 for n in ['phase400_common.py','phase400_model.py','phase400_runtime.py','worker_phase400.py']:
  t=(OLD/'code'/n).read_text()
  t=t.replace('4000101','4090101').replace('4000102','4090102').replace('4000103:','4090103:').replace('seed=40001','seed=40901')
  if n=='phase400_model.py':
   marker='from cloud_adapter.models.backbones.utils import set_requires_grad,set_train'
   assert marker in t;t=t.replace(marker,marker+'\nfrom phase417_physics import PhysicsMsRE as SparseMsREResidual')
   t=t.replace('set_train(self,keys);return self','set_train(self,keys);self.target_msre.freeze_inactive();return self')
  if n=='phase400_runtime.py':
   t=t.replace('model.train();return super().__call__(model)',"model.train();wrapper=super().__call__(model);wrapper.semantic_ids={id(p) for n,p in model.named_parameters() if n.startswith('backbone.target_msre.semantic.')};assert len(wrapper.semantic_ids)==2;return wrapper")
   t=t.replace("'M':354433",f"'M':{count}")
   old="if RUN_ARM!='MH':self.reference=(R/'work_dirs'/RUN_KIND/'MH/STEP_TRACE.jsonl').open()"
   assert old in t;t=t.replace(old,'# Paired traces independently checked by controller.')
   t=t.replace("'MsRE_initial_SHA':", "'common_MsRE_SHA':state_sha({k:v for k,v in model.backbone.target_msre.state_dict().items() if k in __import__('phase411_msre').LayerwiseMsRE().state_dict()}),'MsRE_initial_SHA':")
   # Do not construct a reference module inside the hook: that would advance RNG.
   t=t.replace("if k in __import__('phase411_msre').LayerwiseMsRE().state_dict()", "if not k.startswith(('semantic.','material.','operator.','assignment.','offset.','unmatched.','output.'))")
  if n=='phase400_common.py':
   t=t.replace("cfg.model.backbone.type='Phase400TargetMsRECloudAdapter'", "cfg.optim_wrapper.type='Phase416OptimWrapper';cfg.model.type='Phase413Segmentor';cfg.model.backbone.type='Phase400TargetMsRECloudAdapter'")
  if n=='worker_phase400.py':
   t=t.replace('return cfg',"for loader in [cfg.train_dataloader,cfg.val_dataloader,cfg.test_dataloader]:\n  original=loader.dataset.pipeline\n  modified=[]\n  for transform in original:\n   if transform['type']=='PhotoMetricDistortion':modified.append(dict(type='Phase413KeepRaw'))\n   if transform['type']=='PackSegInputs':\n    if not any(t['type']=='Phase413KeepRaw' for t in modified):modified.append(dict(type='Phase413KeepRaw'))\n    transform=dict(transform);transform['type']='Phase413Pack'\n   modified.append(transform)\n  loader.dataset.pipeline=modified\n return cfg",1)
   t=t.replace("return cfg",f"cfg.train_cfg.max_iters={steps};cfg.train_cfg.val_interval=500;cfg.default_hooks.checkpoint.interval={steps}\n return cfg",1)
   t=t.replace("(120 if kind=='short' else 4000)",f"(120 if kind=='short' else {steps})")
  ast.parse(t);(sr/'code'/n).write_text(t)
 for n in ['phase411_msre.py','phase417_physics.py','phase416_optimizer.py']:shutil.copy2(R/n,sr/'code'/n)
 save(sr.name+'/CODE_LOCK.json',{p.name:sha(p) for p in (sr/'code').glob('*.py')})
 env={**os.environ,'PHASE400_ROOT':str(sr),'PHASE413_MODE':mode,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','TMPDIR':str(sr/'scratch')}
 with (R/(sr.name+'.log')).open('x') as log:
  p=subprocess.Popen([PY,'-B','-u',str(sr/'code/worker_phase400.py'),'formal','M'],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  save('STATUS.json',{'mode':sr.name,'PID':p.pid,'steps':steps,'seconds':time.monotonic()-START})
  try:rc=p.wait(timeout=max(1,7200-(time.monotonic()-START)))
  except subprocess.TimeoutExpired:
   import signal
   os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=30);raise
  assert rc==0,(mode,rc)
 traces=[json.loads(x) for x in (sr/'work_dirs/formal/M/STEP_TRACE.jsonl').read_text().splitlines()]
 refs=[json.loads(x) for x in (REF/'work_dirs/formal/M/STEP_TRACE.jsonl').read_text().splitlines()][:steps]
 assert len(traces)==len(refs)==steps
 for a,b in zip(traces,refs):
  assert all(a[k]==b[k] for k in ['step','draw','product','image_SHA','label_SHA','augmentation_seed'])
  assert len(set(a['LR']))==len(set(b['LR']))==1 and a['LR'][0]==b['LR'][0]
 active=js(sr/'work_dirs/formal/M/ACTIVE_SCOPE.json')
 assert active['common_MsRE_SHA']==js(REF/'work_dirs/formal/M/ACTIVE_SCOPE.json')['MsRE_initial_SHA']
 assert js(sr/f'work_dirs/formal/M/FREEZE_{steps}.json')['passed']
 rows=[]
 for step in ([1000] if steps==1000 else range(500,4001,500)):
  m=js(sr/f'evaluations/M/step_{step}/METRICS.json')
  rows.append({'mode':sr.name,'step':step,'mIoU':m['mIoU'],'IoU':m['IoU'],'confusion':m['confusion']})
 calls=sum(1 for _ in (sr/'CALL_LEDGER.jsonl').open());assert calls==steps+32*(steps//500)
 report={'passed':True,'parameters':count,'steps':steps,'calls':calls,'rows':rows,'paired_inputs_and_LR':True,'common_initialization_equal':True,'mean_step_seconds':sum(x['seconds'] for x in traces[20:])/len(traces[20:]),'wall_seconds':js(sr/'formal_M_COMPLETE.json')['wall']}
 save(sr.name+'/ACCEPTANCE.json',report);return report
def main():
 lock=js(R/'LOCK.json')
 for n,h in lock['files'].items():assert sha(R/n)==h,n
 for n,h in lock['old_files'].items():assert sha(OLD/n)==h,n
 for n,h in lock['reference_files'].items():assert sha(REF/n)==h,n
 source=Path('/home/scv/Cloud-Adapter-light/src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth');assert sha(source)==lock['Source_SHA']
 for row in js(OLD/'INPUT_LOCK.json')['target_rows']:
  for k,s in [('rgb_path','_rgb.npy'),('mask_path','_mask.npy'),('image_valid_path','_image_valid.npy')]:assert sha(Path(row[k]))==row['prepared_sha256'][s]
 for n,h in js(OLD/'CODE_LOCK.json')['existing_dependencies_LF'].items():assert hashlib.sha256((Path('/home/scv/Cloud-Adapter-light')/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h,n
 sys.path[:0]=[str(R),str(OLD/'code'),'/home/scv/Cloud-Adapter-light/src']
 from phase417_physics import self_test,MODES
 tests=self_test();save('SELF_TEST.json',tests)
 prior=Path('/home/scv/shared/phase412_20261101/EXIT_CODE')
 while not prior.exists():
  assert time.monotonic()-START<7200
  save('STATUS.json',{'mode':'waiting_for_phase412','seconds':time.monotonic()-START});time.sleep(15)
 assert prior.read_text()=='0','Phase412 failed; do not silently continue queue'
 smi=shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
 assert not subprocess.check_output([smi,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 reports={}
 for mode in MODES:
  reports[mode]=run(mode,1000,tests[mode]['parameters']);save('SCREEN.json',reports)
 baseline=js(REF/'evaluations/M/step_1000/METRICS.json')['mIoU'];eligible=[]
 selected=[]
 save('PROMOTION.json',{'baseline_1000':baseline,'selected':[],'rule':'Diagnostic only; no promotion'})
 save('COMPLETE.json',{'passed':True,'seconds':time.monotonic()-START,'reports':reports,'selected':selected,'new_calls':sum(v['calls'] for v in reports.values()),'new_updates':sum(v['steps'] for v in reports.values())})
 (R/'EXIT_CODE').write_text('0')
if __name__=='__main__':
 try:main()
 except BaseException:save('FAILURE.json',{'traceback':traceback.format_exc()});(R/'EXIT_CODE').write_text('1');raise
