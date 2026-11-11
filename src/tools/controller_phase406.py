"""Sequential two-seed H replication using sealed Phase400 training code."""
from pathlib import Path
import os,json,hashlib,subprocess,time,shutil,signal,traceback,csv
R=Path('/home/scv/shared/phase406_20261101');OLD=Path('/home/scv/shared/phase400_20261031');PY='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python';START=time.monotonic()
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def run(stage,args,env):
 assert time.monotonic()-START<7200
 with (R/(stage+'.log')).open('x') as f:
  p=subprocess.Popen([PY,'-B','-u',*args],stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env,cwd=R)
  save('STATUS.json',{'stage':stage,'PID':p.pid,'seconds':time.monotonic()-START})
  try:rc=p.wait(timeout=max(1,7200-(time.monotonic()-START)))
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise
 assert rc==0,(stage,rc)
def main():
 lock=js(R/'LOCK.json')
 for n,h in lock['old_files'].items():assert sha(OLD/n)==h,n
 for n,h in lock['new_files'].items():assert sha(R/n)==h,n
 import sys
 sys.path.insert(0,str(R));from phase406_penalty import self_test
 save('PENALTY_SELF_TEST.json',self_test())
 sys.path.insert(0,str(OLD/'code'));from phase400_common import hostfree
 hostfree();assert shutil.disk_usage(R).free>20*1024**3
 smi=shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
 assert not subprocess.check_output([smi,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 source=Path('/home/scv/Cloud-Adapter-light/src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth');assert sha(source)==lock['Source_SHA']
 inp=js(OLD/'INPUT_LOCK.json')
 for row in inp['target_rows']:
  for k,suffix in [('rgb_path','_rgb.npy'),('mask_path','_mask.npy'),('image_valid_path','_image_valid.npy')]:assert sha(Path(row[k]))==row['prepared_sha256'][suffix]
 dep=js(OLD/'CODE_LOCK.json')
 for n,h in dep['existing_dependencies_LF'].items():assert hashlib.sha256((Path('/home/scv/Cloud-Adapter-light')/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h
 initial=[];totalcalls=0;totalupdates=0
 for seed in [40501,40502]:
  sr=R/str(seed);sr.mkdir();(sr/'code').mkdir();(sr/'scratch').mkdir()
  for name in ['INPUT_LOCK.json','PREPARED_MANIFEST.json']:shutil.copy2(OLD/name,sr/name)
  # Keep sealed data identity unchanged; actual new RNG settings live in SEED_LOCK.
  save(str(seed)+'/SEED_LOCK.json',{'training_seed':seed,'sampler_seed':seed,'augmentation_seed':seed*100+3,'head_initialization_seed':seed*100+2,'msre_disabled_seed':seed*100+1,'inherited_input_lock_RNG_fields_are_historical':True})
  for name in ['phase400_common.py','phase400_model.py','phase400_runtime.py','worker_phase400.py']:
   text=(OLD/'code'/name).read_text()
   if name=='phase400_common.py':
    text=text.replace('msre_init_seed=4000101','msre_init_seed='+str(seed*100+1)).replace('head_init_seed=4000102','head_init_seed='+str(seed*100+2))
   if name=='phase400_runtime.py':
    text=text.replace('4000103:',str(seed*100+3)+':').replace('seed=40001','seed='+str(seed))
    original="if RUN_ARM!='MH':self.reference=(R/'work_dirs'/RUN_KIND/'MH/STEP_TRACE.jsonl').open()"
    assert original in text;text=text.replace(original,"# Phase406 trains H only; no cross-arm reference trace.")
    anchor='original_save=runner.save_checkpoint'
    assert text.count(anchor)==1
    text=text.replace(anchor,'from phase406_penalty import install;install(model)\n  '+anchor)
   if name=='worker_phase400.py':text=text.replace('seed=40001','seed='+str(seed))
   (sr/'code'/name).write_text(text)
  shutil.copy2(R/'phase406_penalty.py',sr/'code/phase406_penalty.py')
  save(str(seed)+'/CODE_LOCK.json',{p.name:sha(p) for p in (sr/'code').glob('*.py')})
  env={**os.environ,'PHASE400_ROOT':str(sr),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','TMPDIR':str(sr/'scratch')}
  worker=sr/'code/worker_phase400.py'
  run(str(seed)+'_short',[str(worker),'short','H'],env)
  short=js(sr/'short_H_COMPLETE.json');estimate=4000*(short['mean_step_seconds']+short['host_gap_mean'])
  short_trace=[json.loads(x) for x in (sr/'work_dirs/short/H/STEP_TRACE.jsonl').read_text().splitlines()]
  assert short_trace[0]['losses']['loss_residual_energy']==0
  assert all(x['losses']['loss_residual_energy']>0 for x in short_trace[1:])
  assert estimate*2+900<7200-(time.monotonic()-START),'measured training estimate exceeds remaining budget'
  run(str(seed)+'_formal',[str(worker),'formal','H'],env)
  a=js(sr/'work_dirs/short/H/ACTIVE_SCOPE.json');b=js(sr/'work_dirs/formal/H/ACTIVE_SCOPE.json');assert a['head_initial_SHA']==b['head_initial_SHA'] and b['count']==26144
  initial.append(b['head_initial_SHA']);assert len(set(initial))==len(initial)
  paired=Path('/home/scv/shared/phase405_20261101')/str(seed)/'work_dirs/formal/H'
  assert b['head_initial_SHA']==js(paired/'ACTIVE_SCOPE.json')['head_initial_SHA']
  current=[json.loads(x) for x in (sr/'work_dirs/formal/H/STEP_TRACE.jsonl').read_text().splitlines()]
  reference=[json.loads(x) for x in (paired/'STEP_TRACE.jsonl').read_text().splitlines()]
  assert len(current)==len(reference)==4000
  assert all(all(x[k]==y[k] for k in ['step','draw','product','image_SHA','label_SHA','augmentation_seed','LR']) for x,y in zip(current,reference))
  final=sr/'work_dirs/formal/H/iter_4000.pth';assert final.exists()
  ev=sr/'evaluation';ev.mkdir()
  expected={'old_files':lock['old_files'],'H_SHA':sha(final),'Source_SHA':lock['Source_SHA'],'worker_SHA':sha(R/'evaluate_phase406.py')}
  (ev/'INPUT_EXPECTED.json').write_text(json.dumps(expected,indent=2))
  run(str(seed)+'_evaluate',[str(R/'evaluate_phase406.py')],{**env,'PHASE400_ROOT':str(OLD),'PHASE406_EVAL_ROOT':str(ev),'PHASE406_TRAIN_ROOT':str(sr),'PHASE406_CHECKPOINT':str(final)})
  calls=sum(1 for _ in (sr/'CALL_LEDGER.jsonl').open());assert calls==4408,calls
  totalcalls+=calls+js(ev/'RUN_INFO.json')['calls'];totalupdates+=4120
  save(str(seed)+'/COMPLETE.json',{'passed':True,'seed':seed,'updates':4120,'training_calls':calls,'evaluation_calls':3402,'final_checkpoint':str(final),'checkpoint_SHA':sha(final),'initial_head_SHA':initial[-1]})
 assert totalcalls==15620 and totalupdates==8240
 delivery=R/'delivery';delivery.mkdir()
 for name in ['LOCK.json','phase406.md','PENALTY_SELF_TEST.json']:shutil.copy2(R/name,delivery/name)
 for seed in [40501,40502]:
  sr=R/str(seed);dest=delivery/str(seed);dest.mkdir()
  for name in ['summary.csv','per_group.csv','source_per_image.csv','RUN_INFO.json','ENDPOINT_GATE.json','CALL_LEDGER.jsonl','INPUT_EXPECTED.json']:shutil.copy2(sr/'evaluation'/name,dest/name)
  for name in ['SEED_LOCK.json','CODE_LOCK.json','COMPLETE.json']:shutil.copy2(sr/name,dest/name)
  shutil.copy2(sr/'CALL_LEDGER.jsonl',dest/'TRAIN_CALL_LEDGER.jsonl')
  for kind in ['short','formal']:
   for name in ['ACTIVE_SCOPE.json','STEP_TRACE.jsonl','FREEZE_4000.json' if kind=='formal' else 'FREEZE_100.json']:
    shutil.copy2(sr/'work_dirs'/kind/'H'/name,dest/(kind+'_'+name))
  curve=[{'step':s,**js(sr/'evaluations/H'/('step_'+str(s))/'METRICS.json')} for s in range(500,4001,500)]
  (dest/'CURVE.json').write_text(json.dumps(curve))
 save('COMPLETE.json',{'passed':True,'seconds':time.monotonic()-START,'calls':totalcalls,'updates':totalupdates,'formal_updates':8000,'short_updates':240,'seeds':[40501,40502]})
 shutil.copy2(R/'COMPLETE.json',delivery/'COMPLETE.json')
 manifest={p.relative_to(delivery).as_posix():sha(p) for p in delivery.rglob('*') if p.is_file()};save('DELIVERY_MANIFEST.json',manifest)
 (R/'EXIT_CODE').write_text('0')
if __name__=='__main__':
 try:main()
 except BaseException:save('FAILURE.json',{'traceback':traceback.format_exc(),'seconds':time.monotonic()-START});(R/'EXIT_CODE').write_text('1');raise
