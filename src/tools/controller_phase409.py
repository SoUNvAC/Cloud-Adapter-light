"""Sequential two-seed H replication using sealed Phase400 training code."""
from pathlib import Path
import os,json,hashlib,subprocess,time,shutil,signal,traceback,csv
R=Path('/home/scv/shared/phase409_20261101');OLD=Path('/home/scv/shared/phase400_20261031');PY='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python';START=time.monotonic()
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
 sys.path.insert(0,str(R));from phase409_connection import self_test
 save('CONNECTION_SELF_TEST.json',self_test())
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
 for mode in ['serial','parallel']:
  seed=40901
  sr=R/mode;sr.mkdir();(sr/'code').mkdir();(sr/'scratch').mkdir()
  for name in ['INPUT_LOCK.json','PREPARED_MANIFEST.json']:shutil.copy2(OLD/name,sr/name)
  # Keep sealed data identity unchanged; actual new RNG settings live in SEED_LOCK.
  save(mode+'/SEED_LOCK.json',{'training_seed':seed,'sampler_seed':seed,'augmentation_seed':seed*100+3,'head_initialization_seed':seed*100+2,'msre_initialization_seed':seed*100+1,'inherited_input_lock_RNG_fields_are_historical':True})
  for name in ['phase400_common.py','phase400_model.py','phase400_runtime.py','worker_phase400.py']:
   text=(OLD/'code'/name).read_text()
   if name=='phase400_common.py':
    text=text.replace('msre_init_seed=4000101','msre_init_seed='+str(seed*100+1)).replace('head_init_seed=4000102','head_init_seed='+str(seed*100+2))
   if name=='phase400_runtime.py':
    text=text.replace('4000103:',str(seed*100+3)+':').replace('seed=40001','seed='+str(seed))
    original="if RUN_ARM!='MH':self.reference=(R/'work_dirs'/RUN_KIND/'MH/STEP_TRACE.jsonl').open()"
    assert original in text;text=text.replace(original,"# Phase409 trains M only; paired traces checked by controller.")
   if name=='phase400_model.py':
    text='from phase409_connection import connect\nimport os\n'+text
    text=text.replace('x=block(x)','x=block(x);pre_adapter=x')
    text=text.replace('batch,_,height,width=x.shape',"assert list(self.adapter_index)==list(self.out_indices)==list(self.target_msre.injection_indices)==[2,5,8,11]\n  batch,_,height,width=x.shape")
    old='x=self.target_msre(x,index,grid_h,grid_w)'
    assert text.count(old)==1
    text=text.replace(old,"x=connect(self.target_msre,pre_adapter,x,index,grid_h,grid_w,os.environ['PHASE409_CONNECTION'])")
   if name=='worker_phase400.py':text=text.replace('seed=40001','seed='+str(seed))
   (sr/'code'/name).write_text(text)
  shutil.copy2(R/'phase409_connection.py',sr/'code/phase409_connection.py')
  save(mode+'/CODE_LOCK.json',{p.name:sha(p) for p in (sr/'code').glob('*.py')})
  env={**os.environ,'PHASE409_CONNECTION':mode,'PHASE400_ROOT':str(sr),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','TMPDIR':str(sr/'scratch')}
  worker=sr/'code/worker_phase400.py'
  run(mode+'_short',[str(worker),'short','M'],env)
  short=js(sr/'short_M_COMPLETE.json');estimate=4000*(short['mean_step_seconds']+short['host_gap_mean'])
  assert estimate*2+900<7200-(time.monotonic()-START),'measured training estimate exceeds remaining budget'
  run(mode+'_formal',[str(worker),'formal','M'],env)
  a=js(sr/'work_dirs/short/M/ACTIVE_SCOPE.json');b=js(sr/'work_dirs/formal/M/ACTIVE_SCOPE.json');assert a['head_initial_SHA']==b['head_initial_SHA'] and b['count']==354433
  assert a['MsRE_initial_SHA']==b['MsRE_initial_SHA']
  initial.append(b['MsRE_initial_SHA']);assert len(set(initial))==1
  points=[{'step':step,'mIoU':js(sr/'evaluations/M'/f'step_{step}/METRICS.json')['mIoU']} for step in range(500,4001,500)]
  best=max(points,key=lambda x:(x['mIoU'],-x['step']))
  checkpoint=sr/'work_dirs/formal/M'/f"best_mIoU_iter_{best['step']}.pth"
  assert checkpoint.exists()
  save(mode+'/SELECTION.json',{'best':best,'curve':points,'checkpoint':str(checkpoint),'checkpoint_SHA':sha(checkpoint),'rule':'target pooled mIoU only, earliest tie'})
  run(mode+'_source',[str(worker),'evaluate','M',str(checkpoint),'on','M_best'],env)
  calls=sum(1 for _ in (sr/'CALL_LEDGER.jsonl').open());assert calls==4943,calls
  totalcalls+=calls;totalupdates+=4120
  save(mode+'/COMPLETE.json',{'passed':True,'seed':seed,'mode':mode,'updates':4120,'training_calls':4408,'source_evaluation_calls':535,'best_checkpoint':str(checkpoint),'checkpoint_SHA':sha(checkpoint),'initial_MsRE_SHA':initial[-1]})
 assert totalcalls==9886 and totalupdates==8240
 a=[json.loads(x) for x in (R/'serial/work_dirs/formal/M/STEP_TRACE.jsonl').read_text().splitlines()]
 b=[json.loads(x) for x in (R/'parallel/work_dirs/formal/M/STEP_TRACE.jsonl').read_text().splitlines()]
 assert len(a)==len(b)==4000
 assert all(all(x[k]==y[k] for k in ['step','draw','product','image_SHA','label_SHA','augmentation_seed','LR']) for x,y in zip(a,b))
 run('report',[str(R/'report_phase409.py')],{**os.environ,'PHASE400_ROOT':str(OLD),'PYTHONDONTWRITEBYTECODE':'1'})
 delivery=R/'delivery';delivery.mkdir()
 for name in ['LOCK.json','phase409.md','CONNECTION_SELF_TEST.json','TARGET_SUMMARY.csv','TARGET_GROUPS.csv','SOURCE_SUMMARY.json']:shutil.copy2(R/name,delivery/name)
 for mode in ['serial','parallel']:
  seed=40901
  sr=R/mode;dest=delivery/mode;dest.mkdir()
  for name in ['SEED_LOCK.json','CODE_LOCK.json','COMPLETE.json','SELECTION.json','short_M_COMPLETE.json','formal_M_COMPLETE.json']:shutil.copy2(sr/name,dest/name)
  shutil.copy2(sr/'CALL_LEDGER.jsonl',dest/'TRAIN_CALL_LEDGER.jsonl')
  for kind in ['short','formal']:
   for name in ['ACTIVE_SCOPE.json','STEP_TRACE.jsonl','FREEZE_4000.json' if kind=='formal' else 'FREEZE_100.json']:
    shutil.copy2(sr/'work_dirs'/kind/'M'/name,dest/(kind+'_'+name))
  shutil.copy2(sr/'source_evaluations/M_best/METRICS.json',dest/'SOURCE_METRICS.json')
  curve=[{'step':s,**js(sr/'evaluations/M'/('step_'+str(s))/'METRICS.json')} for s in range(500,4001,500)]
  (dest/'CURVE.json').write_text(json.dumps(curve))
 save('COMPLETE.json',{'passed':True,'seconds':time.monotonic()-START,'calls':totalcalls,'updates':totalupdates,'formal_updates':8000,'short_updates':240,'seed':40901,'modes':['serial','parallel']})
 shutil.copy2(R/'COMPLETE.json',delivery/'COMPLETE.json')
 manifest={p.relative_to(delivery).as_posix():sha(p) for p in delivery.rglob('*') if p.is_file()};save('DELIVERY_MANIFEST.json',manifest)
 (R/'EXIT_CODE').write_text('0')
if __name__=='__main__':
 try:main()
 except BaseException:save('FAILURE.json',{'traceback':traceback.format_exc(),'seconds':time.monotonic()-START});(R/'EXIT_CODE').write_text('1');raise
