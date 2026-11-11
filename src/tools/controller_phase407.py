"""Four fixed-checkpoint target-only matches, then source evaluation."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,time,signal,shutil,traceback
R=Path('/home/scv/shared/phase407_20261101')
PY='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
START=time.monotonic()
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):(R/name).write_text(json.dumps(value,indent=2))
def main():
 lock=js(R/'LOCK.json')
 for n,h in lock['new_files'].items():assert sha(R/n)==h,n
 assert shutil.disk_usage(R).free>10*1024**3
 total=0
 for job in lock['jobs']:
  name=job['name'];out=R/name;out.mkdir()
  prior=Path(job['prior_root'])
  assert sha(prior/'work_dirs/formal/H/iter_4000.pth')==job['checkpoint_SHA']
  assert sha(prior/'evaluation/per_group.csv')==job['groups_SHA']
  assert sha(prior/'evaluation/summary.csv')==job['summary_SHA']
  shutil.copy2(prior/'evaluation/per_group.csv',out/'REFERENCE_GROUPS.csv')
  expected={**job,'old_files':lock['old_files'],'H_SHA':job['checkpoint_SHA'],'Source_SHA':lock['Source_SHA'],'worker_SHA':sha(R/'evaluate_phase407.py')}
  (out/'INPUT_EXPECTED.json').write_text(json.dumps(expected,indent=2))
  env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PHASE400_ROOT':'/home/scv/shared/phase400_20261031','PHASE407_EVAL_ROOT':str(out),'PHASE407_CHECKPOINT':str(prior/'work_dirs/formal/H/iter_4000.pth'),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
  with (R/(name+'.log')).open('x') as f:
   p=subprocess.Popen([PY,'-B','-u',str(R/'evaluate_phase407.py')],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   save('STATUS.json',{'job':name,'PID':p.pid,'seconds':time.monotonic()-START})
   try:rc=p.wait(timeout=max(1,3600-(time.monotonic()-START)))
   except subprocess.TimeoutExpired:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    raise
  assert rc==0,(name,rc)
  info=js(out/'RUN_INFO.json');assert info['passed'] and info['calls']==855 and info['updates']==0
  total+=info['calls']
 assert total==3420
 delivery=R/'delivery';delivery.mkdir()
 for name in ['LOCK.json','phase407.md']:shutil.copy2(R/name,delivery/name)
 for job in lock['jobs']:
  name=job['name'];dest=delivery/name;dest.mkdir()
  for file in ['summary.csv','per_group.csv','source_per_image.csv','RUN_INFO.json','ENDPOINT_GATE.json','SELECTION.json','CALL_LEDGER.jsonl','INPUT_EXPECTED.json']:
   shutil.copy2(R/name/file,dest/file)
 save('COMPLETE.json',{'passed':True,'calls':total,'updates':0,'seconds':time.monotonic()-START})
 shutil.copy2(R/'COMPLETE.json',delivery/'COMPLETE.json')
 save('DELIVERY_MANIFEST.json',{p.relative_to(delivery).as_posix():sha(p) for p in delivery.rglob('*') if p.is_file()})
 (R/'EXIT_CODE').write_text('0')
if __name__=='__main__':
 try:main()
 except BaseException:
  save('FAILURE.json',{'traceback':traceback.format_exc(),'seconds':time.monotonic()-START});(R/'EXIT_CODE').write_text('1');raise
