"""One-shot CPU stages; no remote scheduler, no neural work or 241 mutation."""
from phase242_common import *
import sys,traceback
def main():
 assert not (R/'CONTROLLER_STARTED.json').exists();assert shutil.disk_usage(R).free>50*1024**3;hostfree()
 for n,h in js(R/'BUNDLE_SHA.json').items():assert sha(R/n)==h
 save('CONTROLLER_STARTED.json',{'PID':os.getpid(),'epoch':time.time(),'network_calls':0})
 stages=[('compile',['/usr/bin/g++','-O3','-std=c++17',str(R/'code/phase242_rank.cpp'),'-o',str(R/'code/rank242')]),('kernel',[PY,'-B',str(R/'code/test_phase242.py')]),('smoke',[PY,'-B',str(R/'code/smoke_phase242.py')]),('val',[PY,'-B',str(R/'code/analyze_phase242.py'),'val']),('test',[PY,'-B',str(R/'code/analyze_phase242.py'),'test']),('validate',[PY,'-B',str(R/'code/validate_phase242.py')]),('finalize',[PY,'-B',str(R/'code/finalize_phase242.py')])]
 for stage,cmd in stages:
  started=time.monotonic();save('CONTROLLER_STATUS.json',{'PID':os.getpid(),'stage':stage,'epoch':time.time()})
  with (R/(stage+'.log')).open('x') as f:
   p=subprocess.Popen(cmd,cwd=R/'code',stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'});save('WORKER.json',{'PID':p.pid,'stage':stage})
   try:rc=p.wait(timeout=max(1,43200-cpu_seconds()))
   except subprocess.TimeoutExpired:
    import signal
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    save(R/'cpu_costs'/(stage+'_controller_timeout.json'),{'status':'timeout','seconds':time.monotonic()-started,'peak_RSS_bytes':None});raise
  if stage in ('compile','kernel','smoke','finalize'):save(R/'cpu_costs'/(stage+'.json'),{'status':'passed' if not rc else 'failed','seconds':time.monotonic()-started,'peak_RSS_bytes':None,'resource_note':'small engineering stage peak not sampled'})
  with (R/'STAGE_JOURNAL.jsonl').open('a') as journal:journal.write(json.dumps({'stage':stage,'seconds':time.monotonic()-started,'returncode':rc,'network_calls':0,'gradient_updates':0})+'\n')
  if rc:raise RuntimeError(stage+' exited '+str(rc)+'; preserve records, no blind restart')
 # Close bookkeeping without changing scientific outputs or any val/test lock.
 d=R/'delivery'
 for n in ('cpu_costs/finalize.json','STAGE_JOURNAL.jsonl'):shutil.copy2(R/n,d/n)
 save(d/'FINAL_RUNTIME_COST.json',{'CPU_seconds_including_packaging':cpu_seconds(),'network_calls':0,'gradient_updates':0,'downloads':0});manifest={p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file() and p.name not in ('SHA256SUMS.json','DELIVERY_COMPLETE.json')};save(d/'SHA256SUMS.json',manifest);size=sum(p.stat().st_size for p in d.rglob('*') if p.is_file());assert size<=500*1024**2;save('DELIVERY_COMPLETE.json',{'passed':True,'epoch':time.time(),'formal_bytes':size,'manifest_SHA':sha(d/'SHA256SUMS.json'),'no_full_pixel_arrays_transferred':True});save(d/'DELIVERY_COMPLETE.json',js(R/'DELIVERY_COMPLETE.json'))
 save('CONTROLLER_STATUS.json',{'PID':os.getpid(),'stage':'complete','epoch':time.time()});(R/'EXIT_CODE').write_text('0\n')
if __name__=='__main__':
 try:main()
 except BaseException:
  save('CONTROLLER_FAILURE.json',{'traceback':traceback.format_exc()});(R/'EXIT_CODE').write_text('2\n');raise
