"""Single serial GPU controller; durable stage receipts, no automatic scientific retries."""
import fcntl,subprocess,traceback
from datetime import datetime,timezone,timedelta
from phase318_common import *
def active():
 return [s for s in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines() if 'python' in s.lower() and 'phase318' not in s and not any(t in s for t in ['ssh ','bash -c']) and any(t in s.lower() for t in ['control_phase','controller_phase','worker_phase','train_','evaluate_','execute_phase'])]
def main():
 assert sys.executable=='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
 with (ROOT/'CONTROLLER_LOCK').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:print('alreadyrunning');return
  assert not (ROOT/'EXIT_CODE').exists(),'inspect prior failure; never retry blindly'
  for n,h in load(ROOT/'GPU_CODE_LOCK.json').items():assert sha(R/n)==h,n
  stages=load(ROOT/'STAGES.json');pipeline=[('gates','GLOBAL','Source')]+[('teacher',p,'Source') for p in ['CMC','SPARCS','L8']]+[('selection','GLOBAL','Source')]+[('train',p,a) for p,a in ARMS]+[('evaluate',p,a) for p,a in ARMS]
  for stage,pool,arm in pipeline:
   key='_'.join([stage,pool,arm])
   if key in stages:continue
   while conflict:=active():write(ROOT/'STATUS.json',dict(status='waiting_for_gpu',next_stage=key,conflicts=conflict,controller_PID=os.getpid()));time.sleep(10)
   resource_check();d=ROOT/pool/arm;d.mkdir(parents=True,exist_ok=True)
   for n in ['probes','evaluation']:(d/n).mkdir(exist_ok=True)
   assert sha(ROOT/'DECISION_WINDOW.json')==load(ROOT/'FROZEN_CONFIG.json')['decision_window_sha']
   if stage=='evaluate' and not (ROOT/'ALL_TRAINED.json').exists():
    locks={p+'_'+a:sha(ROOT/p/a/('EQUIVALENT_ARM.json' if (ROOT/p/a/'EQUIVALENT_ARM.json').exists() else 'ENDPOINT_LOCK.json')) for p,a in ARMS};write(ROOT/'ALL_TRAINED.json',dict(status='sealed',endpoints=locks,epoch=time.time()))
   env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',XFORMERS_DISABLED='1',HF_HUB_OFFLINE='1',TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),MPLCONFIGDIR=str(ROOT/'cache'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0')
   write(ROOT/'STATUS.json',dict(status='running',stage=key,controller_PID=os.getpid()));start=time.monotonic();peak=0;minmem=2**63;failure=None
   with (ROOT/(key+'.log')).open('xb') as log:
    process=subprocess.Popen([sys.executable,'-B',str(R/'src/tools/worker_phase318.py'),stage,pool,arm],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
    while process.poll() is None:
     try:
      pp=Path('/proc')/str(process.pid)/'status'
      if pp.exists():peak=max(peak,int(next((s.split()[1] for s in pp.read_text().splitlines() if s.startswith('VmHWM:')),'0'))*1024)
      mem=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024;minmem=min(mem,minmem)
      calls=sum(1 for _ in (ROOT/'CALLS.jsonl').open()) if (ROOT/'CALLS.jsonl').exists() else 0
      if peak>16*1024**3 or mem<4*1024**3 or shutil.disk_usage(ROOT).free<8*1024**3:failure='CPU16GiB/freeRAM4GiB/freedisk8GiB'
      if sum(r['seconds'] for r in stages.values())+time.monotonic()-start>43200:failure='GPU12hours'
      if calls>11000 or budget_bytes()>12000000000:failure='forward11000 or allpersistent12GB'
     except FileNotFoundError:pass
     if failure:process.terminate();break
     time.sleep(1)
    code=process.wait()
   result=dict(status='completed' if code==0 and not failure else 'failed',exit=code,failure=failure,seconds=time.monotonic()-start,peak_CPU=peak,minimum_host_available=minmem,pid=process.pid);write(ROOT/(key+'_RESOURCE.json'),result)
   if result['status']!='completed':write(ROOT/'STATUS.json',dict(status='failed',stage=key,receipt=result));(ROOT/'EXIT_CODE').write_text(str(code or 97));return
   stages[key]=result;write(ROOT/'STAGES.json',stages)
  write(ROOT/'ALL_SEALED.json',dict(status='sealed',endpoints=load(ROOT/'ALL_TRAINED.json')['endpoints'],evaluation={p+'_'+a:sha(ROOT/p/a/('EVALUATION_REFERENCE.json' if (ROOT/p/a/'EVALUATION_REFERENCE.json').exists() else 'EVALUATION_LOCK.json')) for p,a in ARMS},epoch=time.time(),forwards=sum(1 for _ in (ROOT/'CALLS.jsonl').open()),gpu_seconds=sum(r['seconds'] for r in stages.values())))
  write(ROOT/'STATUS.json',dict(status='GPU_complete_pending_independent_CPU',forwards=sum(1 for _ in (ROOT/'CALLS.jsonl').open())));print('Phase318 all selected endpoints and evaluations sealed')
if __name__=='__main__':
 try:
  if '--launch' in sys.argv:
   assert not (ROOT/'LAUNCH.json').exists(),'inspect before relaunch'
   with (ROOT/'controller.log').open('xb') as log:p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve())],cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   write(ROOT/'LAUNCH.json',dict(PID=p.pid,epoch=time.time()));print('controllerPID',p.pid)
  else:main()
 except BaseException:write(ROOT/'CONTROLLER_FAILURE.json',dict(traceback=traceback.format_exc()));raise
