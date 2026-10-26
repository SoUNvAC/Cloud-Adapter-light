"""Single controller; no scientific retry; GPU conflict/resource/call guards and post-both-seal CPU."""
import fcntl
import os
import shutil
import subprocess
import sys
import time
import traceback
from phase315_common import *
def active():
    return [l for l in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines() if 'python' in l.lower() and 'phase315' not in l.lower() and 'bash -c' not in l and 'ssh ' not in l and any(k in l.lower() for k in ['controller_phase','control_phase','worker_phase','train_','evaluate_','execute_phase'])]
def main():
    assert sys.executable=='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python';assert ROOT.exists()
    with (ROOT/'CONTROLLER_LOCK').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print('already running');return
        assert not (ROOT/'EXIT_CODE').exists(),'stopped; diagnose before recovery'
        for n,h in load(ROOT/'EXECUTION_LOCK.json')['code'].items():assert sha(R/n)==h,n
        stages=load(ROOT/'STAGES.json');pipeline=[('replay',30201),('train',31501),('train',31502),('evaluate',31501),('evaluate',31502)]
        for stage,seed in pipeline:
            key=f'{stage}_{seed}'
            if key in stages:continue
            conflict=active()
            if conflict:write(ROOT/'STATUS.json',dict(status='waiting_for_gpu',stage=key,conflicts=conflict));print(json.dumps(conflict));return
            d=ROOT/str(seed);d.mkdir(exist_ok=True)
            for n in ['probes','evaluation']:(d/n).mkdir(exist_ok=True)
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',XFORMERS_DISABLED='1',HF_HUB_OFFLINE='1',TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),MPLCONFIGDIR=str(ROOT/'cache'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0')
            started=time.monotonic();peak=0;minimum=2**63;failure=None;write(ROOT/'STATUS.json',dict(status='running',stage=key,controller_PID=os.getpid()))
            with (ROOT/(key+'.log')).open('xb') as log:
                p=subprocess.Popen([sys.executable,'-B',str(R/'src/tools/worker_phase315.py'),stage,str(seed)],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
                while p.poll() is None:
                    try:
                        path=Path('/proc')/str(p.pid)/'status'
                        if path.exists():peak=max(peak,int(next((l.split()[1] for l in path.read_text().splitlines() if l.startswith('VmHWM:')),'0'))*1024)
                    except FileNotFoundError:pass
                    available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024;minimum=min(minimum,available)
                    calls=sum(1 for _ in (ROOT/'CALLS.jsonl').open()) if (ROOT/'CALLS.jsonl').exists() else 0
                    size=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
                    if peak>16*1024**3 or available<4000000000 or shutil.disk_usage(ROOT).free<4000000000:failure='CPU16GiB or hostfree4GB'
                    if sum(v['seconds'] for v in stages.values())+time.monotonic()-started>28800:failure='GPU pipeline8h'
                    if calls>2800:failure='forwards2800'
                    if size*4+100000000>6000000000:failure='allcopies6GB'
                    if failure:p.terminate();break
                    time.sleep(.5)
                code=p.wait()
            receipt=dict(status='completed' if code==0 and not failure else 'failed',exit_code=code,failure=failure,seconds=time.monotonic()-started,peak_cpu_bytes=peak,minimum_available_memory=minimum,PID=p.pid);write(ROOT/(key+'_RESOURCE.json'),receipt)
            if receipt['status']!='completed':write(ROOT/'STATUS.json',dict(status='failed',stage=key,receipt=receipt));(ROOT/'EXIT_CODE').write_text(str(code or 97));return
            stages[key]=receipt;write(ROOT/'STAGES.json',stages)
        write(ROOT/'BOTH_SEALED.json',dict(status='sealed',epochs=[load(ROOT/str(s)/'ENDPOINT_LOCK.json')['sealed_epoch'] for s in [31501,31502]],evaluation_locks={str(s):sha(ROOT/str(s)/'EVALUATION_LOCK.json') for s in [31501,31502]},epoch=time.time(),forwards=sum(1 for _ in (ROOT/'CALLS.jsonl').open())))
        write(ROOT/'STATUS.json',dict(status='gpu_complete_pending_CPU',stages=stages));print('both seeds and evaluations sealed; independent CPU next')
if __name__=='__main__':
    try:
        if '--launch' in sys.argv:
            assert ROOT.exists() and not (ROOT/'LAUNCH.json').exists(),'inspect before another launch'
            with (ROOT/'controller.log').open('xb') as log:
                p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve())],cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(ROOT/'LAUNCH.json',dict(PID=p.pid,epoch=time.time()));print('controller PID',p.pid)
        else:main()
    except BaseException:
        if ROOT.exists():write(ROOT/'CONTROLLER_FAILURE.json',dict(traceback=traceback.format_exc()))
        raise
