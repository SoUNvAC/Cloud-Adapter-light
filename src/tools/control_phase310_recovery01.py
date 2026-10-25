"""Serialized, bounded prepare/train/evaluate/independent-CPU pipeline."""
import fcntl
from datetime import datetime,timezone,timedelta
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from phase310_common import R,ROOT,OLD,PREV,PL,SOURCE,sha,load,write,append

def conflicts():
    return [s for s in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
        if 'python' in s.lower() and 'phase310' not in s.lower() and 'bash -c' not in s and 'ssh ' not in s
        and any(k in s.lower() for k in ('controller_phase','control_phase','train_','/train.py','infer_','evaluate_','execute_phase','worker_phase'))]

def initialize():
    marker=ROOT/'RECOVERY01_LOCK.json'
    if marker.exists():
        for n,h in load(marker)['code'].items():assert sha(R/n)==h,n
        return
    assert load(ROOT/'STATUS.json')['status']=='failed' and load(ROOT/'STATUS.json')['stage']=='train'
    assert not (ROOT/'TRAIN_CURVE.jsonl').exists() and not (ROOT/'ENDPOINT_LOCK.json').exists() and not (ROOT/'MODEL_ACCEPTANCE.json').exists()
    calls=[__import__('json').loads(s) for s in (ROOT/'CALLS.jsonl').read_text().splitlines()]
    assert len(calls)==3 and all(s['context']=='gates' for s in calls)
    failure=(ROOT/'FAILURES.jsonl').read_text();assert "name 't' is not defined" in failure
    saved=ROOT/'recovery01';saved.mkdir()
    for n in ('STATUS.json','STAGES.json','EXIT_CODE','FAILURES.jsonl','train.log','train_RESOURCE.json','initial_lora.pth','PREPROCESS_ACCEPTANCE.json','TRAIN_ACCESS.jsonl'):
        if (ROOT/n).exists():shutil.copyfile(ROOT/n,saved/n)
    shutil.copytree(ROOT/'acceptance',saved/'acceptance')
    for n in ('train.log','train_RESOURCE.json','EXIT_CODE'):(ROOT/n).unlink()
    stages=load(ROOT/'STAGES.json');stages['train_failed01']=load(saved/'train_RESOURCE.json');write(ROOT/'STAGES.json',stages)
    files=[R/'src/tools'/n for n in ('control_phase310_recovery01.py','worker_phase310_recovery01.py')]
    for p in files:shutil.copyfile(p,ROOT/'code'/p.name)
    write(marker,dict(reason='Original accepted-step AST also depends on loop crop coordinates t/l; bind original top/left without changing input or math',code={p.relative_to(R).as_posix():sha(p) for p in files},original_execution_lock_sha256=sha(ROOT/'EXECUTION_LOCK.json'),prior_calls=3,prior_formal_updates=0,original_failure_preserved=True,original_masks_and_protocol_unchanged=True,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))

def main():
    initialize()
    with (ROOT/'CONTROLLER_LOCK').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print('already_running');return
        if (ROOT/'EXIT_CODE').exists():print('already_terminated '+(ROOT/'EXIT_CODE').read_text());return
        execution=load(ROOT/'EXECUTION_LOCK.json')
        for name,h in execution['code'].items():assert sha(R/name)==h,name
        assert sha(R/'src/research_plans/phase310.md')==execution['protocol_sha256']
        for name,h in execution['inputs'].items():assert sha(name)==h,name
        stages=load(ROOT/'STAGES.json')
        for stage in ('prepare','train','evaluate','analyze'):
            if stage in stages:continue
            if stage in ('train','evaluate'):
                active=conflicts()
                if active:
                    write(ROOT/'STATUS.json',dict(status='waiting_for_gpu',stage=stage,conflicts=active,completed=list(stages)))
                    print((ROOT/'STATUS.json').read_text());return
            if stage in ('evaluate','analyze'):assert load(ROOT/'ENDPOINT_LOCK.json')['status']=='sealed'
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',XFORMERS_DISABLED='1',HF_HUB_OFFLINE='1',
                TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),MPLCONFIGDIR=str(ROOT/'cache'),
                OMP_NUM_THREADS='4' if stage!='analyze' else '1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0')
            script='analyze_phase310.py' if stage=='analyze' else 'prepare_phase310.py' if stage=='prepare' else 'worker_phase310_recovery01.py'
            write(ROOT/'STATUS.json',dict(status='running',stage=stage,PID=os.getpid(),completed=list(stages)))
            started=time.monotonic();peak=0;minimum=2**63;failure=None
            prior=sum(v['seconds'] for k,v in stages.items() if k.startswith(('train','evaluate')))
            with (ROOT/(stage+'.log')).open('xb') as log:
                p=subprocess.Popen([sys.executable,'-B',str(R/'src/tools'/script),stage],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
                while p.poll() is None:
                    status=Path('/proc')/str(p.pid)/'status'
                    if status.exists():
                        try:peak=max(peak,int(next((s.split()[1] for s in status.read_text().splitlines() if s.startswith('VmHWM:')),'0'))*1024)
                        except FileNotFoundError:pass
                    available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
                    minimum=min(minimum,available)
                    size=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
                    if peak>16*1024**3:failure='CPU16GiB'
                    if available<4000000000:failure='Host free4GB'
                    if shutil.disk_usage(ROOT).free<4000000000:failure='Disk free4GB'
                    if stage in ('train','evaluate') and time.monotonic()-started+prior>14400:failure='GPU pipeline4h'
                    # Stream directly to local extraction: two persistent copies, no tar archive.
                    if size*2+64000000>4000000000:failure='Two-copy persistent4GB with summary reserve'
                    if failure:p.terminate();break
                    time.sleep(.5)
                code=p.wait()
            receipt=dict(status='completed' if code==0 and not failure else 'failed',exit_code=code,failure=failure,
                PID=p.pid,seconds=time.monotonic()-started,peak_cpu_bytes=peak,minimum_available_memory=minimum)
            write(ROOT/(stage+'_RESOURCE.json'),receipt)
            if receipt['status']!='completed':
                append('FAILURES.jsonl',dict(stage=stage,receipt=receipt))
                write(ROOT/'STATUS.json',dict(status='failed',stage=stage,receipt=receipt));(ROOT/'EXIT_CODE').write_text(str(code or 97));return
            stages[stage]=receipt;write(ROOT/'STAGES.json',stages)
        n=sum(1 for _ in (ROOT/'CALLS.jsonl').open());assert n<=2400
        write(ROOT/'STATUS.json',dict(status='completed',stages=stages,neural_calls=n));(ROOT/'EXIT_CODE').write_text('0')
        print((ROOT/'STATUS.json').read_text())

if __name__=='__main__':
    if '--launch' in sys.argv:
        initialize()
        assert not (ROOT/'RECOVERY01_LAUNCH.json').exists()
        with (ROOT/'controller.log').open('ab') as log:
            p=subprocess.Popen([sys.executable,'-B',__file__],cwd=R,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(ROOT/'RECOVERY01_LAUNCH.json',dict(PID=p.pid));print('recovery01 controller PID',p.pid)
    else:
        try:main()
        except BaseException:append('FAILURES.jsonl',dict(stage='recovery01_controller',traceback=traceback.format_exc()));raise
