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
from phase304_common import R,ROOT,OLD,PREV,SOURCE,sha,load,write,append

def conflicts():
    return [s for s in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
        if 'python' in s.lower() and 'phase304' not in s.lower() and 'bash -c' not in s and 'ssh ' not in s
        and any(k in s.lower() for k in ('controller_phase','control_phase','train_','/train.py','infer_','evaluate_','execute_phase','worker_phase'))]

def initialize():
    marker=ROOT/'RECOVERY01_LOCK.json'
    if marker.exists():
        for name,h in load(marker)['code'].items():assert sha(R/name)==h,name
        return
    state=load(ROOT/'STATUS.json');assert state['status']=='failed' and state['stage']=='evaluate'
    assert (ROOT/'EXIT_CODE').read_text().strip()=='1'
    stages=load(ROOT/'STAGES.json');assert set(stages)=={'prepare','train'}
    assert all(v['exit_code']==0 for v in stages.values())
    calls=[__import__('json').loads(s) for s in (ROOT/'CALLS.jsonl').read_text().splitlines()]
    assert len(calls)==1852 and not any('evaluation' in r['kind'] for r in calls)
    assert not list((ROOT/'evaluation').iterdir()) and not (ROOT/'EVALUATION_LOCK.json').exists()
    failure=(ROOT/'FAILURES.jsonl').read_text()
    assert 'PermissionError' in failure and 'EVAL_IMAGES.json' in failure
    end=load(ROOT/'ENDPOINT_LOCK.json');assert end['status']=='sealed' and end['updates']==955 and end['skipped']==45
    for name,key in [('final_lora.pth','final_sha256'),('PSEUDO_LOCK.json','pseudo_lock_sha256'),('TRAIN_CURVE.jsonl','curve_sha256'),('EXECUTION_LOCK.json','execution_lock_sha256')]:assert sha(ROOT/name)==end[key]
    saved=ROOT/'recovery01';saved.mkdir()
    for name in ('STATUS.json','STAGES.json','EXIT_CODE','evaluate.log','evaluate_RESOURCE.json','FAILURES.jsonl'):
        shutil.copyfile(ROOT/name,saved/name)
    for name in ('evaluate.log','evaluate_RESOURCE.json','EXIT_CODE'):(ROOT/name).unlink()
    stages['evaluate_failed_01']=load(saved/'evaluate_RESOURCE.json');write(ROOT/'STAGES.json',stages)
    files=[R/'src/tools'/n for n in ('control_phase304_recovery01.py','evaluate_phase304_recovery01.py')]
    for p in files:shutil.copyfile(p,ROOT/'code'/p.name)
    write(marker,dict(reason='Manifest identity SHA must precede image_guard, which intentionally blocks EVAL_IMAGES opens',
        code={p.relative_to(R).as_posix():sha(p) for p in files},endpoint_sha256=sha(ROOT/'ENDPOINT_LOCK.json'),
        execution_lock_sha256=sha(ROOT/'EXECUTION_LOCK.json'),previous_calls=1852,failed_evaluation_calls=0,
        training_unchanged=True,rules_unchanged=True,original_failure_preserved=True,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()))

def main():
    with (ROOT/'CONTROLLER_LOCK').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print('already_running');return
        initialize()
        if (ROOT/'EXIT_CODE').exists():print('already_terminated '+(ROOT/'EXIT_CODE').read_text());return
        execution=load(ROOT/'EXECUTION_LOCK.json')
        for name,h in execution['code'].items():assert sha(R/name)==h,name
        assert sha(R/'src/research_plans/phase304.md')==execution['protocol_sha256']
        for name,h in execution['inputs'].items():assert sha(name)==h,name
        stages=load(ROOT/'STAGES.json')
        for stage in ('prepare','train','evaluate','analyze'):
            if stage in stages:continue
            if stage!='analyze':
                active=conflicts()
                if active:
                    write(ROOT/'STATUS.json',dict(status='waiting_for_gpu',stage=stage,conflicts=active,completed=list(stages)))
                    print((ROOT/'STATUS.json').read_text());return
            if stage in ('evaluate','analyze'):assert load(ROOT/'ENDPOINT_LOCK.json')['status']=='sealed'
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',XFORMERS_DISABLED='1',HF_HUB_OFFLINE='1',
                TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),MPLCONFIGDIR=str(ROOT/'cache'),
                OMP_NUM_THREADS='4' if stage!='analyze' else '1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            assert stage in ('evaluate','analyze'),'Recovery cannot train or prepare'
            script='analyze_phase304.py' if stage=='analyze' else 'evaluate_phase304_recovery01.py'
            write(ROOT/'STATUS.json',dict(status='running',stage=stage,PID=os.getpid(),completed=list(stages)))
            started=time.monotonic();peak=0;minimum=2**63;failure=None
            prior=sum(v['seconds'] for k,v in stages.items() if k!='analyze')
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
                    if stage!='analyze' and time.monotonic()-started+prior>14400:failure='GPU pipeline4h'
                    # Stream directly to local extraction: two persistent copies, no tar archive.
                    if size*2+64000000>6000000000:failure='Two-copy persistent6GB with summary reserve'
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
        n=sum(1 for _ in (ROOT/'CALLS.jsonl').open());assert n<=3000
        write(ROOT/'STATUS.json',dict(status='completed',stages=stages,neural_calls=n));(ROOT/'EXIT_CODE').write_text('0')
        print((ROOT/'STATUS.json').read_text())

if __name__=='__main__':
    if sys.argv[-1]=='--launch':
        with (ROOT/'recovery01_controller.log').open('xb') as log:
            p=subprocess.Popen([sys.executable,'-B',__file__,'run'],cwd=R,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
        print('Recovery01 controller PID '+str(p.pid))
    else:
        try:main()
        except BaseException:
            if ROOT.exists():append('FAILURES.jsonl',dict(stage='recovery01_controller',traceback=traceback.format_exc()))
            raise
