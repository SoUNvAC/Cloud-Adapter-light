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
    assert R==Path('/home/scv/Cloud-Adapter-light')
    assert sys.executable=='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
    if ROOT.exists():return
    assert datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')=='20261025','Start date changed; preserve actual date'
    ROOT.mkdir()
    for n in ('tmp','cache','code','masks','acceptance','probes','evaluation'):(ROOT/n).mkdir()
    assert sha(R/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth')==SOURCE
    subprocess.run(['git','diff','--exit-code','0e40461e2defaf405b340c4c66c9d0892cc766d5','--','src/cloud_adapter/models','src/configs'],cwd=R,check=True,capture_output=True)
    files=list((R/'src/tools').glob('*phase310*.py'))+[R/'src/tools'/n for n in ('execute_phase302.py','prepare_phase302.py','phase302_images.py','analyze_phase304.py','worker_phase304.py','transfer_phase302.py')]
    for p in files:shutil.copyfile(p,ROOT/'code'/p.name)
    shutil.copyfile(R/'src/research_plans/phase310.md',ROOT/'code/phase310.md')
    for n in ('FIT_IMAGES.json','EVAL_IMAGES.json','CROP_SCHEDULE.json','SOURCE_ADMISSION.json'):shutil.copyfile(PL/n,ROOT/n)
    evidence=[PL/n for n in ('EXECUTION_LOCK.json','ENDPOINT_LOCK.json','EVALUATION_LOCK.json','PSEUDO_LOCK.json','CROP_REPRODUCTION.csv','initial_lora.pth')]
    original=load(PL/'EXECUTION_LOCK.json')
    for n in ('worker_phase304.py','execute_phase302.py','prepare_phase302.py','phase302_images.py'):
        assert sha(R/'src/tools'/n)==original['code']['src/tools/'+n],n
    assert load(PL/'STATUS.json')['status']=='completed'
    assert load(ROOT/'SOURCE_ADMISSION.json')['source_sha256']==SOURCE
    write(ROOT/'EXECUTION_LOCK.json',dict(code={p.relative_to(R).as_posix():sha(p) for p in files},
        protocol_sha256=sha(R/'src/research_plans/phase310.md'),source_sha256=SOURCE,
        inputs={str(p):sha(p) for p in evidence+[ROOT/n for n in ('FIT_IMAGES.json','EVAL_IMAGES.json','CROP_SCHEDULE.json','SOURCE_ADMISSION.json')]},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
        start_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(),runtime=sys.executable,
        cap_calls=2400,cap_gpu_seconds=14400,cap_cpu_bytes=16*1024**3,min_available_bytes=4000000000,
        cap_new_persistent_including_all_backup_copies=4000000000,
        compute_location='gzs only; latest user instruction overrides local CPU wording',
        training_rule='Fixed Q90 PL + lambda1 T1 forward KL on image-defined P; separate same512 frozen Source; fresh30201 LoRA',
        truth_access='independent CPU only after endpoint seal',prior_phases_read_only=True))
    write(ROOT/'STAGES.json',{})

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
            script='analyze_phase310.py' if stage=='analyze' else 'prepare_phase310.py' if stage=='prepare' else 'worker_phase310.py'
            write(ROOT/'STATUS.json',dict(status='running',stage=stage,PID=os.getpid(),completed=list(stages)))
            started=time.monotonic();peak=0;minimum=2**63;failure=None
            prior=sum(v['seconds'] for k,v in stages.items() if k in ('train','evaluate'))
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
    try:main()
    except BaseException:
        if ROOT.exists():append('FAILURES.jsonl',dict(stage='controller',traceback=traceback.format_exc()))
        raise
