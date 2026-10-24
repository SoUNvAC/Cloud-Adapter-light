"""One bounded CPU-first diagnostic; wait rather than compete for the GPU."""
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
from phase303_common import R,OLD,ROOT,sha,load,write


def initialize():
    assert sys.executable=='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
    if ROOT.exists():return
    ROOT.mkdir();(ROOT/'tmp').mkdir();(ROOT/'cache').mkdir();(ROOT/'code').mkdir()
    files=list((R/'src/tools').glob('*phase303*.py'))+[R/'src/tools/execute_phase302.py',R/'src/tools/phase302_images.py',R/'src/tools/prepare_phase302.py']
    code={p.relative_to(R).as_posix():sha(p) for p in files}
    for p in files:shutil.copyfile(p,ROOT/'code'/p.name)
    shutil.copyfile(R/'src/research_plans/phase303.md',ROOT/'code/phase303.md')
    source=R/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth'
    assert sha(source)=='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
    # Model/config implementation must match the Phase302 GPU launch commit.
    subprocess.run(['git','diff','--exit-code','0e40461e2defaf405b340c4c66c9d0892cc766d5','--','src/cloud_adapter/models','src/configs'],cwd=R,check=True,capture_output=True)
    write(ROOT/'EXECUTION_LOCK.json',dict(code=code,source_sha256=sha(source),protocol_sha256=sha(R/'src/research_plans/phase303.md'),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
        runtime=sys.executable,cpu_seconds_cap=3600,gpu_seconds_cap=3600,call_cap=900,persistent_bytes_cap=3000000000,
        cpu_memory_cap=16*1024**3,host_free_min=4000000000,planned_recovery_calls=873,
        immutable_phase302=True,truth_or_development_access=False))
    write(ROOT/'STAGES.json',{})


def gpu_conflict():
    text=subprocess.check_output(['ps','-eo','pid,args'],text=True)
    return [s for s in text.splitlines() if 'python' in s.lower() and 'phase303' not in s.lower()
        and 'bash -c' not in s and 'ssh ' not in s and any(k in s.lower() for k in ('train_','/train.py','infer_','evaluate_','execute_phase','worker_phase'))]


def main():
    initialize();lock=load(ROOT/'EXECUTION_LOCK.json')
    for name,h in lock['code'].items():assert sha(R/name)==h,name
    assert sha(R/'src/research_plans/phase303.md')==lock['protocol_sha256']
    stages=load(ROOT/'STAGES.json')
    if (ROOT/'EXIT_CODE').exists():print(json.dumps(dict(status='already_terminated',exit_code=(ROOT/'EXIT_CODE').read_text())));return
    for stage in ('cached','recover','diagnose','quantiles'):
        if stage in stages:continue
        if stage=='recover':
            active=gpu_conflict()
            if active:
                value=dict(status='waiting_for_gpu',stage=stage,conflicts=active,completed=list(stages))
                write(ROOT/'STATUS.json',value);print(json.dumps(value));return
        write(ROOT/'STATUS.json',dict(status='running',stage=stage,PID=os.getpid(),completed=list(stages)))
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',XFORMERS_DISABLED='1',HF_HUB_OFFLINE='1',
            TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),MPLCONFIGDIR=str(ROOT/'cache'),
            OMP_NUM_THREADS='4' if stage=='recover' else '1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        started=time.monotonic();peak=0;minimum=2**63;failure=None
        prior=sum(v['seconds'] for k,v in stages.items() if (k=='recover')==(stage=='recover'))
        with (ROOT/(stage+'.log')).open('xb') as log:
            p=subprocess.Popen([sys.executable,'-B',str(R/'src/tools/worker_phase303.py'),stage],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
            while p.poll() is None:
                status=Path('/proc')/str(p.pid)/'status'
                if status.exists():
                    lines=status.read_text().splitlines()
                    peak=max(peak,int(next((s.split()[1] for s in lines if s.startswith('VmHWM:')),'0'))*1024)
                available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
                minimum=min(minimum,available)
                if peak>16*1024**3:failure='CPU16GiB'
                if available<4000000000:failure='Host free4GB'
                if shutil.disk_usage(ROOT).free<4000000000:failure='Disk free4GB'
                if time.monotonic()-started+prior>3600:failure='Stage time budget1h'
                if sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())>3000000000:failure='New persistent3GB'
                if failure:p.terminate();break
                time.sleep(.25)
            code=p.wait()
        receipt=dict(status='completed' if code==0 and not failure else 'failed',exit_code=code,failure=failure,
            seconds=time.monotonic()-started,peak_cpu_bytes=peak,minimum_available_memory=minimum)
        write(ROOT/(stage+'_RESOURCE.json'),receipt)
        if receipt['status']!='completed':
            write(ROOT/'STATUS.json',dict(status='failed',stage=stage,receipt=receipt));(ROOT/'EXIT_CODE').write_text(str(code or 97));print(json.dumps(receipt));return
        stages[stage]=receipt;write(ROOT/'STAGES.json',stages)
    write(ROOT/'STATUS.json',dict(status='completed',stages=stages,neural_calls=sum(1 for _ in (ROOT/'CALLS.jsonl').open())))
    (ROOT/'EXIT_CODE').write_text('0');print((ROOT/'STATUS.json').read_text())


if __name__=='__main__':main()
