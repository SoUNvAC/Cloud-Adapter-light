"""One bounded CPU run with immutable inputs/code and no neural runtime import."""
from datetime import datetime,timezone,timedelta
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from phase307_common import *

def main():
    assert sys.executable=='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
    assert R==Path('/home/scv/Cloud-Adapter-light')
    assert not ROOT.exists(),'Preserve previous run, do not retry blindly'
    assert datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')=='20261025'
    available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
    assert available>=4000000000 and shutil.disk_usage(ROOT.parent).free>=4000000000
    ROOT.mkdir();(ROOT/'code').mkdir();(ROOT/'tmp').mkdir()
    files=sorted((R/'src/tools').glob('*phase307*.py'))+[R/'src/tools'/n for n in ('phase65d_recovery_math.py','auc_phase306.py')]
    for p in files:shutil.copyfile(p,ROOT/'code'/p.name)
    shutil.copyfile(R/'src/research_plans/phase307.md',ROOT/'code/phase307.md')
    write(ROOT/'EXECUTION_LOCK.json',dict(code={p.relative_to(R).as_posix():sha(p) for p in files},
        protocol_sha256=sha(R/'src/research_plans/phase307.md'),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
        start_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(),runtime=sys.executable,
        neural_calls=0,training_updates=0,download=0,cap_seconds=3600,cap_CPU_bytes=16*1024**3,
        cap_new_bytes_including_backups=1000000000,min_free_bytes=4000000000,threshold=.90,
        rule='Supervised offline LOGO D/K probes; RGB3/RGB6Q/MS6; no segmentation/network training',
        sampling_seed=30701,max_group_candidates=8192,max_fits=96,penalty=.001,threshold_Cloud=.50,
        optimizer=dict(method='L-BFGS-B',maxiter=500,gtol=1e-7,ftol=1e-12),BLAS_threads=4,
        bands=list(BANDS),indices=list(INDICES),q_bins=[[0,.5],[.5,.7],[.7,.9]],classifier_fits_cap=96,
        history_read_only=True))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
        TMPDIR=str(ROOT/'tmp'),CUDA_VISIBLE_DEVICES='')
    write(ROOT/'STATUS.json',dict(status='running',PID=os.getpid(),neural_calls=0))
    started=time.monotonic();peak=0;minimum=2**63;failure=None
    with (ROOT/'cpu.log').open('xb') as log:
        p=subprocess.Popen([sys.executable,'-B',str(R/'src/tools/analyze_phase307.py')],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
        while p.poll() is None:
            mem=Path('/proc')/str(p.pid)/'status'
            try:peak=max(peak,int(next((s.split()[1] for s in mem.read_text().splitlines() if s.startswith('VmHWM:')),'0'))*1024)
            except FileNotFoundError:pass
            available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
            minimum=min(minimum,available)
            size=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
            if time.monotonic()-started>3600:failure='CPU time1h'
            if peak>16*1024**3:failure='CPU memory16GiB'
            if available<4000000000:failure='Host free4GB'
            if shutil.disk_usage(ROOT).free<4000000000:failure='Disk free4GB'
            if size*5+10000000>1000000000:failure='Five-copy new persistent1GB'
            if failure:p.terminate();break
            time.sleep(.25)
        code=p.wait()
    fit_count=sum(1 for _ in (ROOT/'FIT_STARTS.jsonl').open()) if (ROOT/'FIT_STARTS.jsonl').exists() else 0
    assert fit_count<=96
    receipt=dict(classifier_fits=fit_count,status='completed' if code==0 and failure is None else 'failed',exit_code=code,failure=failure,
        seconds=time.monotonic()-started,peak_CPU_bytes=peak,minimum_available_bytes=minimum,
        neural_calls=0,training_updates=0,PID=p.pid)
    write(ROOT/'RESOURCE.json',receipt);write(ROOT/'STATUS.json',receipt);(ROOT/'EXIT_CODE').write_text(str(code or (97 if failure else 0)))
    print(__import__('json').dumps(receipt))

if __name__=='__main__':main()
