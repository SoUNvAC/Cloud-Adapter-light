"""CPU-only serial pipeline, total-time/RAM/disk/persistent hard limits, no retries."""
import fcntl
import shutil
import subprocess
import sys
import traceback
from datetime import datetime,timezone,timedelta
from phase312_common import *

def initialize():
    assert R==Path('/home/scv/Cloud-Adapter-light') and sys.executable=='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
    assert not ROOT.exists(),'Inspect any existing experiment; no duplicate initialization'
    assert datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')=='20261025'
    assert sha(R/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth')==SOURCE
    assert sha(PL/'EXECUTION_LOCK.json')=='35965ffd9733c9a5ed98039ac3f2ce6351d32a6dbfcc41b4abb8fc6f23df9182'
    assert sha(PL/'PSEUDO_LOCK.json')=='7b32234108d3c2031962bd908fa8b16cf5ba8c1838ab2652ba40fd1c49b6add1'
    for n in ('FIT_IMAGES.json','EVAL_IMAGES.json'):assert sha(PL/n)==sha(OLD/n),n
    fit=load(PL/'FIT_IMAGES.json')['rows'];dev=load(PL/'EVAL_IMAGES.json')['rows']
    assert len(fit)==100 and len(dev)==32 and len({r['product'] for r in fit})==100 and len({r['product'] for r in dev})==32
    assert not ({r['product'] for r in fit}&{r['product'] for r in dev}) and not ({r['group_id'] for r in fit}&{r['group_id'] for r in dev})
    ROOT.mkdir()
    for name in ('models','predictions','readouts','code','tmp'):(ROOT/name).mkdir()
    files=list((R/'src/tools').glob('*phase312*.py'))+[R/'src/tools'/n for n in ('phase65d_recovery_math.py','analyze_phase304.py','transfer_phase302.py','auc_phase306.py')]
    for p in files:shutil.copyfile(p,ROOT/'code'/p.name)
    shutil.copyfile(R/'src/research_plans/phase312.md',ROOT/'code/phase312.md')
    for n in ('FIT_IMAGES.json','EVAL_IMAGES.json'):shutil.copyfile(PL/n,ROOT/n)
    inputs=[PL/n for n in ('EXECUTION_LOCK.json','ENDPOINT_LOCK.json','EVALUATION_LOCK.json','PSEUDO_LOCK.json','FIT_IMAGES.json','EVAL_IMAGES.json')]
    write(ROOT/'EXECUTION_LOCK.json',dict(start_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),runtime=sys.executable,code={p.relative_to(R).as_posix():sha(p) for p in files},protocol_SHA=sha(R/'src/research_plans/phase312.md'),inputs={str(p):sha(p) for p in inputs},fit_images_SHA=sha(ROOT/'FIT_IMAGES.json'),eval_images_SHA=sha(ROOT/'EVAL_IMAGES.json'),fit_dev_product_and_group_disjoint=True,Source_SHA=SOURCE,neural_training_and_forwards=0,logistic_fits_cap=3,CPU_seconds_cap=7200,CPU_bytes_cap=16*1024**3,persistent_all_hosts_cap=2000000000,min_free_memory_disk=4000000000,compute_location='gzs CPU only',no_scheduled_checks=True))
    write(ROOT/'STAGES.json',{})

def main():
    initialize();started=time.monotonic()
    with (ROOT/'CONTROLLER_LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);stages={};execution=load(ROOT/'EXECUTION_LOCK.json')
        for stage in ('fit','predict','readout','analyze'):
            for n,h in execution['code'].items():assert sha(R/n)==h,n
            for n,h in execution['inputs'].items():assert sha(n)==h,n
            if stage=='analyze':assert load(ROOT/'READOUT_LOCK.json')['status']=='sealed'
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='',TMPDIR=str(ROOT/'tmp'))
            write(ROOT/'STATUS.json',dict(status='running',stage=stage,PID=os.getpid(),completed=list(stages)));peak=0;minimum=2**63;failure=None;t=time.monotonic()
            script='analyze_phase312.py' if stage=='analyze' else 'worker_phase312.py'
            with (ROOT/(stage+'.log')).open('xb') as log:
                p=subprocess.Popen([sys.executable,'-B',str(R/'src/tools'/script),stage],env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT)
                while p.poll() is None:
                    status=Path('/proc')/str(p.pid)/'status'
                    try:peak=max(peak,int(next((s.split()[1] for s in status.read_text().splitlines() if s.startswith('VmHWM:')),'0'))*1024)
                    except FileNotFoundError:pass
                    free=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024;minimum=min(minimum,free)
                    size=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
                    if peak>16*1024**3:failure='CPU16GiB'
                    if free<4000000000 or shutil.disk_usage(ROOT).free<4000000000:failure='Host free memory/disk4GB'
                    if time.monotonic()-started>7200:failure='Total pipeline2h'
                    if size*4+64000000>2000000000:failure='Two hosts including raw/archive/summary2GB'
                    if failure:p.terminate();break
                    time.sleep(.5)
                code=p.wait()
            receipt=dict(status='completed' if code==0 and not failure else 'failed',exit_code=code,failure=failure,seconds=time.monotonic()-t,peak_cpu_bytes=peak,minimum_available_memory=minimum,PID=p.pid)
            write(ROOT/(stage+'_RESOURCE.json'),receipt)
            if receipt['status']!='completed':
                append('FAILURES.jsonl',dict(stage=stage,receipt=receipt));write(ROOT/'STATUS.json',dict(status='failed',stage=stage));(ROOT/'EXIT_CODE').write_text(str(code or 97));return
            stages[stage]=receipt;write(ROOT/'STAGES.json',stages)
        fits=load(ROOT/'FIT_LOCK.json');write(ROOT/'STATUS.json',dict(status='completed',scientific_status='complete' if len(fits['successful_models'])==3 else 'partial_failed_models',stages=stages,total_seconds=time.monotonic()-started,neural_forwards=0,logistic_fit_attempts=fits['fit_attempts'],successful_models=fits['successful_models']));(ROOT/'EXIT_CODE').write_text('0')
        print((ROOT/'STATUS.json').read_text())

if __name__=='__main__':
    try:main()
    except BaseException:
        if ROOT.exists():append('FAILURES.jsonl',dict(stage='controller',traceback=traceback.format_exc()))
        raise
