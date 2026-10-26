"""Post-barrier CPU child supervision; RSS/host-free/disk budgets, no automatic scientific retry."""
import os
import shutil
import subprocess
import sys
import time
from phase315_common import *
def main():
    assert load(ROOT/'BOTH_SEALED.json')['status']=='sealed' and not (ROOT/'CPU_STAGES.json').exists()
    lock=load(ROOT/'EXECUTION_LOCK.json')
    for n,h in lock['code'].items():assert sha(R/n)==h,n
    for p in (R/'src/tools').glob('*phase315*.py'):
        dest=ROOT/'code'/p.name
        if dest.exists():assert sha(dest)==sha(p),p.name
        else:shutil.copyfile(p,dest)
    write(ROOT/'CPU_CODE_LOCK.json',dict(code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase315*.py') if p.is_file()},training_execution_lock_SHA=sha(ROOT/'EXECUTION_LOCK.json'),both_sealed_SHA=sha(ROOT/'BOTH_SEALED.json')))
    stages={};env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),MPLCONFIGDIR=str(ROOT/'cache'))
    for stage,script in [('analysis','analyze_phase315.py'),('audit','audit_phase315.py'),('admission','admission_phase315.py'),('report','report_phase315.py')]:
        started=time.monotonic();peak=0;minimum=2**63;failure=None
        with (ROOT/('CPU_'+stage+'.log')).open('xb') as log:
            p=subprocess.Popen([sys.executable,'-B',str(R/'src/tools'/script)],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
            while p.poll() is None:
                try:
                    path=Path('/proc')/str(p.pid)/'status'
                    if path.exists():peak=max(peak,int(next((l.split()[1] for l in path.read_text().splitlines() if l.startswith('VmHWM:')),'0'))*1024)
                except FileNotFoundError:pass
                available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024;minimum=min(minimum,available)
                size=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
                if peak>16*1024**3 or available<4000000000 or shutil.disk_usage(ROOT).free<4000000000 or size*4+100000000>6000000000:failure='CPU16GiB/hostfree4GB/diskfree4GB/allcopies6GB'
                if failure:p.terminate();break
                time.sleep(.5)
            code=p.wait()
        receipt=dict(status='passed' if code==0 and not failure else 'failed',seconds=time.monotonic()-started,exit_code=code,failure=failure,peak_cpu_bytes=peak,minimum_available_memory=minimum);stages[stage]=receipt;write(ROOT/'CPU_STAGES.json',stages)
        if receipt['status']!='passed':write(ROOT/'STATUS.json',dict(status='failed',stage='CPU_'+stage,receipt=receipt));raise RuntimeError('CPU stage failed; preserve results; no retry: '+stage)
    print('All Phase315 CPU stages passed')
if __name__=='__main__':
    if '--launch' in sys.argv:
        assert (ROOT/'BOTH_SEALED.json').exists() and not (ROOT/'CPU_LAUNCH.json').exists()
        with (ROOT/'CPU_controller.log').open('xb') as log:p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve())],cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(ROOT/'CPU_LAUNCH.json',dict(PID=p.pid,epoch=time.time()));print('CPU controller PID',p.pid)
    else:main()
