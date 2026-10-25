"""Two bounded documentary stages; preserve original runtime and every prior artifact."""
import os
import sys
import time
import shutil
import subprocess
from datetime import datetime,timezone,timedelta
from phase309_common import *
def tree_peak(pid):
    processes={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            rows=p.joinpath('status').read_text().splitlines()
            values={r.split(':',1)[0]:r.split(':',1)[1].strip() for r in rows if ':' in r}
            processes[int(p.name)]=(int(values['PPid']),int(values.get('VmHWM','0 kB').split()[0])*1024)
        except (OSError,ValueError,KeyError):pass
    found={pid}
    while True:
        new=found|{p for p,(parent,_) in processes.items() if parent in found}
        if new==found:break
        found=new
    return sum(processes.get(p,(0,0))[1] for p in found)
def main():
    stage=sys.argv[1];assert stage in ('intake','analysis')
    assert sys.executable=='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
    assert R==Path('/home/scv/Cloud-Adapter-light')
    if stage=='intake':
        assert not ROOT.exists() and datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')=='20261025'
        ROOT.mkdir();(ROOT/'tmp').mkdir();(ROOT/'code').mkdir()
    else:assert load(ROOT/'STATUS.json')['status']=='awaiting_document_review'
    assert not (ROOT/(stage+'_LOCK.json')).exists(),'No blind repeat'
    files=sorted((R/'src/tools').glob('*phase309*.py'))
    for p in files:shutil.copyfile(p,ROOT/'code'/(stage+'_'+p.name))
    if stage=='intake':shutil.copyfile(R/'src/research_plans/phase309.md',ROOT/'code/phase309.md')
    write(ROOT/(stage+'_LOCK.json'),dict(code={p.relative_to(R).as_posix():sha(p) for p in files},protocol_sha256=sha(R/'src/research_plans/phase309.md'),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),start_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(),runtime=sys.executable,cap_CPU_bytes=8*1024**3,cap_total_seconds=3600,cap_new_persistent_bytes=500000000,max_download_bytes=20000000,max_web_documents=10,fits=0,training_updates=0,neural_calls=0))
    previous=load(ROOT/'RESOURCE.json')['stages'] if (ROOT/'RESOURCE.json').exists() else []
    started=time.monotonic();peak=0;minimum=2**63;failure=None
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT/'tmp'),PIP_DISABLE_PIP_VERSION_CHECK='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
    write(ROOT/'STATUS.json',dict(status='running',stage=stage,PID=os.getpid()))
    with (ROOT/(stage+'.log')).open('xb') as log:
        p=subprocess.Popen([sys.executable,'-B',str(R/'src/tools'/('intake_phase309.py' if stage=='intake' else 'audit_phase309.py'))],env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while p.poll() is None:
            peak=max(peak,tree_peak(p.pid))
            available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024;minimum=min(minimum,available)
            size=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())
            if sum(r['seconds'] for r in previous)+time.monotonic()-started>3600:failure='total execution1h'
            if peak>8*1024**3:failure='CPU tree8GiB'
            if available<4000000000 or shutil.disk_usage(ROOT).free<4000000000:failure='host free memory/disk4GB'
            if size*6+20000000>500000000:failure='persistent six-copy500MB'
            if failure:os.killpg(p.pid,15);break
            time.sleep(.2)
        code=p.wait()
    stages=previous+[dict(stage=stage,seconds=time.monotonic()-started,peak_CPU_tree_bytes=peak,minimum_available_bytes=minimum,exit_code=code,failure=failure)]
    status='failed' if code or failure else 'awaiting_document_review' if stage=='intake' else 'completed'
    receipt=dict(status=status,stages=stages,total_seconds=sum(s['seconds'] for s in stages),peak_CPU_tree_bytes=max(s['peak_CPU_tree_bytes'] for s in stages),minimum_available_bytes=min(s['minimum_available_bytes'] for s in stages),fits=0,training_updates=0,neural_calls=0)
    write(ROOT/'RESOURCE.json',receipt);write(ROOT/'STATUS.json',receipt);(ROOT/'EXIT_CODE').write_text(str(code or (97 if failure else 0)));print(json.dumps(receipt))
if __name__=='__main__':main()
