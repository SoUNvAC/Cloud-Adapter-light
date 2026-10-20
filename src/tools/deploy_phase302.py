"""Launch the exact committed Phase302 GPU pipeline once, after CPU admission."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from prepare_phase302 import R, ROOT, load, sha, write


def main():
    assert load(ROOT/'SOURCE_ADMISSION.json')['status']=='passed'
    assert not (ROOT/'DISPATCH.json').exists(), 'No duplicate launch'
    assert Path(sys.executable).resolve()==Path('/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python').resolve()
    import torch
    assert torch.cuda.is_available()
    free,total=torch.cuda.mem_get_info();assert free>=20*1024**3
    # Allocated GPU process memory must be idle before starting this one job.
    memory=torch.cuda.memory_allocated();assert memory==0
    output=subprocess.check_output(['ps','-eo','pid,args'],text=True)
    active=[s for s in output.splitlines() if 'python' in s and any(k in s for k in
        ('execute_phase300.py','run_phase80_training.py','run_phase85_training.py','tools/train.py','execute_phase302.py'))
        and 'bash -c' not in s and 'ssh ' not in s]
    assert not active, active
    files=sorted(p for p in (R/'src/tools').glob('*phase302*') if p.is_file())
    code={str(p.relative_to(R)):sha(p) for p in files}
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
    for name,digest in code.items():
        assert __import__('hashlib').sha256(subprocess.check_output(['git','show',commit+':'+name],cwd=R)).hexdigest()==digest,name
    write(ROOT/'EXECUTION_LOCK.json',dict(code=code,git_commit=commit,
        admission_sha256=sha(ROOT/'SOURCE_ADMISSION.json'),protocol_sha256=sha(R/'src/research_plans/phase302.md'),
        runtime=sys.executable,seed=30201,planned_main_calls=2492,cap_calls=4000,cap_gpu_seconds=14400,
        policy='single fixed endpoint; no labels/development access in train worker; no automatic retries'))
    env=dict(os.environ,PHASE302_ROOT=str(ROOT),PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT/'tmp'),
        XDG_CACHE_HOME=str(ROOT/'cache'),MPLCONFIGDIR=str(ROOT/'cache'),HF_HUB_OFFLINE='1',XFORMERS_DISABLED='1')
    with (ROOT/'controller.log').open('xb') as f:
        process=subprocess.Popen([sys.executable,'-B',str(R/'src/tools/execute_phase302.py'),'controller'],
            cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    write(ROOT/'DISPATCH.json',dict(PID=process.pid,epoch=time.time(),git_commit=commit))
    print(json.dumps(dict(PID=process.pid,root=str(ROOT),git_commit=commit)))


if __name__=='__main__': main()
