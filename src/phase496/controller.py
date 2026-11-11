"""Preflight both fresh networks, then parallel 40K and checkpoint evaluation."""
import json, os, subprocess, sys, time, traceback, hashlib
from pathlib import Path
R=Path('/home/scv/shared/phase496_20261110')
def status(state,**kw):
    (R/'STATUS.json').write_text(json.dumps(dict(state=state,time=time.time(),**kw),indent=2))
def start(module,variant,work):
    log=(R/(work+'.log')).open('x')
    return subprocess.Popen([sys.executable,'-u','-m',module,variant,'--work-dir',str(R/work)],stdout=log,stderr=subprocess.STDOUT)
def main():
    for v in ['control','texture']:
        status('preflight',variant=v)
        p=start('phase496.preflight',v,'preflight_'+v)
        assert p.wait()==0, 'preflight '+v
    # Common task layers and VFM must start identically; extra module RNG is isolated.
    a,b=[json.loads((R/('preflight_'+v)/'INITIAL.json').read_text()) for v in ['control','texture']]
    assert a['common_sha']==b['common_sha'] and a['frozen_sha']==b['frozen_sha']
    (R/'PREFLIGHT_ACCEPTED.json').write_text(json.dumps(dict(passed=True,common_sha=a['common_sha'],frozen_sha=a['frozen_sha'])))
    start_time=time.time()
    ps={v:start('phase496.train',v,v) for v in ['control','texture']}
    status('training',pids={v:p.pid for v,p in ps.items()})
    while any(p.poll() is None for p in ps.values()):
        for v,p in ps.items():
            assert p.poll() in (None,0), f'{v} failed with {p.returncode}; other run retained'
        time.sleep(30)
    assert all(p.returncode==0 for p in ps.values())
    status('evaluating',training_wall_seconds=time.time()-start_time)
    from .evaluate import main as evaluate
    evaluate()
    status('complete',total_wall_seconds=time.time()-start_time)
if __name__=='__main__':
    try:main()
    except BaseException:
        (R/'FAILURE.txt').write_text(traceback.format_exc());status('failed');raise
