from pathlib import Path
import json
import os
import subprocess
import sys
import time
import traceback

R=Path('/home/scv/shared/phase440_20261103')

def main():
    env=dict(os.environ,**json.loads(Path('/home/scv/shared/phase429_20261102/launch_env.json').read_text()))
    env['PYTHONPATH']='/home/scv/shared/phase431_20261102/code:/home/scv/shared/phase429_20261102/dinov3'
    children={}
    for arm in ('source','adapted'):
        with (R/f'{arm}.log').open('x') as log:
            children[arm]=subprocess.Popen([sys.executable,'-u',str(R/'worker_phase440.py'),arm],env=env,stdout=log,stderr=subprocess.STDOUT)
    while True:
        states={k:dict(pid=p.pid,exit_code=p.poll()) for k,p in children.items()}
        (R/'STATUS.json').write_text(json.dumps(states,indent=2))
        if all(p.poll() is not None for p in children.values()):break
        time.sleep(10)
    assert all(p.returncode==0 for p in children.values()),states
    subprocess.run([sys.executable,'-u',str(R/'report_phase440.py')],check=True,env=env)

if __name__=='__main__':
    try:main()
    except BaseException:
        (R/'FAILURE.json').write_text(json.dumps(dict(error=traceback.format_exc())))
        raise
