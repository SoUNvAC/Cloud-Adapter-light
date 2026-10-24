"""Detach one serialized controller without starting duplicate GPU workers."""
import os
from pathlib import Path
import subprocess
import sys
from control_phase304 import initialize
from phase304_common import R,ROOT,load,write

def main():
    initialize()
    marker=ROOT/'LAUNCH.json'
    if marker.exists():
        prior=load(marker)
        if (Path('/proc')/str(prior['PID'])).exists():print('already_running');return
        if (ROOT/'EXIT_CODE').exists():print('already_terminated');return
        assert (ROOT/'STATUS.json').exists() and load(ROOT/'STATUS.json')['status']=='waiting_for_gpu','Inspect previous launch failure before restart'
    with (ROOT/'controller.log').open('ab') as log:
        p=subprocess.Popen([sys.executable,'-B',str(R/'src/tools/control_phase304.py')],cwd=R,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(marker,dict(PID=p.pid));print('Phase304 controller PID '+str(p.pid))

if __name__=='__main__':main()
