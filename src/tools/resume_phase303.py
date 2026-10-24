"""Serialize this controller and wait for other GPU controllers across job gaps."""
import fcntl
import json
import subprocess
from phase303_common import ROOT,load,write


def main():
    assert ROOT.exists()
    with (ROOT/'CONTROLLER_LOCK').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print(json.dumps(dict(status='already_running')));return
        stages=load(ROOT/'STAGES.json')
        if 'recover' not in stages:
            active=[line for line in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
                if 'python' in line.lower() and 'phase303' not in line.lower() and 'bash -c' not in line and 'ssh ' not in line
                and any(s in line.lower() for s in ('controller_phase','control_phase','train_','/train.py','infer_','evaluate_','execute_phase','worker_phase'))]
            if active:
                state=dict(status='waiting_for_gpu',stage='recover',completed=list(stages),conflicts=active)
                write(ROOT/'STATUS.json',state);print(json.dumps(state));return
        from control_phase303 import main as run
        run()


if __name__=='__main__':main()
