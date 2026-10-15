"""One detached launch after read-only identity/resource gates; no waiting loop."""
import os,sys,subprocess,json,time
from phase123_common import *
def main():
    setup();assert not ROOT.exists(),'existing123 root: never duplicate launch'
    from install_phase111_runtime import host_free
    free=host_free();assert free>=4000000000
    import numpy as np
    members=load(REPO/'src/research_results/phase122_20261015/MEMBER_INPUT_LOCK.json')
    assert len(members['fit'])==100 and len(members['calibration'])==53
    assert not ({g['group_id'] for g in members['fit']}&{g['group_id'] for g in members['calibration']})
    path=ROOT.with_suffix('.console.log');receipt=ROOT.with_suffix('.launch.json');assert not receipt.exists()
    with path.open('xb') as log:
        p=subprocess.Popen([sys.executable,'-B',str(REPO/'src/tools/run_phase123.py')],cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    record=dict(PID=p.pid,launched_epoch=time.time(),source_lock_sha256=sha(LOCK),host_free_bytes=free,console=str(path),interval_hours=1)
    write(receipt,record);print(json.dumps(record))
if __name__=='__main__':main()
