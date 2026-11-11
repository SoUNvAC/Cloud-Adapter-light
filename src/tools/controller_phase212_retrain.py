"""One non-restarting sequential controller for the two user-authorized fits."""
from pathlib import Path
import json
import subprocess
import time

BASE=Path('/home/scv/shared/phase212_retrain_20261020_run02')
PY='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
def dump(p,x):
    t=Path(str(p)+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)

def main():
    start=time.monotonic()
    for target in ['l8','sparcs']:
        assert not (BASE/target).exists(),'Existing target artifacts; do not restart blindly'
        with (BASE/(target+'.console.log')).open('x') as o,(BASE/(target+'.error.log')).open('x') as e:
            p=subprocess.Popen([PY,'-B',str(BASE/'code/train_phase212_head.py'),'--target',target],cwd='/home/scv/Cloud-Adapter-light/src',stdout=o,stderr=e,stdin=subprocess.DEVNULL)
            dump(BASE/'CONTROLLER_STATUS.json',{'stage':'training','target':target,'worker_PID':p.pid,'started_epoch':time.time()})
            code=p.wait()
        if code:
            dump(BASE/'CONTROLLER_STATUS.json',{'stage':'failed','target':target,'worker_exit':code,'elapsed_seconds':time.monotonic()-start});(BASE/'EXIT_CODE').write_text(str(code)+'\n');return
    receipts={t:json.loads((BASE/t/'COMPLETED.json').read_text()) for t in ['l8','sparcs']}
    calls=sum(sum(r['calls'].values()) for r in receipts.values())
    assert calls==15784
    dump(BASE/'RETRAINING_COMPLETE.json',{'stage':'both_heads_retrained','heads':receipts,'calls':calls,'elapsed_seconds':time.monotonic()-start,'prior_failed_calls':0,'prior_failed_seconds':1.3337932010181248,'cumulative_execution_seconds':time.monotonic()-start+1.3337932010181248,'interpolation_evaluation':'not_started; new endpoint and supporting gates next'})
    dump(BASE/'CONTROLLER_STATUS.json',{'stage':'both_heads_retrained','calls':calls,'elapsed_seconds':time.monotonic()-start});(BASE/'EXIT_CODE').write_text('0\n')

if __name__=='__main__':main()
