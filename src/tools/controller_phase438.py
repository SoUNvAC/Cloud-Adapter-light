"""Five parallel controlled training runs and complete calibrated learning curves."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import traceback

R=Path('/home/scv/shared/phase438_20261103')

def wait_all(children,stage):
    while True:
        states={k:dict(pid=p.pid,exit_code=p.poll()) for k,p in children.items()}
        (R/'STATUS.json').write_text(json.dumps(dict(stage=stage,workers=states),indent=2))
        if all(p.poll() is not None for p in children.values()):break
        time.sleep(15)
    assert all(p.returncode==0 for p in children.values()),states

def main():
    env=dict(os.environ,**json.loads(Path('/home/scv/shared/phase429_20261102/launch_env.json').read_text()))
    env['PYTHONPATH']='/home/scv/shared/phase431_20261102/code:/home/scv/shared/phase429_20261102/dinov3'
    children={}
    for arm in ('' if '--diagnostics-only' in sys.argv else 'ABCDE'):
        with (R/f'{arm}.log').open('a') as log:
            children[arm]=subprocess.Popen([sys.executable,'-u',str(R/'worker_phase438.py'),arm],env=env,stdout=log,stderr=subprocess.STDOUT)
    if children:wait_all(children,'training')
    traces=[(R/a/'INPUT_TRACE.jsonl').read_bytes() for a in 'ABCDE']
    paired=all(t==traces[0] for t in traces)
    (R/'INPUT_PAIRING.json').write_text(json.dumps(dict(target_input_traces_equal=paired)))
    assert paired, 'Target input pairing failed'
    children={}
    for arm in ['SOURCE']+list('ABCDE'):
        with (R/f'eval_{arm}.log').open('a') as log:
            children[arm]=subprocess.Popen([sys.executable,'-u',str(R/'evaluate_arm_phase438.py'),arm],env=env,stdout=log,stderr=subprocess.STDOUT)
    wait_all(children,'diagnostics')
    rows=[];curves=[]
    for arm in ['SOURCE']+list('ABCDE'):
        for step in ([0] if arm=='SOURCE' else range(500,4001,500)):
            root=R/arm/'diagnostics'/str(step)
            target=json.loads((root/'final_fit_bn.json').read_text())
            raw=json.loads((root/'raw_target.json').read_text())
            train=json.loads((root/'target_train.json').read_text())
            source=json.loads((root/'source_val.json').read_text())
            row=dict(arm=arm,step=step,mIoU=target['mIoU'],surface=target['IoU'][0],cloud=target['IoU'][1],shadow=target['IoU'][2],source_mIoU=source['mIoU'],source_IoU=source['IoU'],train_mIoU=train['mIoU'],train_IoU=train['IoU'],raw_mIoU=raw['mIoU'])
            curves.append(row)
            if step in (0,4000):rows.append(row)
    (R/'CURVES.json').write_text(json.dumps(curves,indent=2))
    (R/'SUMMARY.json').write_text(json.dumps(rows,indent=2))
    lines=['# Phase438: 10% supervision damage diagnosis','','All final results use fixed 4000-step checkpoints. Source validation uses its own fixed source-train RGB BN calibration.','','|Arm|Target mIoU|Surface|Cloud|Shadow|Train mIoU|Source mIoU|Raw target mIoU|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in rows:lines.append(f"|{x['arm']}|{x['mIoU']:.4f}|{x['surface']:.4f}|{x['cloud']:.4f}|{x['shadow']:.4f}|{x['train_mIoU']:.4f}|{x['source_mIoU']:.4f}|{x['raw_mIoU']:.4f}|")
    (R/'RESULT.md').write_text('\n'.join(lines)+'\n')
    if (R/'FAILURE.json').exists():(R/'FAILURE.json').rename(R/'FAILURE_resolved.json')
    (R/'COMPLETE.json').write_text(json.dumps(dict(passed=True,training_steps=20000,target_input_traces_equal=paired)))

if __name__=='__main__':
    try:main()
    except BaseException:
        (R/'FAILURE.json').write_text(json.dumps(dict(error=traceback.format_exc())))
        raise
