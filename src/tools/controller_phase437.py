"""Run nested label budgets, calibrate final checkpoints, report matched gains."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import traceback

R=Path('/home/scv/shared/phase437_20261103')

def main():
    env=dict(os.environ,**json.loads(Path('/home/scv/shared/phase429_20261102/launch_env.json').read_text()))
    env['PYTHONPATH']='/home/scv/shared/phase431_20261102/code:/home/scv/shared/phase429_20261102/dinov3'
    children={}
    for budget in ('43602','43603'):
        with (R/f'{budget}.log').open('x') as log:
            children[budget]=subprocess.Popen([sys.executable,'-u',str(R/'worker_phase437.py'),budget],env=env,stdout=log,stderr=subprocess.STDOUT)
    while True:
        states={k:dict(pid=p.pid,exit_code=p.poll()) for k,p in children.items()}
        (R/'STATUS.json').write_text(json.dumps(states,indent=2))
        if all(p.poll() is not None for p in children.values()):break
        time.sleep(15)
    assert all(p.returncode==0 for p in children.values()),states
    for budget in children:
        with (R/f'calibration_{budget}.log').open('x') as log:
            subprocess.run([sys.executable,'-u',str(R/'calibrate_phase437.py'),budget],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    ref=Path('/home/scv/shared/phase434_20261103')
    zero=json.loads((ref/'source_fit_bn.json').read_text())
    full=json.loads((ref/'final_fit_bn.json').read_text())
    source=json.loads((ref/'source.json').read_text())
    import statistics
    reports={'43601':json.loads(Path('/home/scv/shared/phase436_20261103/25/calibration/final_fit_bn.json').read_text())}
    reports.update({b:json.loads((R/b/'calibration/final_fit_bn.json').read_text()) for b in children})
    rows=[]
    for seed,report in reports.items():
        rows.append(dict(sampling_seed=seed,groups=38,mIoU=report['mIoU'],shadow_IoU=report['Shadow_IoU'],gain_vs_source=report['mIoU']-source['mIoU'],gain_vs_calibrated_zero=report['mIoU']-zero['mIoU'],full_gain_retained_percent=100*(report['mIoU']-zero['mIoU'])/(full['mIoU']-zero['mIoU'])))
    stats={k:dict(mean=statistics.mean(x[k] for x in rows),sample_std=statistics.stdev(x[k] for x in rows),min=min(x[k] for x in rows),max=max(x[k] for x in rows)) for k in ('mIoU','shadow_IoU','gain_vs_calibrated_zero','full_gain_retained_percent')}
    summary=dict(rows=rows,statistics=stats,training_seed=43301,zero_mIoU=zero['mIoU'],full_mIoU=full['mIoU'])
    (R/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
    lines=['# Phase437: 25% supervision across sampling seeds','','38/153 fit groups; same source checkpoint, training seed 43301, 4000 steps and 153 fit RGB BN calibration. Seed 43601 reused from Phase436.','','|Sampling seed|mIoU|Shadow IoU|Gain vs calibrated 0%|Full supervision gain retained|','|---|---:|---:|---:|---:|']
    for x in rows:
        lines.append(f"|{x['sampling_seed']}|{x['mIoU']:.4f}|{x['shadow_IoU']:.4f}|{x['gain_vs_calibrated_zero']:+.4f}|{x['full_gain_retained_percent']:.2f}%|")
    lines+=['',f"mIoU mean +/- sample SD: {stats['mIoU']['mean']:.4f} +/- {stats['mIoU']['sample_std']:.4f}",'','Sampling variation only; fixed training seed. Fixed exposed 32-group development evaluation. Final 4000-step checkpoints used without selection.']
    (R/'RESULT.md').write_text('\n'.join(lines)+'\n')
    (R/'COMPLETE.json').write_text(json.dumps(dict(passed=True,new_training_steps=8000,sampling_seeds=3)))

if __name__=='__main__':
    try:main()
    except BaseException:
        (R/'FAILURE.json').write_text(json.dumps(dict(error=traceback.format_exc())))
        raise
