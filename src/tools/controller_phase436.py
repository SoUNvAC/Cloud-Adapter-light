"""Run nested label budgets, calibrate final checkpoints, report matched gains."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import traceback

R=Path('/home/scv/shared/phase436_20261103')

def main():
    env=dict(os.environ,**json.loads(Path('/home/scv/shared/phase429_20261102/launch_env.json').read_text()))
    env['PYTHONPATH']='/home/scv/shared/phase431_20261102/code:/home/scv/shared/phase429_20261102/dinov3'
    children={}
    for budget in ('10','25','50'):
        with (R/f'{budget}.log').open('x') as log:
            children[budget]=subprocess.Popen([sys.executable,'-u',str(R/'worker_phase436.py'),budget],env=env,stdout=log,stderr=subprocess.STDOUT)
    while True:
        states={k:dict(pid=p.pid,exit_code=p.poll()) for k,p in children.items()}
        (R/'STATUS.json').write_text(json.dumps(states,indent=2))
        if all(p.poll() is not None for p in children.values()):break
        time.sleep(15)
    assert all(p.returncode==0 for p in children.values()),states
    for budget in children:
        with (R/f'calibration_{budget}.log').open('x') as log:
            subprocess.run([sys.executable,'-u',str(R/'calibrate_phase436.py'),budget],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    ref=Path('/home/scv/shared/phase434_20261103')
    zero=json.loads((ref/'source_fit_bn.json').read_text())
    full=json.loads((ref/'final_fit_bn.json').read_text())
    source=json.loads((ref/'source.json').read_text())
    rows=[]
    for budget,count,report in [('0',0,zero)]+[(b,len(json.loads((R/'SUBSETS.json').read_text())[b]),json.loads((R/b/'calibration/final_fit_bn.json').read_text())) for b in children]+[('100',153,full)]:
        rows.append(dict(budget=budget,groups=count,actual_percent=100*count/153,mIoU=report['mIoU'],shadow_IoU=report['Shadow_IoU'],gain_vs_source=report['mIoU']-source['mIoU'],gain_vs_calibrated_zero=report['mIoU']-zero['mIoU'],full_gain_retained_percent=100*(report['mIoU']-zero['mIoU'])/(full['mIoU']-zero['mIoU'])))
    (R/'SUMMARY.json').write_text(json.dumps(rows,indent=2))
    lines=['# Phase436：目标标签比例实验','','Phase431源权重初始化；全参数4000步；各组统一153 fit RGB校准BN。0%与100%引用Phase434匹配端点。','','|监督比例|组数|实际比例|目标mIoU|云影IoU|相对源权重增益|相对0%同BN增益|全监督收益保留|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in rows:
        lines.append(f"|{x['budget']}%|{x['groups']}|{x['actual_percent']:.2f}%|{x['mIoU']:.4f}|{x['shadow_IoU']:.4f}|{x['gain_vs_source']:+.4f}|{x['gain_vs_calibrated_zero']:+.4f}|{x['full_gain_retained_percent']:.2f}%|")
    lines+=['','增益单位为百分点。固定32组development；单seed、单次嵌套抽样。主结果固定4000步，未经开发集择优。训练中的最佳/最终原始BN结果保存在各组COMPLETE.json。']
    (R/'RESULT.md').write_text('\n'.join(lines)+'\n')
    (R/'COMPLETE.json').write_text(json.dumps(dict(passed=True,training_steps=12000)))

if __name__=='__main__':
    try:main()
    except BaseException:
        (R/'FAILURE.json').write_text(json.dumps(dict(error=traceback.format_exc())))
        raise
