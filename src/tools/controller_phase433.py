"""Parallel jobs and an automatically generated four-model comparison table."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

R=Path('/home/scv/shared/phase433_20261103')


def main():
    children={}
    for phase in (429,430,431,432):
        parent=Path(f'/home/scv/shared/phase{phase}_20261102')
        base=Path('/home/scv/shared/phase429_20261102')
        env=dict(os.environ,**json.loads((base/'launch_env.json').read_text()))
        env['PYTHONPATH']=str(parent/'code')+':'+str(parent)+':'+str(base/'dinov3')
        with (R/f'{phase}.log').open('x') as stream:
            children[phase]=subprocess.Popen([sys.executable,'-u',str(R/'worker_phase433.py'),str(phase)],env=env,stdout=stream,stderr=subprocess.STDOUT)
    while True:
        states={str(k):dict(pid=p.pid,exit_code=p.poll()) for k,p in children.items()}
        (R/'STATUS.json').write_text(json.dumps(states,indent=2))
        if all(p.poll() is not None for p in children.values()):break
        time.sleep(15)
    assert all(p.returncode==0 for p in children.values()),states
    traces=[(R/str(k)/'INPUT_TRACE.jsonl').read_bytes() for k in children]
    paired=all(t==traces[0] for t in traces)
    rows=[]
    for phase in children:
        d=json.loads((R/str(phase)/'COMPLETE.json').read_text())
        parent=Path(f'/home/scv/shared/phase{phase}_20261102/catalogue_best')
        prior=json.loads((parent/'COMPLETE.json').read_text())['summary']
        prior=next(x for x in prior if x['variant']==('original_msre' if phase==430 else 'fused'))
        rows.append(dict(phase=phase,before=prior['mIoU'],best=d['best']['mIoU'],final=d['final']['mIoU'],
            gain=d['best']['mIoU']-prior['mIoU'],best_step=d['best_step'],shadow_IoU=d['best']['Shadow_IoU'],
            macro_AP=d['best']['macro_shadow_AP'],pooled_AP=d['best']['pooled_shadow_AP'],seconds=d['seconds']))
    (R/'SUMMARY.json').write_text(json.dumps(dict(rows=rows,input_traces_equal=paired),indent=2))
    lines=['# Phase433：Catalogue全网络微调','','153 fit / 固定32 development，4000步，四模型均解冻主干。','',
           '|模型|适配前mIoU|最佳mIoU|最终mIoU|最佳增量|最佳步数|最佳云影IoU|最佳宏AP|',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in rows:
        lines.append(f"|{x['phase']}|{x['before']:.4f}|{x['best']:.4f}|{x['final']:.4f}|{x['gain']:+.4f}|{x['best_step']}|{x['shadow_IoU']:.4f}|{x['macro_AP']:.4f}|")
    lines+=['',f'抽查输入批次完全配对：{paired}。','单seed、已暴露开发集；充分监督参考，不能单独认定为标注噪声天花板。']
    (R/'RESULT.md').write_text('\n'.join(lines)+'\n')
    (R/'COMPLETE.json').write_text(json.dumps(dict(passed=True,training_steps=16000,input_traces_equal=paired)))


if __name__=='__main__':main()
