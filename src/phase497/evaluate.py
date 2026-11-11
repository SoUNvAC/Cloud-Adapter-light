"""Reload final and validation-selected weights, evaluate unchanged full source val."""
import json,hashlib
from pathlib import Path
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.utils import register_all_modules
R=Path('/home/scv/shared/phase497_20261110')
def main():
    register_all_modules()
    rows=[]
    for v in ['texture']:
        assert json.loads((R/v/'COMPLETE.json').read_text())['steps']==40000
        final=R/v/'iter_40000.pth'
        best=list((R/v).glob('best_mIoU_iter_*.pth'))
        assert len(best)==1 and final.exists()
        for kind,ckpt in [('final',final),('best',best[0])]:
            cfg=Config.fromfile(str(Path(__file__).with_name('config.py')))
            cfg.model.backbone.variant=v
            cfg.work_dir=str(R/f'eval_{v}_{kind}')
            cfg.load_from=str(ckpt)
            cfg.test_dataloader=cfg.val_dataloader
            cfg.test_evaluator=dict(type='ExactIoU497',iou_metrics=['mIoU'])
            cfg.custom_hooks=[]
            runner=Runner.from_cfg(cfg)
            assert len(runner.test_dataloader.dataset)==535
            metrics=runner.test()
            # IoUMetric console table is retained; use its cumulative confusion to
            # report exact per-class IoU by additionally requesting class metrics below.
            row=dict(variant=v,checkpoint=kind,path=str(ckpt),sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),metrics=metrics)
            rows.append(row)
            (R/'RESULT.json').write_text(json.dumps(rows,indent=2))
            del runner
            import gc,torch
            gc.collect();torch.cuda.empty_cache()
    lines=['# Phase497','', 'CloudSEN验证535张；目标监督0%；两组官方DINOv3初始化，任务模块40K重训。','', '| 网络 | 检查点 | 地表IoU | 云IoU | 云影IoU | mIoU |','|---|---|---|---|---|---|']
    for r in rows:
        m=r['metrics']
        lines.append(f"| {r['variant']} | {r['checkpoint']} | {m['IoU_surface']:.4f} | {m['IoU_cloud']:.4f} | {m['IoU_shadow']:.4f} | {m['mIoU_exact']:.4f} |")
    (R/'RESULT.md').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__':main()
