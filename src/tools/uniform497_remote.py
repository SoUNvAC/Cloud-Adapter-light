from pathlib import Path
import json,sys
R=Path('/home/scv/shared/phase497_20261110')
assert json.loads((R/'STATUS.json').read_text())['state']=='complete'
sys.path[:0]=[str(R/'code'),'/home/scv/Cloud-Adapter-light/src']
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.utils import register_all_modules
register_all_modules()
cfg=Config.fromfile(str(R/'code/phase497/config.py'))
cfg.model.backbone.update(variant='texture',repo='/home/scv/shared/phase429_20261102/dinov3',weights='/home/scv/shared/phase429_20261102/dinov3_splus_native.pth')
cfg.work_dir=str(R/'eval_uniform_final')
cfg.load_from=str(R/'texture/iter_40000.pth')
cfg.test_dataloader=cfg.val_dataloader
cfg.test_dataloader.dataset.data_root='/home/scv/Cloud-Adapter-light/src/data/cloudsen12_high_l1c'
cfg.test_evaluator=dict(type='ExactIoU497',iou_metrics=['mIoU'])
cfg.custom_hooks=[]
runner=Runner.from_cfg(cfg)
assert len(runner.test_dataloader.dataset)==535
runner.model.backbone.texture_adapter.uniform_reference=True
confidence=[]
def record(module,args,out):
    confidence.append((args[0].shape[0],float(module.last_confidence)))
runner.model.backbone.texture_adapter.register_forward_hook(record)
metrics=runner.test()
assert runner.model.backbone.texture_adapter.uniform_reference
out=dict(checkpoint='texture/iter_40000.pth',intervention='uniform candidate weights preserving learned null confidence',metrics=metrics)
out['mean_reference_confidence']=sum(n*v for n,v in confidence)/sum(n for n,v in confidence)
out['learned_null_logit']=float(runner.model.backbone.texture_adapter.null_logit.detach())
out['learned_temperature']=float(runner.model.backbone.texture_adapter.log_temperature.detach().exp())
(R/'UNIFORM_REFERENCE.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
