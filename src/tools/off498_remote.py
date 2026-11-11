from pathlib import Path
import json,sys
R=Path('/home/scv/shared/phase498_20261110')
assert json.loads((R/'STATUS.json').read_text())['state']=='complete'
sys.path[:0]=[str(R/'code'),'/home/scv/Cloud-Adapter-light/src']
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.utils import register_all_modules
register_all_modules()
cfg=Config.fromfile(str(R/'code/phase498/config.py'))
cfg.model.backbone.update(variant='texture',repo='/home/scv/shared/phase429_20261102/dinov3',weights='/home/scv/shared/phase429_20261102/dinov3_splus_native.pth')
cfg.work_dir=str(R/'eval_early_off_final')
cfg.load_from=str(R/'texture/iter_40000.pth')
cfg.test_dataloader=cfg.val_dataloader
cfg.test_dataloader.dataset.data_root='/home/scv/Cloud-Adapter-light/src/data/cloudsen12_high_l1c'
cfg.test_evaluator=dict(type='ExactIoU498',iou_metrics=['mIoU'])
cfg.custom_hooks=[]
runner=Runner.from_cfg(cfg)
assert len(runner.test_dataloader.dataset)==535
runner.model.backbone.texture_adapter.early_enabled=False
metrics=runner.test()
assert not runner.model.backbone.texture_adapter.early_enabled
out=dict(checkpoint='texture/iter_40000.pth',intervention='disable only four early context injections; retain late texture branch',metrics=metrics)
(R/'EARLY_OFF.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
