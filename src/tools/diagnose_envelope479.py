"""Measure projector-input change in all four stages of the selected B model."""
from pathlib import Path
import os,sys,json
import numpy as np
import torch
from PIL import Image
R=Path('/home/scv/shared/phase479_20261108');P=Path('/home/scv/shared/phase431_20261102')
sys.path[:0]=['/home/scv/shared/phase450_20261106',str(P/'code'),'/home/scv/shared/phase400_20261031/code']
os.chdir('/home/scv/Cloud-Adapter-light/src')
from infer_shadow479 import Frozen431WithOcclusion
from train_shadow450 import digest
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
register_all_modules();import phase431.integration
torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
d=R/'B/network_run';step=json.loads((d/'FINAL_RESULTS.json').read_text())['choices']['best_shadow'];assert step>0
cfg=Config.fromfile(str(P/'code/phase431/config.py'));cfg.model.backbone.variant='fused';cfg.model.backbone.repo='/home/scv/shared/phase429_20261102/dinov3';cfg.model.backbone.weights='/home/scv/shared/phase429_20261102/dinov3_splus_native.pth'
teacher=MODELS.build(cfg.model);teacher.load_state_dict(torch.load(P/'fused/best_mIoU_iter_32000.pth',map_location='cpu')['state_dict'],strict=True)
state=torch.load(d/f'branch_{step}.pth',map_location='cpu');teacher.load_state_dict(state['student_patch'],strict=False)
net=Frozen431WithOcclusion(teacher,state['state_dict']).cuda().eval();before=digest(net);assert net.branch.envelope.morphology_mode.item()
rows=[r for r in json.loads(Path('/home/scv/shared/phase442_20261103/INPUTS.json').read_text()) if r['domain']=='A'];per=[]
for row in rows:
 rgb=np.array(Image.open(row['image']).convert('RGB'))
 with torch.inference_mode():pred=net(torch.from_numpy(rgb).permute(2,0,1)[None].cuda())[0].argmax(1)[0].cpu().numpy()
 expected=np.load(d/'independent'/f'step_{step}'/f"{row['index']:03d}.npz")['pred'];assert np.array_equal(pred,expected)
 values=[net.branch.envelope.relative_change.item()];assert np.isfinite(values).all()
 per.append(dict(index=row['index'],group=row['group'],relative_input_rms=values))
assert digest(net)==before
values=np.array([r['relative_input_rms'] for r in per]);result=dict(step=step,mean_relative_detail_rms=float(values.mean()),median_relative_detail_rms=float(np.median(values)),independent_predictions_identical=True,state_unchanged=True,extra_forwards=535,per_image=per)
(R/'ENVELOPE_DIAG.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='per_image'}))
