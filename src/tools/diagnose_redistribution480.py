"""Measure first-pyramid residual and patch-mean conservation on all535 images."""
from pathlib import Path
import os,sys,json
import numpy as np
import torch
from PIL import Image
R=Path('/home/scv/shared/phase480_20261108');P=Path('/home/scv/shared/phase431_20261102')
sys.path[:0]=['/home/scv/shared/phase450_20261106',str(P/'code'),'/home/scv/shared/phase400_20261031/code']
os.chdir('/home/scv/Cloud-Adapter-light/src')
from infer_shadow480 import Frozen431WithOcclusion
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
net=Frozen431WithOcclusion(teacher,state['state_dict']).cuda().eval();before=digest(net);assert net.branch.redistribution.conserve_mode.item()
rows=[r for r in json.loads(Path('/home/scv/shared/phase442_20261103/INPUTS.json').read_text()) if r['domain']=='A'];per=[];projection={}
def projection_check(m,args,out):
 from torch.nn import functional as F
 lowmean=F.avg_pool2d(args[0].double(),4,4)
 expected=F.conv2d(lowmean,m.weight.double())
 actual=F.avg_pool2d(out.double(),4,4)
 # Redo only this projection, not the model, to isolate reduced-precision convolution.
 with torch.backends.cudnn.flags(allow_tf32=False):strict=F.conv2d(args[0],m.weight)
 strictmean=F.avg_pool2d(strict.double(),4,4)
 projection.update(lowrank_mean_max=lowmean.abs().max().item(),linear_expected_mean_max=expected.abs().max().item(),actual_mean_max=actual.abs().max().item(),strict_fp32_mean_max=strictmean.abs().max().item(),projection_rounding_max=(actual-expected).abs().max().item(),delta_rms=out.square().mean().sqrt().item())
net.branch.redistribution.inject.register_forward_hook(projection_check)
for row in rows:
 rgb=np.array(Image.open(row['image']).convert('RGB'))
 with torch.inference_mode():pred=net(torch.from_numpy(rgb).permute(2,0,1)[None].cuda())[0].argmax(1)[0].cpu().numpy()
 expected=np.load(d/'independent'/f'step_{step}'/f"{row['index']:03d}.npz")['pred'];assert np.array_equal(pred,expected)
 values=[net.branch.redistribution.relative_change.item()];assert np.isfinite(values).all()
 per.append(dict(index=row['index'],group=row['group'],relative_input_rms=values,patch_mean_residual_max=net.branch.redistribution.mean_residual.item(),projection=dict(projection)))
assert digest(net)==before
values=np.array([r['relative_input_rms'] for r in per]);result=dict(step=step,mean_relative_detail_rms=float(values.mean()),median_relative_detail_rms=float(np.median(values)),independent_predictions_identical=True,state_unchanged=True,extra_forwards=535,per_image=per)
result['maximum_patch_mean_residual']=max(v['patch_mean_residual_max'] for v in per)
result['original_absolute_tolerance']=1e-5
result['original_absolute_tolerance_passed']=result['maximum_patch_mean_residual']<1e-5
result['projection_maxima']={k:max(v['projection'][k] for v in per) for k in projection}
result['cudnn_allow_tf32']=torch.backends.cudnn.allow_tf32
(R/'REDISTRIBUTION_DIAG.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='per_image'}))
