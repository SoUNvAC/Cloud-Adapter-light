from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
P=Path('/home/scv/shared/phase431_20261102');R=Path('/home/scv/shared/phase461_20261106')
sys.path[:0]=['/home/scv/shared/phase460_20261106','/home/scv/shared/phase450_20261106',str(P/'code'),'/home/scv/shared/phase400_20261031/code']
from infer_shadow460 import Frozen431WithOcclusion
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
register_all_modules();import phase431.integration
torch.set_num_threads(4)
cfg=Config.fromfile(str(P/'code/phase431/config.py'));cfg.model.backbone.variant='fused';cfg.model.backbone.repo='/home/scv/shared/phase429_20261102/dinov3';cfg.model.backbone.weights='/home/scv/shared/phase429_20261102/dinov3_splus_native.pth'
teacher=MODELS.build(cfg.model);teacher.load_state_dict(torch.load(P/'fused/best_mIoU_iter_32000.pth',map_location='cpu')['state_dict'],strict=True)
state=torch.load('/home/scv/shared/phase460_20261106/B/network_run/branch_1000.pth',map_location='cpu');loaded=teacher.load_state_dict(state['student_patch'],strict=False);assert not loaded.unexpected_keys
net=Frozen431WithOcclusion(teacher,state['state_dict']).cuda().eval();module=net.branch.reference_decoder;captured={}
for i,m in enumerate(module.project):m.register_forward_hook(lambda m,a,o,i=i:captured.update({i:o}))
files=sorted(Path('/home/scv/Cloud-Adapter-light/src/data/cloudsen12_high_l1c/img_dir/train').glob('*.png'));indices=np.random.default_rng(46100).choice(len(files),64,replace=False);records=[]
with torch.inference_mode():
 for index in indices:
  path=files[index];rgb=torch.from_numpy(np.array(Image.open(path).convert('RGB'))).permute(2,0,1)[None].cuda();net(rgb)
  for i,z in captured.items():
   h,w=z.shape[-2:];low=F.adaptive_avg_pool2d(z,(min(h,32),min(w,32)));memory=F.adaptive_avg_pool2d(low,(16,16));ground=module.region[i](memory).softmax(1)[:,0].flatten(1)
   q=F.normalize(module.match[i](low).flatten(2),dim=1).transpose(1,2);k=F.normalize(module.match[i](memory).flatten(2),dim=1)
   attention=(8*(q@k)+(.02+ground[:,None]).log()).softmax(-1)
   ref=(attention@memory.flatten(2).transpose(1,2)).transpose(1,2)
   records.append(dict(name=path.name,stage=i,normalized_entropy=float((-(attention*attention.clamp_min(1e-9).log()).sum(-1)/np.log(256)).mean()),top8_mass=float(attention.topk(8,dim=-1).values.sum(-1).mean()),reference_spatial_std=float(ref.std(-1).mean()),query_spatial_std=float(low.flatten(2).std(-1).mean())))
result=dict(training_images=64,optimization_steps=0,indices=indices.tolist(),records=records,stage_means={str(i):{k:float(np.mean([r[k] for r in records if r['stage']==i])) for k in ['normalized_entropy','top8_mass','reference_spatial_std','query_spatial_std']} for i in range(4)})
(R/'REFERENCE_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['stage_means']))
