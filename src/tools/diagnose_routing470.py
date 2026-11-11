"""Post-training route usage by baseline error type, labels for analysis only."""
from pathlib import Path
import os,sys,json
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
R=Path('/home/scv/shared/phase470_20261107');P=Path('/home/scv/shared/phase431_20261102')
sys.path[:0]=['/home/scv/shared/phase450_20261106',str(P/'code'),'/home/scv/shared/phase400_20261031/code']
os.chdir('/home/scv/Cloud-Adapter-light/src')
from infer_shadow470 import Frozen431WithOcclusion
from shadow_experts470 import ErrorExperts
from train_shadow450 import digest
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
register_all_modules();import phase431.integration
torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
d=R/'B/network_run';step=json.loads((d/'FINAL_RESULTS.json').read_text())['choices']['best_shadow']
cfg=Config.fromfile(str(P/'code/phase431/config.py'));cfg.model.backbone.variant='fused';cfg.model.backbone.repo='/home/scv/shared/phase429_20261102/dinov3';cfg.model.backbone.weights='/home/scv/shared/phase429_20261102/dinov3_splus_native.pth'
teacher=MODELS.build(cfg.model);teacher.load_state_dict(torch.load(P/'fused/best_mIoU_iter_32000.pth',map_location='cpu')['state_dict'],strict=True)
state=torch.load(d/f'branch_{step}.pth',map_location='cpu');teacher.load_state_dict(state['student_patch'],strict=False)
net=Frozen431WithOcclusion(teacher,state['state_dict']).cuda().eval();before=digest(net)
rows=[r for r in json.loads(Path('/home/scv/shared/phase442_20261103/INPUTS.json').read_text()) if r['domain']=='A']
count=np.zeros(4);sums=np.zeros((4,4));hard=np.zeros((4,4));per=[]
for row in rows:
 rgb=np.array(Image.open(row['image']).convert('RGB'));raw=np.array(Image.open(row['mask']));truth=raw.copy()
 for k,v in {0:0,1:1,2:1,3:2}.items():truth[raw==k]=v
 old=np.load(Path('/home/scv/shared/phase450_20261106/network_run/independent/step_6000')/f"{row['index']:03d}.npz")['pred']
 t=np.zeros_like(truth);t[(truth==2)&(old!=2)]=1;t[(truth==0)&(old==2)]=2;t[(truth==1)&(old==2)]=3;t[truth==255]=255
 with torch.inference_mode():
  pred=net(torch.from_numpy(rgb).permute(2,0,1)[None].cuda())[0].argmax(1)[0].cpu().numpy()
  expected=np.load(d/'independent'/f'step_{step}'/f"{row['index']:03d}.npz")['pred'];assert np.array_equal(pred,expected)
  gate=net.branch.experts.route_logits.softmax(1)
  tt=torch.from_numpy(t.astype(np.int64))[None].cuda();valid=tt!=255
  mass=F.adaptive_avg_pool2d(F.one_hot(tt.clamp(0,3),4).permute(0,3,1,2).float()*valid[:,None],gate.shape[-2:])
  s=torch.einsum('bchw,bkhw->ck',mass,gate).cpu().numpy()
  hg=F.one_hot(gate.argmax(1),4).permute(0,3,1,2).float();h=torch.einsum('bchw,bkhw->ck',mass,hg).cpu().numpy()
  c=mass.sum((0,2,3)).cpu().numpy()
 count+=c;sums+=s;hard+=h;per.append(dict(index=row['index'],group=row['group'],mass=c.tolist(),route_weight_sum=s.tolist(),hard_route_sum=h.tolist()))
assert digest(net)==before
result=dict(step=step,types=['other','missed_shadow','surface_false_shadow','cloud_false_shadow'],mean_route_by_error_type=(sums/count[:,None]).tolist(),hard_route_by_error_type=(hard/count[:,None]).tolist(),quarter_grid_mass=count.tolist(),independent_predictions_identical=True,state_unchanged=True,extra_forwards=535,per_image=per)
(R/'ROUTING_DIAG.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='per_image'}))
