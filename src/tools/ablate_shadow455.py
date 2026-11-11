"""Remove only the new path from the trained network, with unchanged argmax."""
from pathlib import Path
import sys,os,json
import numpy as np
import torch
from PIL import Image
R=Path('/home/scv/shared/phase455_20261106/A/network_run');P=Path('/home/scv/shared/phase431_20261102')
sys.path[:0]=['/home/scv/shared/phase450_20261106',str(P/'code'),'/home/scv/shared/phase400_20261031/code']
os.chdir('/home/scv/Cloud-Adapter-light/src')
from infer_shadow455 import Frozen431WithOcclusion
from shadow_branch450 import LocalOcclusionBranch
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
from report_phase400 import describe
register_all_modules();import phase431.integration
torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
final=json.loads((R/'FINAL_RESULTS.json').read_text());step=final['choices']['best_shadow']
cfg=Config.fromfile(str(P/'code/phase431/config.py'));cfg.model.backbone.variant='fused';cfg.model.backbone.repo='/home/scv/shared/phase429_20261102/dinov3';cfg.model.backbone.weights='/home/scv/shared/phase429_20261102/dinov3_splus_native.pth'
teacher=MODELS.build(cfg.model);teacher.load_state_dict(torch.load(P/'fused/best_mIoU_iter_32000.pth',map_location='cpu')['state_dict'],strict=True)
state=torch.load(R/f'branch_{step}.pth',map_location='cpu')['state_dict']
net=Frozen431WithOcclusion(teacher,state).cuda().eval()
off=LocalOcclusionBranch().cuda().eval();off.load_state_dict({n:t for n,t in state.items() if not n.startswith('continuity.')},strict=True)
cms={m:np.zeros((3,3),np.int64) for m in ['enabled','disabled']};per=[]
rows=[r for r in json.loads(Path('/home/scv/shared/phase442_20261103/INPUTS.json').read_text()) if r['domain']=='A']
for row in rows:
 rgb=torch.from_numpy(np.array(Image.open(row['image']).convert('RGB'))).permute(2,0,1)[None].cuda().float()
 raw=np.array(Image.open(row['mask']));y=raw.copy()
 for k,v in {0:0,1:1,2:1,3:2}.items():y[raw==k]=v
 valid=y!=255;y=y[valid].astype(np.int64)
 with torch.inference_mode():
  logits,base,_,_=net(rgb);disabled,_,_=off(net.values['detail'],net.values['features'],rgb/255,base)
 entry=dict(index=row['index'],group=row['group'])
 for name,scores in [('enabled',logits),('disabled',disabled)]:
  pred=scores.argmax(1)[0].cpu().numpy()[valid];cm=np.bincount(3*y+pred,minlength=9).reshape(3,3);cms[name]+=cm;entry[name]=cm.tolist()
 per.append(entry)
assert cms['enabled'].tolist()==next(v for v in final['results'] if v['step']==step)['confusion']
result=dict(passed=True,step=step,images=535,extra_backbone_forwards=535,enabled=describe(cms['enabled']),disabled=describe(cms['disabled']),note='Same trained shared weights; removal tests dependence on the new path, not an independently trained architecture control; fixed argmax')
(R/'MODULE_ABLATION.json').write_text(json.dumps(result,indent=2));(R/'MODULE_ABLATION_PER_IMAGE.json').write_text(json.dumps(per));print(json.dumps(result))
