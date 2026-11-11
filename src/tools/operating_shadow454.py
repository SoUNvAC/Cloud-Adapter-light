"""Posthoc operating-point diagnostic, never used to select the reported model."""
from pathlib import Path
import sys,os,json
import numpy as np
import torch
from PIL import Image
R=Path('/home/scv/shared/phase454_20261106/A/network_run');P=Path('/home/scv/shared/phase431_20261102')
sys.path[:0]=['/home/scv/shared/phase450_20261106',str(P/'code'),'/home/scv/shared/phase400_20261031/code']
os.chdir('/home/scv/Cloud-Adapter-light/src')
from infer_shadow450 import Frozen431WithOcclusion
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
net=Frozen431WithOcclusion(teacher,torch.load(R/f'branch_{step}.pth',map_location='cpu')['state_dict']).cuda().eval()
old=LocalOcclusionBranch().cuda().eval();old.load_state_dict(torch.load('/home/scv/shared/phase450_20261106/network_run/branch_6000.pth',map_location='cpu')['state_dict'],strict=True)
offsets=[-1.,-.5,-.25,0.,.25,.5,1.];cms={m:np.zeros((len(offsets),3,3),np.int64) for m in ['450','454']};hist={m:np.zeros((2,4096),np.int64) for m in cms};clipped={m:0 for m in cms}
rows=[r for r in json.loads(Path('/home/scv/shared/phase442_20261103/INPUTS.json').read_text()) if r['domain']=='A']
for row in rows:
 rgb=torch.from_numpy(np.array(Image.open(row['image']).convert('RGB'))).permute(2,0,1)[None].cuda().float()
 raw=np.array(Image.open(row['mask']));y=raw.copy()
 for k,v in {0:0,1:1,2:1,3:2}.items():y[raw==k]=v
 valid=y!=255;y=y[valid].astype(np.int64)
 with torch.inference_mode():
  new,base,_,_=net(rgb);original,_,_=old(net.values['detail'],net.values['features'],rgb/255,base)
 for name,logits in [('450',original),('454',new)]:
  scores=logits[0].cpu().numpy()[:,valid];margin=scores[2]-scores[:2].max(0);other=scores[:2].argmax(0)
  for j,offset in enumerate(offsets):
   pred=np.where(margin+offset>0,2,other);cms[name][j]+=np.bincount(3*y+pred,minlength=9).reshape(3,3)
  clipped[name]+=int(((margin<=-40)|(margin>=40)).sum());bins=np.minimum(4095,np.maximum(0,((margin+40)*4096/80).astype(int)))
  for c in [0,1]:hist[name][c]+=np.bincount(bins[(y==2)==bool(c)],minlength=4096)
native=next(v for v in final['results'] if v['step']==step)
assert cms['454'][3].tolist()==native['confusion']
base_metrics=json.loads(Path('/home/scv/shared/phase450_20261106/network_run/FINAL_RESULTS.json').read_text())['results'][0]
assert cms['450'][3].tolist()==base_metrics['confusion']
target_fpr=base_metrics['nonShadow_FPR']/100;summary={}
for name in cms:
 neg,pos=hist[name][:,::-1];tp=np.cumsum(pos);fp=np.cumsum(neg);recall=tp/pos.sum();fpr=fp/neg.sum();precision=tp/np.maximum(1,tp+fp)
 summary[name]=dict(offset_curve=[dict(offset=o,**describe(cm)) for o,cm in zip(offsets,cms[name])],approx_binned_AP=float((precision*pos/pos.sum()).sum()),approx_recall_at_450_native_FPR=float(np.interp(target_fpr,fpr,recall)),histogram_clipped_pixels=clipped[name])
np.savez_compressed(R/'OPERATING_HISTOGRAM.npz',**hist,edges=np.linspace(-40,40,4097))
(R/'OPERATING_DIAG.json').write_text(json.dumps(dict(step=step,images=535,extra_backbone_forwards=535,diagnostic_only=True,note='Pooled pixel metrics on same validation set; histogram approximations; no threshold selected for main results',summary=summary),indent=2))
print(json.dumps({k:{a:b for a,b in v.items() if a!='offset_curve'} for k,v in summary.items()}))
