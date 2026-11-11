"""Remove only the new path from the trained network, with unchanged argmax."""
from pathlib import Path
import sys,os,json
import numpy as np
import torch
from PIL import Image
R=Path('/home/scv/shared/phase458_20261106/A/network_run');P=Path('/home/scv/shared/phase431_20261102')
sys.path[:0]=['/home/scv/shared/phase450_20261106',str(P/'code'),'/home/scv/shared/phase400_20261031/code']
os.chdir('/home/scv/Cloud-Adapter-light/src')
from infer_shadow458 import Frozen431WithOcclusion
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
off=LocalOcclusionBranch().cuda().eval();off.load_state_dict({n:t for n,t in state.items() if not n.startswith('region.')},strict=True)
cms={m:np.zeros((3,3),np.int64) for m in ['enabled','disabled','propagation_disabled']};per=[]
bound=dict(valid=0,shadow=0,saturated=0,gate_sum=0.,delta_abs_sum=0.,new_path_logit_abs_sum=0.,unreachable_current_gate=0,unreachable_max_gate=0,forced_FP_current_gate=0,forced_FP_max_gate=0)
rows=[r for r in json.loads(Path('/home/scv/shared/phase442_20261103/INPUTS.json').read_text()) if r['domain']=='A']
for row in rows:
 rgb=torch.from_numpy(np.array(Image.open(row['image']).convert('RGB'))).permute(2,0,1)[None].cuda().float()
 raw=np.array(Image.open(row['mask']));y=raw.copy()
 for k,v in {0:0,1:1,2:1,3:2}.items():y[raw==k]=v
 valid=y!=255;y=y[valid].astype(np.int64)
 with torch.inference_mode():
  logits,base,delta,gate=net(rgb);disabled,_,_=off(net.values['detail'],net.values['features'],rgb/255,base)
 base_scores=base[0].cpu().numpy()[:,valid];margin=base_scores[2]-base_scores[:2].max(0);g=gate[0,0].cpu().numpy()[valid];d=delta[0,0].cpu().numpy()[valid];shadow=y==2
 bound['valid']+=len(y);bound['shadow']+=int(shadow.sum());bound['saturated']+=int((np.abs(d)/(4*g+1e-8)>.95).sum());bound['gate_sum']+=float(g.astype(np.float64).sum());bound['delta_abs_sum']+=float(np.abs(d).astype(np.float64).sum());bound['new_path_logit_abs_sum']+=float((logits-disabled).abs()[0,2].cpu().numpy()[valid].astype(np.float64).sum())
 bound['unreachable_current_gate']+=int((shadow&(margin+4*g<=0)).sum());bound['unreachable_max_gate']+=int((shadow&(margin+4<=0)).sum());bound['forced_FP_current_gate']+=int((~shadow&(margin-4*g>0)).sum());bound['forced_FP_max_gate']+=int((~shadow&(margin-4>0)).sum())
 with torch.inference_mode():
  net.branch.region.propagation_enabled=False
  no_prop,_,_=net.branch(net.values['detail'],net.values['features'],rgb/255,base)
  net.branch.region.propagation_enabled=True
 entry=dict(index=row['index'],group=row['group'])
 for name,scores in [('enabled',logits),('disabled',disabled),('propagation_disabled',no_prop)]:
  pred=scores.argmax(1)[0].cpu().numpy()[valid];cm=np.bincount(3*y+pred,minlength=9).reshape(3,3);cms[name]+=cm;entry[name]=cm.tolist()
 per.append(entry)
assert cms['enabled'].tolist()==next(v for v in final['results'] if v['step']==step)['confusion']
result=dict(passed=True,step=step,images=535,extra_backbone_forwards=535,enabled=describe(cms['enabled']),disabled=describe(cms['disabled']),propagation_disabled=describe(cms['propagation_disabled']),note='Same trained shared weights; removal tests dependence on the new path, not an independently trained architecture control; fixed argmax')
bound['current_gate_oracle_shadow_IoU']=100*(bound['shadow']-bound['unreachable_current_gate'])/(bound['shadow']+bound['forced_FP_current_gate'])
bound['max_gate_oracle_shadow_IoU']=100*(bound['shadow']-bound['unreachable_max_gate'])/(bound['shadow']+bound['forced_FP_max_gate'])
bound['note']='Label-aware feasibility bound only, not a model prediction; assumes ideal per-pixel residual and does not enforce surface/cloud IoU protection. Current gate bound conditional on this checkpoint; max gate uses absolute architecture limit4.'
(R/'RESIDUAL_CAPACITY.json').write_text(json.dumps(bound,indent=2))
(R/'MODULE_ABLATION.json').write_text(json.dumps(result,indent=2));(R/'MODULE_ABLATION_PER_IMAGE.json').write_text(json.dumps(per));print(json.dumps(result))
