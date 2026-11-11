"""Quantify learned transport and distance accuracy; no state update."""
from pathlib import Path
import os,sys,json
import numpy as np
import torch
from PIL import Image
R=Path('/home/scv/shared/phase471_20261107');P=Path('/home/scv/shared/phase431_20261102')
sys.path[:0]=['/home/scv/shared/phase450_20261106',str(P/'code'),'/home/scv/shared/phase400_20261031/code']
os.chdir('/home/scv/Cloud-Adapter-light/src')
from infer_shadow471 import Frozen431WithOcclusion
from shadow_transport471 import distance_target
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
net=Frozen431WithOcclusion(teacher,state['state_dict']).cuda().eval();before=digest(net);captured={}
net.branch.transport.motion.register_forward_hook(lambda m,a,o:captured.update(motion=o))
net.branch.transport.register_forward_hook(lambda m,a,o:captured.update(relative_feature_rms=(o.square().mean()/(a[0][0].square().mean()+1e-12)).sqrt().item()))
rows=[r for r in json.loads(Path('/home/scv/shared/phase442_20261103/INPUTS.json').read_text()) if r['domain']=='A']
per=[];count=0;flow_sum=0.;gate_sum=0.;proxy_sum=0.;mae_sum=0.;valid_sum=0;cm=np.zeros((2,2),np.int64);hist=np.zeros(160,np.int64);edges=np.linspace(0,8.00001,161)
for row in rows:
 rgb=np.array(Image.open(row['image']).convert('RGB'));raw=np.array(Image.open(row['mask']));truth=raw.copy()
 for k,v in {0:0,1:1,2:1,3:2}.items():truth[raw==k]=v
 with torch.inference_mode():
  pred=net(torch.from_numpy(rgb).permute(2,0,1)[None].cuda())[0].argmax(1)[0].cpu().numpy()
  expected=np.load(d/'independent'/f'step_{step}'/f"{row['index']:03d}.npz")['pred'];assert np.array_equal(pred,expected)
  m=net.branch.transport;flow=m.flow.square().sum(1,keepdim=True).sqrt()*4
  gate=captured['motion'][:,2:3].sigmoid()*torch.exp(-4*m.sdf.abs())
  target,valid=distance_target(torch.from_numpy(truth.astype(np.int64))[None].cuda(),m.sdf.shape[-2:])
  mae=((m.sdf-target).abs()*valid).sum().item();n=valid.sum().item()
  a=(target>0)[valid].long();b=(m.sdf>0)[valid].long();c=torch.bincount(2*a+b,minlength=4).reshape(2,2).cpu().numpy()
  values=flow.flatten().cpu().numpy();h=np.histogram(values,bins=edges)[0];assert values.max()<=8.00001
  f=flow.sum().item();g=gate.sum().item();p=(flow*gate).sum().item();num=flow.numel()
 count+=num;flow_sum+=f;gate_sum+=g;proxy_sum+=p;mae_sum+=mae;valid_sum+=n;cm+=c;hist+=h
 per.append(dict(index=row['index'],group=row['group'],mean_flow_pixels=f/num,mean_gate=g/num,blend_weighted_flow_proxy_pixels=p/num,relative_feature_rms=captured['relative_feature_rms'],distance_mae=mae/max(n,1),distance_head_confusion=c.tolist()))
assert digest(net)==before and hist.sum()==count
result=dict(step=step,mean_flow_pixels=flow_sum/count,mean_gate=gate_sum/count,blend_weighted_flow_proxy_pixels=proxy_sum/count,distance_mae=mae_sum/valid_sum,distance_head_shadow_IoU=100*cm[1,1]/(cm[1].sum()+cm[:,1].sum()-cm[1,1]),distance_head_confusion=cm.tolist(),flow_hist=hist.tolist(),flow_edges=edges.tolist(),independent_predictions_identical=True,state_unchanged=True,extra_forwards=535,per_image=per)
result['mean_relative_feature_rms']=float(np.mean([p['relative_feature_rms'] for p in per]))
(R/'TRANSPORT_DIAG.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['per_image','flow_hist','flow_edges']}))
