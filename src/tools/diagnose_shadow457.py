"""Frozen-weight gradient audit on fixed source training images, no optimization."""
from pathlib import Path
import sys,os,json,random,hashlib
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset
R=Path('/home/scv/shared/phase457_20261106');P=Path('/home/scv/shared/phase431_20261102')
sys.path[:0]=['/home/scv/shared/phase456_20261106','/home/scv/shared/phase450_20261106',str(P/'code')]
os.chdir('/home/scv/Cloud-Adapter-light/src')
from infer_shadow456 import Frozen431WithOcclusion
from train_shadow450 import Images,digest
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
torch.set_num_threads(4);torch.manual_seed(45701);random.seed(45701);np.random.seed(45701)
register_all_modules();import phase431.integration
cfg=Config.fromfile(str(P/'code/phase431/config.py'));cfg.model.backbone.variant='fused';cfg.model.backbone.repo='/home/scv/shared/phase429_20261102/dinov3';cfg.model.backbone.weights='/home/scv/shared/phase429_20261102/dinov3_splus_native.pth'
teacher=MODELS.build(cfg.model);teacher.load_state_dict(torch.load(P/'fused/best_mIoU_iter_32000.pth',map_location='cpu')['state_dict'],strict=True)
data=Images('train');indices=np.random.default_rng(45701).choice(len(data),64,replace=False).tolist()
random.seed(45701);batches=list(DataLoader(Subset(data,indices),batch_size=4,shuffle=False,num_workers=0))
results=[]
for step in [0,1000,6000]:
 state=torch.load(f'/home/scv/shared/phase456_20261106/A/network_run/branch_{step}.pth',map_location='cpu')['state_dict']
 net=Frozen431WithOcclusion(teacher,state).cuda().eval();before=digest(net);capture={}
 handle=net.branch.geometry.register_forward_hook(lambda m,args,out:capture.update(prob=args[0],feature=args[1],injection=out))
 params=list(net.branch.geometry.named_parameters());groups={'scene':[i for i,(n,p) in enumerate(params) if n.startswith('scene.')],'inject':[i for i,(n,p) in enumerate(params) if n.startswith('inject.')]}
 rows=[]
 for rgb,y,names in batches:
  y=y.cuda();logits,base,delta,gate=net(rgb.cuda());valid=y!=255;old=base.argmax(1);hard=(old!=y)&((old==2)|(y==2))&valid
  weights=torch.ones_like(y,dtype=torch.float);weights[hard]=4.;weights[(y==2)&(old==2)]=1.5
  ce=F.cross_entropy(logits,y,ignore_index=255,reduction='none');ce=(ce*weights*valid).sum()/(weights*valid).sum()
  prob=logits.softmax(1);oldprob=base.softmax(1);keep=valid&(old==y)&(y!=2)
  kl=2*((oldprob*(base-logits.log_softmax(1))).sum(1)*keep).sum()/keep.sum().clamp_min(1)
  soft=logits.sum()*0
  for c in [0,1]:
   gt=(y==c)&valid;p=prob[:,c]*valid;q=oldprob[:,c]*valid
   soft=soft+2*F.relu((q*gt).sum()/((q+gt-q*gt).sum()+1)-(p*gt).sum()/((p+gt-p*gt).sum()+1))
  reg=.01*delta.square().mean();terms=dict(ce=ce,keep=kl,soft=soft,reg=reg);grads={}
  for name,loss in terms.items():
   values=torch.autograd.grad(loss,[p for n,p in params],retain_graph=True,allow_unused=True)
   grads[name]=[torch.zeros_like(p).flatten() if g is None else g.detach().flatten() for g,(n,p) in zip(values,params)]
  row=dict(names=list(names),loss={n:float(v.detach()) for n,v in terms.items()},groups={})
  for group,ids in groups.items():
   vectors={n:torch.cat([v[i] for i in ids]) for n,v in grads.items()};protection=vectors['keep']+vectors['soft']+vectors['reg'];g=vectors['ce']
   row['groups'][group]=dict(norm={n:float(v.norm()) for n,v in vectors.items()},protection_norm=float(protection.norm()),ce_protection_cosine=float(F.cosine_similarity(g[None],protection[None],eps=1e-20)),total_norm=float((g+protection).norm()))
  with torch.no_grad():
   f=capture['feature'].detach();q=capture['prob'].detach();inj=capture['injection'].detach();q2=q.clone();q2[:,1]=q2[:,1].roll((q.shape[-2]//2,q.shape[-1]//2),(-2,-1))
   shifted=net.branch.geometry(q2,f)
   row['feature_rms']=float(f.square().mean().sqrt());row['injection_rms']=float(inj.square().mean().sqrt());row['cloud_position_intervention_rms']=float((inj-shifted).square().mean().sqrt())
  rows.append(row)
 assert digest(net)==before;handle.remove()
 results.append(dict(step=step,weight_unchanged=True,rows=rows))
 # Teacher hook handles belong to each wrapper; clear after finishing it.
 teacher.backbone.spatial._forward_hooks.clear();teacher.backbone._forward_hooks.clear()
 del net;torch.cuda.empty_cache()
out=dict(passed=True,source_train_only=True,images=64,indices=indices,target_label_supervision_percent=0,updates=0,results=results)
(R/'GRADIENT_DIAG.json').write_text(json.dumps(out,indent=2))
summary=[]
for x in results:
 rows=x['rows'];z=dict(step=x['step'],feature_rms=float(np.mean([v['feature_rms'] for v in rows])),injection_rms=float(np.mean([v['injection_rms'] for v in rows])),cloud_intervention_rms=float(np.mean([v['cloud_position_intervention_rms'] for v in rows])),groups={})
 for g in groups:
  z['groups'][g]={k:float(np.mean([v['groups'][g][k] for v in rows])) for k in ['protection_norm','ce_protection_cosine','total_norm']}
  z['groups'][g]['norm']={k:float(np.mean([v['groups'][g]['norm'][k] for v in rows])) for k in terms}
 summary.append(z)
(R/'GRADIENT_SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
