"""Zero-update training-set audit of routing scale and gradient conflict."""
from pathlib import Path
import sys,os,json,random,hashlib
import numpy as np
import torch
from torch.nn import functional as F
R=Path('/home/scv/shared/phase465_20261107');P=Path('/home/scv/shared/phase431_20261102');SRC=Path('/home/scv/Cloud-Adapter-light/src')
sys.path[:0]=['/home/scv/shared/phase464_20261107','/home/scv/shared/phase450_20261106',str(P/'code')];os.chdir(SRC)
from infer_shadow464 import Frozen431WithOcclusion
from train_shadow450 import Images,digest,sha
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
register_all_modules();import phase431.integration
torch.set_num_threads(4);torch.manual_seed(46501);random.seed(46501)
torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
cfg=Config.fromfile(str(P/'code/phase431/config.py'));cfg.model.backbone.variant='fused';cfg.model.backbone.repo='/home/scv/shared/phase429_20261102/dinov3';cfg.model.backbone.weights='/home/scv/shared/phase429_20261102/dinov3_splus_native.pth'
def build():
 m=MODELS.build(cfg.model);m.load_state_dict(torch.load(P/'fused/best_mIoU_iter_32000.pth',map_location='cpu')['state_dict'],strict=True);return m
origin=Path('/home/scv/shared/phase464_20261107/B/network_run/branch_1000.pth');state=torch.load(origin,map_location='cpu');student=build();student.load_state_dict(state['student_patch'],strict=False)
network=Frozen431WithOcclusion(student,state['state_dict']).cuda().eval();teacher=build().cuda().eval().requires_grad_(False)
network.requires_grad_(False)
for r in network.branch.routing:r.requires_grad_(True)
before=digest(network);teacher_before=digest(teacher);measurements=[];gradients=[]
def hook(slot):
 def run(r,args,output):
  original,x,ctx=args
  with torch.no_grad():
   b,n,c=x.shape;ix=torch.arange(0,n,16,device=x.device);xs=x[:,ix];flat=ctx.flatten(2).transpose(1,2)
   split=lambda v:v.reshape(b,-1,8,64).transpose(1,2)
   q,k,v=split(original.q(xs)),split(original.k(flat)),split(original.v(flat))
   logits=q@k.transpose(-1,-2)/8
   qp=r.query_region(F.layer_norm(xs,(c,))).softmax(-1);kp=r.key_region(ctx).flatten(2).softmax(1)
   bias=torch.einsum('bnc,hcd,bdm->bhnm',qp,4*r.relation.tanh(),kp)
   a=logits.softmax(-1);ar=(logits+bias).softmax(-1)
   base=original.out((a@v).transpose(1,2).flatten(2));routed=original.out((ar@v).transpose(1,2).flatten(2))
   measurements.append(dict(layer=slot,logit_row_std=float(logits.std(-1).mean()),bias_row_std=float(bias.std(-1).mean()),attention_L1=float((a-ar).abs().sum(-1).mean()),relative_update_change=float((routed-base).norm()/base.norm().clamp_min(1e-10))))
 return run
for i,r in enumerate(network.branch.routing):r.register_forward_hook(hook(i))
params=list(network.branch.routing.named_parameters());groups={'relation':[i for i,(n,p) in enumerate(params) if n.endswith('relation')],'region_predictors':[i for i,(n,p) in enumerate(params) if not n.endswith('relation')]}
train=Images('train');indices=np.random.default_rng(46501).choice(len(train),32,replace=False).tolist();trace=[]
for start in range(0,32,4):
 batch=[train[i] for i in indices[start:start+4]];rgb=torch.stack([v[0] for v in batch]).cuda().float();y=torch.stack([v[1] for v in batch]).cuda();trace.extend([v[2] for v in batch])
 meta=[dict(img_shape=(512,512),ori_shape=(512,512),pad_shape=(512,512),padding_size=[0,0,0,0]) for _ in rgb]
 score=student.encode_decode((rgb-network.mean)/network.std,meta).clamp_min(1e-8);base=(score/score.sum(1,keepdim=True)).log();logits,delta,_=network.branch(network.values['detail'],network.values['features'],rgb/255,base)
 with torch.no_grad():
  ref=teacher.encode_decode((rgb-network.mean)/network.std,meta).clamp_min(1e-8);ref=(ref/ref.sum(1,keepdim=True)).log()
 valid=y!=255;old=ref.argmax(1);weights=torch.ones_like(y,dtype=torch.float);weights[(old!=y)&((old==2)|(y==2))&valid]=4;weights[(old==2)&(y==2)]=1.5
 ce=(F.cross_entropy(logits,y,ignore_index=255,reduction='none')*weights*valid).sum()/(weights*valid).sum();p=logits.softmax(1);q=ref.softmax(1);keep=valid&(old==y)&(y!=2)
 preserve=2*((q*(ref-logits.log_softmax(1))).sum(1)*keep).sum()/keep.sum().clamp_min(1)
 for c in [0,1]:
  gt=(y==c)&valid;pc=p[:,c]*valid;qc=q[:,c]*valid;preserve=preserve+2*F.relu((qc*gt).sum()/((qc+gt-qc*gt).sum()+1)-(pc*gt).sum()/((pc+gt-pc*gt).sum()+1))
 losses={'CE':ce,'preserve':preserve,'auxiliary':sum(r.auxiliary(y) for r in network.branch.routing)/4}
 vectors={}
 for name,loss in losses.items():
  g=torch.autograd.grad(loss,[p for n,p in params],retain_graph=True,allow_unused=True)
  vectors[name]={group:torch.cat([(torch.zeros_like(params[i][1]) if g[i] is None else g[i]).flatten() for i in ids]) for group,ids in groups.items()}
 entry={'losses':{k:float(v.detach()) for k,v in losses.items()}}
 for group in groups:
  v={k:g[group] for k,g in vectors.items()};entry[group]={k+'_norm':float(x.norm()) for k,x in v.items()};entry[group]['CE_preserve_cosine']=float(F.cosine_similarity(v['CE'][None],v['preserve'][None]).item())
 gradients.append(entry)
assert digest(network)==before and digest(teacher)==teacher_before
summary=[dict(layer=i,**{k:float(np.mean([m[k] for m in measurements if m['layer']==i])) for k in ['logit_row_std','bias_row_std','attention_L1','relative_update_change']}) for i in range(4)]
out=dict(passed=True,optimizer_updates=0,training_images=32,sampling_seed=46501,indices=indices,images=trace,checkpoint=str(origin),checkpoint_sha=sha(origin),summary=summary,measurements=measurements,gradients=gradients)
(R/'ROUTING_DIAG.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(summary=summary,gradient_mean={group:{k:float(np.mean([v[group][k] for v in gradients])) for k in gradients[0][group]} for group in groups})))
