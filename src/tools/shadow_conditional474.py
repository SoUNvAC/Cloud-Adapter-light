"""Separate cloud-content and ground-occlusion paths with conditional composition."""
import torch
from torch import nn
from torch.nn import functional as F

def pool(x,k):
 return F.avg_pool2d(F.pad(x,(k//2,)*4,mode='replicate'),k,stride=1)

def head(channels):
 m=nn.Sequential(nn.Conv2d(channels,32,1),nn.GELU(),nn.Conv2d(32,32,3,padding=1,groups=32),nn.GELU(),nn.Conv2d(32,16,1),nn.GELU(),nn.Conv2d(16,1,1))
 nn.init.zeros_(m[-1].weight);nn.init.zeros_(m[-1].bias)
 return m

class ConditionalOcclusion(nn.Module):
 def __init__(self,conditional=True):
  super().__init__()
  self.detail=nn.ModuleList([nn.Conv2d(64,16,1) for _ in range(2)])
  self.semantic=nn.ModuleList([nn.ModuleList([nn.Conv2d(384,8,1) for _ in range(4)]) for _ in range(2)])
  self.cloud=head(38);self.shadow=head(43)
  self.register_buffer('conditional_mode',torch.tensor(conditional,dtype=torch.bool))
  self.enabled=True;self.conditional_enabled=True;self.last_delta=None;self.original_logits=None

 @staticmethod
 def compose(logits,delta,conditional):
  base=logits.log_softmax(1);c,s=delta[:,0:1],delta[:,1:2]
  if not conditional:return (base+torch.cat([torch.zeros_like(c),c,s],1)).log_softmax(1)
  cloud=base[:,1:2]-torch.logsumexp(base[:,[0,2]],1,keepdim=True)+c
  shadow=base[:,2:3]-base[:,:1]+s
  return torch.cat([F.logsigmoid(-cloud)+F.logsigmoid(-shadow),F.logsigmoid(cloud),F.logsigmoid(-cloud)+F.logsigmoid(shadow)],1)

 def forward(self,detail,features,rgb,logits):
  if not self.enabled:return logits
  size=detail.shape[-2:];sem=[]
  for projections in self.semantic:
   sem.append(sum(F.interpolate(p(F.layer_norm(x.permute(0,2,3,1),(384,)).permute(0,3,1,2)),size=size,mode='bilinear',align_corners=False) for p,x in zip(projections,features))/4)
  low=F.adaptive_avg_pool2d(rgb,size);p=F.adaptive_avg_pool2d(logits.softmax(1),size)
  light=(low+.02).log();chroma=low/(low.sum(1,keepdim=True)+.02)
  texture=((low-pool(low,3)).square().mean(1,keepdim=True)+1e-12).sqrt()
  contrast=torch.cat([light-(pool(low,k)+.02).log() for k in [5,13,29]],1)
  scene=sem[0].mean((2,3),keepdim=True).expand_as(sem[0])
  c=self.cloud(torch.cat([self.detail[0](detail),sem[0],low,p,scene],1))
  s=self.shadow(torch.cat([self.detail[1](detail),sem[1],light,chroma,texture,contrast,p],1))
  self.last_delta=F.interpolate(torch.cat([c,s],1),size=rgb.shape[-2:],mode='bilinear',align_corners=False)
  return self.compose(logits,self.last_delta,bool(self.conditional_mode) and self.conditional_enabled)

def install(branch,module):
 def apply(m,args,out):
  module.original_logits=out[0].detach()
  return module(*args[:3],out[0]),out[1],out[2]
 branch.register_forward_hook(apply)
