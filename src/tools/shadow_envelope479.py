"""Multiscale dark-valley descriptors injected before spatial interactions."""
import torch
from torch import nn
from torch.nn import functional as F

def maximum(x,k):
 # Separable flat square dilation with replicate boundary extension.
 x=F.max_pool2d(F.pad(x,(k//2,k//2,0,0),mode='replicate'),(1,k),stride=1)
 return F.max_pool2d(F.pad(x,(0,0,k//2,k//2),mode='replicate'),(k,1),stride=1)

def closing(x,k):return -maximum(-maximum(x,k),k)

def average(x,k):
 x=F.avg_pool2d(F.pad(x,(k//2,k//2,0,0),mode='replicate'),(1,k),stride=1)
 return F.avg_pool2d(F.pad(x,(0,0,k//2,k//2),mode='replicate'),(k,1),stride=1)

class ShadowEnvelope(nn.Module):
 def __init__(self,morphology=True):
  super().__init__();self.register_buffer('morphology_mode',torch.tensor(morphology,dtype=torch.bool))
  self.mix=nn.Sequential(nn.Conv2d(79,32,1),nn.GELU(),nn.Conv2d(32,32,3,padding=1,groups=32),nn.GELU(),nn.Conv2d(32,64,1))
  nn.init.zeros_(self.mix[-1].weight);nn.init.zeros_(self.mix[-1].bias)
  self.enabled=True;self.rgb=None;self.relative_change=None
 def descriptor(self,rgb,size):
  x=F.adaptive_avg_pool2d(rgb,size).clamp_min(1/255).log();features=[]
  for k in [5,13,29]:
   ref=closing(x,k) if bool(self.morphology_mode) else average(x,k)
   gap=(ref-x).clamp_min(0);mean=gap.mean(1,keepdim=True)
   spread=(gap-mean).square().mean(1,keepdim=True).add(1e-12).sqrt()
   features.extend([gap,mean,spread])
  return torch.cat(features,1)
 def forward(self,context,detail):
  if not self.enabled:return context,detail
  desc=self.descriptor(self.rgb,detail.shape[-2:]);delta=self.mix(torch.cat([detail,desc],1))
  self.relative_change=(delta.square().mean()/(detail.square().mean()+1e-12)).sqrt().detach()
  return context+F.adaptive_avg_pool2d(delta,context.shape[-2:]),detail+delta

def install(backbone,module):
 def prepare(m,args):
  x=args[0];mean=x.new_tensor([123.675,116.28,103.53])[None,:,None,None];std=x.new_tensor([58.395,57.12,57.375])[None,:,None,None]
  module.rgb=((x*std+mean)/255).clamp(0,1)
 backbone.register_forward_pre_hook(prepare)
 backbone.spatial.register_forward_hook(lambda m,a,o:module(*o),prepend=True)
