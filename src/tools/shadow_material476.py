"""Chromaticity-guided learned spatial differences before VFM interactions."""
import torch
from torch import nn
from torch.nn import functional as F

def adjacent(x):
 b,c,h,w=x.shape
 u=F.unfold(F.pad(x,(1,1,1,1),mode='replicate'),3).reshape(b,c,9,h,w)
 return torch.cat([u[:,:,:4],u[:,:,5:]],2)

class MaterialConvolution(nn.Module):
 def __init__(self,chroma=True):
  super().__init__();self.kernel=nn.Parameter(torch.zeros(1,64,8,1,1))
  self.mix=nn.Conv2d(64,64,1,bias=False)
  with torch.no_grad():self.mix.weight.copy_(torch.eye(64)[:,:,None,None])
  self.log_temperature=nn.Parameter(torch.tensor(-.5))
  self.register_buffer('chroma_mode',torch.tensor(chroma,dtype=torch.bool))
  self.enabled=True;self.chroma_enabled=True;self.rgb=None;self.relative_change=None

 def guide(self,rgb,size):
  g=F.adaptive_avg_pool2d(rgb,size).clamp_min(1/255).log()
  if bool(self.chroma_mode) and self.chroma_enabled:g=g-g.mean(1,keepdim=True)
  return g

 def forward(self,context,detail):
  if not self.enabled:return context,detail
  g=self.guide(self.rgb,detail.shape[-2:]);difference=adjacent(g)-g[:,:,None]
  tau=F.softplus(self.log_temperature)+.01
  affinity=torch.exp(-difference.square().mean(1,keepdim=True)/tau.square())
  delta=self.mix(((adjacent(detail)-detail[:,:,None])*self.kernel*affinity).sum(2))
  self.relative_change=(delta.square().mean()/(detail.square().mean()+1e-12)).sqrt().detach()
  return context+F.adaptive_avg_pool2d(delta,context.shape[-2:]),detail+delta

def install(backbone,module):
 def prepare(m,args):
  x=args[0];mean=x.new_tensor([123.675,116.28,103.53])[None,:,None,None];std=x.new_tensor([58.395,57.12,57.375])[None,:,None,None]
  module.rgb=((x*std+mean)/255).clamp(0,1)
 backbone.register_forward_pre_hook(prepare)
 backbone.spatial.register_forward_hook(lambda m,a,o:module(*o),prepend=True)
