"""Content/chroma matched brighter references in the native MsRE projector input."""
import torch
from torch import nn
from torch.nn import functional as F

def neighbors(x):
 b,c,h,w=x.shape
 u=F.unfold(x,3,dilation=2,padding=2).reshape(b,c,9,h*w)
 return torch.cat([u[:,:,:4],u[:,:,5:]],2)

class ContrastProjector(nn.Module):
 def __init__(self):
  super().__init__()
  self.embed=nn.Conv2d(384,16,1,bias=False)
  self.log_temperature=nn.Parameter(torch.tensor(1.))
  self.log_chroma=nn.Parameter(torch.tensor(1.))
  self.log_brightness=nn.Parameter(torch.tensor(0.))
  self.register_buffer('strength',torch.tensor(0.))
  self.enabled=True;self.appearance_enabled=True;self.rgb=None;self.relative_changes=[]

 def forward(self,x):
  if not self.enabled or self.strength.item()==0:return x
  b,c,h,w=x.shape;n=h*w
  z=self.embed(F.layer_norm(x.permute(0,2,3,1),(384,)).permute(0,3,1,2))
  q=F.normalize(z.flatten(2),dim=1);k=F.normalize(neighbors(z),dim=1)
  score=(q[:,:,None]*k).sum(1)*(1+F.softplus(self.log_temperature))
  rgb=F.adaptive_avg_pool2d(self.rgb,(h,w));chroma=rgb/(rgb.sum(1,keepdim=True)+.02)
  luminance=(rgb*rgb.new_tensor([.2126,.7152,.0722])[None,:,None,None]).sum(1,keepdim=True)
  if self.appearance_enabled:
   color=(neighbors(chroma)-chroma.flatten(2)[:,:,None]).abs().sum(1)
   light=(neighbors((luminance+.02).log())-(luminance+.02).log().flatten(2)[:,:,None])[:,0].tanh()
   score=score-F.softplus(self.log_chroma)*color+F.softplus(self.log_brightness)*light
  valid=neighbors(torch.ones_like(x[:,:1]))[:,0]>0
  weight=score.masked_fill(~valid,-1e4).softmax(1)
  reference=(neighbors(x)*weight[:,None]).sum(2).reshape_as(x)
  delta=self.strength*(x-reference)
  self.relative_changes.append((delta.square().mean()/(x.square().mean()+1e-12)).sqrt().detach())
  return x+delta

def install(backbone,module):
 def prepare(m,args):
  x=args[0];mean=x.new_tensor([123.675,116.28,103.53])[None,:,None,None];std=x.new_tensor([58.395,57.12,57.375])[None,:,None,None]
  module.rgb=((x*std+mean)/255).clamp(0,1);module.relative_changes=[]
 backbone.register_forward_pre_hook(prepare)
 backbone.refinement.projector.register_forward_pre_hook(lambda m,a:(module(a[0]),))
