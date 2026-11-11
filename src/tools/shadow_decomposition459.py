"""Cloud/content versus occlusion decomposition before Mask2Former pixel decoder."""
import torch
from torch import nn
from torch.nn import functional as F
class CloudShadowDecomposition(nn.Module):
 def __init__(self):
  super().__init__()
  self.content=nn.ModuleList([nn.Sequential(nn.Conv2d(384,24,1),nn.GELU(),nn.Conv2d(24,24,3,padding=1,groups=24),nn.GELU()) for _ in range(4)])
  self.occlusion=nn.ModuleList([nn.Sequential(nn.Conv2d(384,24,1),nn.GELU(),nn.Conv2d(24,24,3,padding=1,groups=24),nn.GELU()) for _ in range(4)])
  self.cloud=nn.ModuleList([nn.Conv2d(24,1,1) for _ in range(4)])
  self.shadow=nn.ModuleList([nn.Conv2d(24,1,1) for _ in range(4)])
  self.reconstruct=nn.ModuleList([nn.Conv2d(48,384,1) for _ in range(4)])
  self.inject=nn.ModuleList([nn.Conv2d(72,384,1) for _ in range(4)])
  for m in self.inject:nn.init.zeros_(m.weight);nn.init.zeros_(m.bias)
  self.enabled=True;self.cache=[]
 def forward(self,features):
  self.cache=[];out=[]
  for i,x in enumerate(features):
   c=self.content[i](x);s=self.occlusion[i](x);cl=self.cloud[i](c);sh=self.shadow[i](s)
   # Occlusion is conditioned softly on content; no cloud-neighbour requirement.
   cross=c*torch.sigmoid(s)
   residual=self.inject[i](torch.cat([c,s,cross],1))
   out.append(x+residual if self.enabled else x)
   if torch.is_grad_enabled():self.cache.append((cl,sh,self.reconstruct[i](torch.cat([c,s],1)),x.detach()))
  return out
 def auxiliary(self,y):
  total=y.new_zeros((),dtype=torch.float32)
  valid=(y!=255)[:,None].float();noncloud=((y!=255)&(y!=1))[:,None].float()
  for cl,sh,reconstruction,x in self.cache:
   size=cl.shape[-2:];v=F.adaptive_avg_pool2d(valid,size);nc=F.adaptive_avg_pool2d(noncloud,size)
   cloud=F.adaptive_avg_pool2d((y==1)[:,None].float(),size)/v.clamp_min(1e-6)
   shadow=F.adaptive_avg_pool2d((y==2)[:,None].float(),size)/nc.clamp_min(1e-6)
   total=total+.1*(F.binary_cross_entropy_with_logits(cl,cloud,reduction='none')*v).sum()/v.sum().clamp_min(1)
   total=total+.1*(F.binary_cross_entropy_with_logits(sh,shadow,reduction='none')*nc).sum()/nc.sum().clamp_min(1)
   total=total+.01*(((reconstruction-x).square().mean(1,keepdim=True)/(x.square().mean(1,keepdim=True)+1e-4))*v).sum()/v.sum().clamp_min(1)
  return total/4
 def hook(self,module,args):return (self(args[0]),)+args[1:]
