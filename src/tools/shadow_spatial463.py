"""Directional RGB attenuation cues injected before CloudAdapter interactions."""
import torch
from torch import nn
from torch.nn import functional as F

class AttenuationSpatialPrior(nn.Module):
 def __init__(self):
  super().__init__()
  self.cue_encoder=nn.Sequential(nn.Conv2d(54,32,1),nn.GELU(),nn.Conv2d(32,32,3,padding=1,groups=32),nn.GELU())
  self.fuse=nn.Sequential(nn.Conv2d(96,64,1),nn.GELU(),nn.Conv2d(64,64,3,padding=3,dilation=3,groups=64),nn.GELU())
  self.region=nn.Conv2d(64,3,1)
  self.detail_out=nn.Conv2d(64,64,1)
  self.context_out=nn.Conv2d(64,64,1)
  for m in [self.detail_out,self.context_out]:nn.init.zeros_(m.weight);nn.init.zeros_(m.bias)
  self.enabled=True;self.cues_enabled=True;self.cache=None

 def cues(self,rgb,size):
  x=F.adaptive_avg_pool2d(rgb,size).clamp(1/255,1).log()
  lum=x.mean(1,keepdim=True);chroma=x-lum
  cues=[x,chroma]
  for r in [2,8,24]:
   pad=F.pad(x,(r,r,r,r),mode='replicate');h,w=x.shape[-2:]
   for dy,dx in [(0,-r),(0,r),(-r,0),(r,0)]:
    other=pad[:,:,r+dy:r+dy+h,r+dx:r+dx+w];diff=other-x
    common=diff.mean(1,keepdim=True)
    cues.extend([common,diff-common])
  out=torch.cat(cues,1)
  return out if self.cues_enabled else out*0

 def forward(self,image,outputs):
  context,detail=outputs
  if not self.enabled:return outputs
  mean=image.new_tensor([123.675,116.28,103.53])[None,:,None,None]
  std=image.new_tensor([58.395,57.12,57.375])[None,:,None,None]
  rgb=(image*std+mean)/255
  z=self.fuse(torch.cat([detail,self.cue_encoder(self.cues(rgb,detail.shape[-2:]))],1))
  if torch.is_grad_enabled():self.cache=self.region(z)
  return context+self.context_out(F.adaptive_avg_pool2d(z,context.shape[-2:])),detail+self.detail_out(z)

 def hook(self,module,args,outputs):return self(args[0],outputs)

 def auxiliary(self,y):
  logits=self.cache;valid=(y!=255)[:,None].float()
  coverage=F.adaptive_avg_pool2d(valid,logits.shape[-2:])
  target=F.adaptive_avg_pool2d(torch.stack([(y==c).float() for c in range(3)],1),logits.shape[-2:])/coverage.clamp_min(1e-6)
  return .1*(-(target*logits.log_softmax(1)).sum(1,keepdim=True)*coverage).sum()/coverage.sum().clamp_min(1)
