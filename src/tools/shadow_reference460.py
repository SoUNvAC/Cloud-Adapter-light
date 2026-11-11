"""Image-wide clear-ground reference attention inside a segmentation decoder."""
import torch
from torch import nn
from torch.nn import functional as F
class GroundReferenceDecoder(nn.Module):
 def __init__(self):
  super().__init__()
  self.project=nn.ModuleList([nn.Sequential(nn.Conv2d(384,32,1),nn.GELU(),nn.Conv2d(32,32,3,padding=1,groups=32),nn.GELU()) for _ in range(4)])
  self.match=nn.ModuleList([nn.Conv2d(32,16,1,bias=False) for _ in range(4)])
  self.region=nn.ModuleList([nn.Conv2d(32,3,1) for _ in range(4)])
  self.fuse=nn.ModuleList([nn.Sequential(nn.Conv2d(161,64,1),nn.GELU(),nn.Conv2d(64,64,3,padding=1,groups=64),nn.GELU(),nn.Conv2d(64,384,1)) for _ in range(4)])
  for m in self.fuse:nn.init.zeros_(m[-1].weight);nn.init.zeros_(m[-1].bias)
  self.enabled=True;self.reference_enabled=True;self.cache=[]
 def forward(self,features):
  self.cache=[];outputs=[]
  for i,x in enumerate(features):
   z=self.project[i](x);h,w=z.shape[-2:];low=F.adaptive_avg_pool2d(z,(min(h,32),min(w,32)));memory=F.adaptive_avg_pool2d(low,(16,16))
   regional=self.region[i](memory);p=regional.softmax(1);ground=p[:,0].flatten(1);cloud=p[:,1:2]
   q=F.normalize(self.match[i](low).flatten(2),dim=1).transpose(1,2);k=F.normalize(self.match[i](memory).flatten(2),dim=1)
   affinity=8*(q@k)+(.02+ground[:,None]).log()
   attention=affinity.softmax(-1)
   ref=(attention@memory.flatten(2).transpose(1,2)).transpose(1,2).reshape(low.shape)
   confidence=(attention@ground[:,:,None]).transpose(1,2).reshape(low.shape[0],1,*low.shape[-2:])
   if not self.reference_enabled:ref=low;confidence=confidence*0
   ref=F.interpolate(ref,size=(h,w),mode='bilinear',align_corners=False);confidence=F.interpolate(confidence,size=(h,w),mode='bilinear',align_corners=False)
   context=(memory*cloud).sum((2,3),keepdim=True)/cloud.sum((2,3),keepdim=True).clamp_min(1e-6)
   joint=torch.cat([z,ref,z-ref,z*ref,context.expand(-1,-1,h,w),confidence],1)
   residual=self.fuse[i](joint);outputs.append(x+residual if self.enabled else x)
   if torch.is_grad_enabled():self.cache.append(regional)
  return outputs
 def auxiliary(self,y):
  valid=(y!=255)[:,None].float();total=valid.sum()*0
  onehot=torch.stack([(y==c).float() for c in range(3)],1)
  for scores in self.cache:
   coverage=F.adaptive_avg_pool2d(valid,scores.shape[-2:]);fractions=F.adaptive_avg_pool2d(onehot,scores.shape[-2:])/coverage.clamp_min(1e-6)
   loss=-(fractions*scores.log_softmax(1)).sum(1,keepdim=True)
   total=total+.1*(loss*coverage).sum()/coverage.sum().clamp_min(1)
  return total/4
 def hook(self,module,args):return (self(args[0]),)+args[1:]
