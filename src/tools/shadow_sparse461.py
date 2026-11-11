"""Sparse surface correspondence with observed RGB contrast, before pixel decoder."""
import torch
from torch import nn
from torch.nn import functional as F
class SparseOcclusionDecoder(nn.Module):
 def __init__(self):
  super().__init__()
  self.project=nn.ModuleList([nn.Sequential(nn.Conv2d(384,32,1),nn.GELU(),nn.Conv2d(32,32,3,padding=1,groups=32),nn.GELU()) for _ in range(4)])
  self.match=nn.ModuleList([nn.Conv2d(32,16,1,bias=False) for _ in range(4)])
  self.region=nn.ModuleList([nn.Conv2d(32,3,1) for _ in range(4)])
  self.fuse=nn.ModuleList([nn.Sequential(nn.Conv2d(41,64,1),nn.GELU(),nn.Conv2d(64,64,3,padding=1,groups=64),nn.GELU(),nn.Conv2d(64,384,1)) for _ in range(4)])
  for m in self.fuse:nn.init.zeros_(m[-1].weight);nn.init.zeros_(m[-1].bias)
  self.enabled=True;self.reference_enabled=True;self.reference_shuffle=False;self.rgb=None;self.cache=[]
 def forward(self,features):
  assert self.rgb is not None
  self.cache=[];outputs=[];rgb=self.rgb
  mean=F.avg_pool2d(F.pad(rgb,(2,2,2,2),mode='replicate'),5,stride=1)
  texture=((rgb-mean).square().mean(1,keepdim=True)+1e-6).sqrt()/(rgb.mean(1,keepdim=True)+.02)
  for i,x in enumerate(features):
   z=self.project[i](x);h,w=z.shape[-2:];size=(min(h,32),min(w,32));low=F.adaptive_avg_pool2d(z,size);memory=F.adaptive_avg_pool2d(low,(16,16))
   regional=self.region[i](memory);ground=regional.softmax(1)[:,0].flatten(1)
   q=F.normalize(self.match[i](low).flatten(2),dim=1).transpose(1,2);k=F.normalize(self.match[i](memory).flatten(2),dim=1)
   color=F.adaptive_avg_pool2d(rgb,size);chroma=color/(color.sum(1,keepdim=True)+.02);tx=F.adaptive_avg_pool2d(texture,size)
   cue=torch.cat([(color+.02).log(),chroma,(tx+.02).log()],1)
   bank=F.adaptive_avg_pool2d(cue,(16,16)).flatten(2)
   cq=chroma.flatten(2).transpose(1,2);ck=bank[:,3:6]
   color_distance=(cq.square().sum(-1,keepdim=True)+ck.square().sum(1)[:,None]-2*cq@ck).clamp_min(0)
   affinity=8*(q@k)+(.02+ground[:,None]).log()-4*color_distance
   values,indices=affinity.topk(8,dim=-1);attention=torch.zeros_like(affinity).scatter(-1,indices,values.softmax(-1))
   ref=(attention@memory.flatten(2).transpose(1,2)).transpose(1,2).reshape(low.shape)
   refcue=(attention@bank.transpose(1,2)).transpose(1,2).reshape(cue.shape)
   confidence=(attention@ground[:,:,None]).transpose(1,2).reshape(low.shape[0],1,*size)
   if self.reference_shuffle:
    ref=torch.roll(ref.flatten(2),17,-1).reshape_as(ref);refcue=torch.roll(refcue.flatten(2),17,-1).reshape_as(refcue);confidence=torch.roll(confidence.flatten(2),17,-1).reshape_as(confidence)
   if not self.reference_enabled:ref=low;refcue=cue;confidence=confidence*0
   mismatch=1-F.cosine_similarity(low,ref,dim=1)[:,None]
   # 32 semantic differences + 7 photometric differences + confidence + mismatch.
   pair=torch.cat([low-ref,cue-refcue,confidence,mismatch],1)
   pair=F.interpolate(pair,size=(h,w),mode='bilinear',align_corners=False)
   residual=self.fuse[i](pair);outputs.append(x+residual if self.enabled else x)
   if torch.is_grad_enabled():self.cache.append(regional)
  return outputs
 def auxiliary(self,y):
  valid=(y!=255)[:,None].float();total=valid.sum()*0;onehot=torch.stack([(y==c).float() for c in range(3)],1)
  for scores in self.cache:
   coverage=F.adaptive_avg_pool2d(valid,scores.shape[-2:]);fractions=F.adaptive_avg_pool2d(onehot,scores.shape[-2:])/coverage.clamp_min(1e-6)
   total=total+.1*(-(fractions*scores.log_softmax(1)).sum(1,keepdim=True)*coverage).sum()/coverage.sum().clamp_min(1)
  return total/4
 def hook(self,module,args):return (self(args[0]),)+args[1:]
