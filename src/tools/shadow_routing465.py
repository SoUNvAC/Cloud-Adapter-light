"""Scale-controlled context reading, with optional class-balanced values."""
import types
import torch
from torch import nn
from torch.nn import functional as F

class RegionRouting(nn.Module):
 def __init__(self):
  super().__init__();self.query_region=nn.Linear(384,3);self.key_region=nn.Conv2d(64,3,1)
  self.relation=nn.Parameter(torch.eye(3)[None].repeat(8,1,1)*2)
  self.register_buffer('gain',torch.tensor(0.));self.register_buffer('class_mode',torch.tensor(True))
  self.enabled=True;self.class_enabled=True;self.cache=None
 def forward(self,original,patches,context):
  b,n,c=patches.shape;h,w=context.shape[-2:];assert n==h*w
  qr=self.query_region(F.layer_norm(patches,(c,)));kr=self.key_region(context)
  if torch.is_grad_enabled():self.cache=(qr.transpose(1,2).reshape(b,3,h,w),kr)
  def split(x):return x.reshape(b,-1,original.heads,original.head_dim).transpose(1,2)
  flat=context.flatten(2).transpose(1,2);q,k,v=split(original.q(patches)),split(original.k(flat)),split(original.v(flat))
  base=F.scaled_dot_product_attention(q,k,v)
  if not self.enabled or self.gain.item()==0:return patches+original.out(base.transpose(1,2).flatten(2))
  score=q@k.transpose(-1,-2)/original.head_dim**.5
  # Row centering removes a softmax-invariant shift. Scaling prevents saturation.
  score=score-score.mean(-1,keepdim=True);scaled=score/score.std(-1,keepdim=True,unbiased=False).clamp_min(1)
  attention=scaled.softmax(-1)
  if bool(self.class_mode) and self.class_enabled:
   kp=kr.flatten(2).softmax(1);qp=qr.softmax(-1)
   mixture=torch.einsum('bnc,hcd->bhnd',qp,self.relation.softmax(-1))
   routed=torch.zeros_like(base)
   for cls in range(3):
    weights=attention*kp[:,None,cls,None,:]
    value=(weights@v)/weights.sum(-1,keepdim=True).clamp_min(1e-6)
    routed=routed+mixture[:,:,:,cls,None]*value
  else:routed=attention@v
  read=(1-self.gain)*base+self.gain*routed
  return patches+original.out(read.transpose(1,2).flatten(2))
 def auxiliary(self,y):
  valid=(y!=255)[:,None].float();onehot=torch.stack([(y==c).float() for c in range(3)],1);total=valid.sum()*0
  for z in self.cache:
   coverage=F.adaptive_avg_pool2d(valid,z.shape[-2:]);target=F.adaptive_avg_pool2d(onehot,z.shape[-2:])/coverage.clamp_min(1e-6)
   total=total+(-(target*z.log_softmax(1)).sum(1,keepdim=True)*coverage).sum()/coverage.sum().clamp_min(1)
  return .1*total/2

def install(backbone,routes):
 for original,route in zip(backbone.interactions,routes):
  def build(r):
   def forward(layer,patches,context):return r(layer,patches,context)
   return forward
  original.forward=types.MethodType(build(route),original)
