"""Shadow-fraction-guided subpatch detail with optional patch-mean conservation."""
import torch
from torch import nn
from torch.nn import functional as F

class ShadowRedistribution(nn.Module):
 def __init__(self,conserve=True,contrast=False,normalized=False,ordinal=True):
  super().__init__();self.register_buffer('conserve_mode',torch.tensor(conserve,dtype=torch.bool))
  self.register_buffer('contrast_mode',torch.tensor(contrast,dtype=torch.bool));self.contrast_enabled=True
  self.detail=nn.Conv2d(64,16,1);self.semantic=nn.Conv2d(384,16,1)
  self.mix=nn.Sequential(nn.Conv2d(32,32,1),nn.GELU(),nn.Conv2d(32,32,3,padding=1,groups=32),nn.GELU(),nn.Conv2d(32,16,1),nn.GELU())
  self.shadow=nn.Conv2d(16,1,1);self.inject=nn.Conv2d(16,384,1,bias=False);nn.init.zeros_(self.inject.weight)
  self.register_buffer('normalized_mode',torch.tensor(normalized,dtype=torch.bool))
  self.evidence=nn.Sequential(nn.Conv2d(12,16,3,padding=1,bias=False),nn.GELU(),nn.Conv2d(16,16,3,padding=1,groups=16,bias=False),nn.GELU(),nn.Conv2d(16,16,1,bias=False),nn.GELU())
  self.evidence_inject=nn.Conv2d(16,384,1,bias=False);nn.init.zeros_(self.evidence_inject.weight)
  self.evidence_enabled=True;self.rgb=None;self.last_direction=None
  self.register_buffer('ordinal_mode',torch.tensor(ordinal,dtype=torch.bool))
  self.ordinal=nn.Sequential(nn.Conv2d(48,16,3,padding=1,bias=False),nn.GELU(),nn.Conv2d(16,16,3,padding=1,groups=16,bias=False),nn.GELU(),nn.Conv2d(16,16,1,bias=False),nn.GELU())
  self.ordinal_inject=nn.Conv2d(16,384,1,bias=False);nn.init.zeros_(self.ordinal_inject.weight)
  self.ordinal_enabled=True
  with torch.random.fork_rng(devices=[]):
   torch.random.default_generator.manual_seed(49801)
   self.early_inject=nn.ModuleList([nn.Conv2d(32,64,1,bias=False) for _ in range(4)])
   for layer in self.early_inject:nn.init.zeros_(layer.weight)
  self.early_enabled=True;self.fields=None
  self.enabled=True;self.cnn=None;self.native=None;self.shadow_logits=None;self.relative_change=None;self.mean_residual=None
 def prepare_evidence(self):
  # Reuse exactly the same learned evidence in early context and late decoding.
  self.fields=(self.evidence(self.texture_evidence(self.rgb,None)),
               self.ordinal(self.ordinal_evidence(self.rgb)))
 def early_context(self,context,slot):
  if not self.enabled or not self.early_enabled:return context
  assert self.fields is not None
  code=F.adaptive_avg_pool2d(torch.cat(self.fields,1),context.shape[-2:])
  return context+self.early_inject[slot](code)
 def forward(self,native,detail):
  base=F.interpolate(native,size=detail.shape[-2:],mode='bilinear',align_corners=False)
  if not self.enabled:return torch.zeros_like(base)
  assert detail.shape[-2:]==(native.shape[-2]*4,native.shape[-1]*4)
  sem=self.semantic(F.layer_norm(native.permute(0,2,3,1),(384,)).permute(0,3,1,2))
  sem=F.interpolate(sem,size=detail.shape[-2:],mode='bilinear',align_corners=False)
  z=self.mix(torch.cat([self.detail(detail),sem],1));self.shadow_logits=self.shadow(z)
  p=self.shadow_logits.sigmoid()
  if bool(self.contrast_mode) and self.contrast_enabled:p=p-F.interpolate(F.avg_pool2d(p,4,4),scale_factor=4,mode='nearest')
  detail_code=z*p
  if bool(self.conserve_mode):detail_code=detail_code-F.interpolate(F.avg_pool2d(detail_code,4,4),scale_factor=4,mode='nearest')
  delta=self.inject(detail_code)
  if self.evidence_enabled:
   assert self.fields is not None, 'Call prepare_evidence for this image before interaction'
   field=F.interpolate(self.fields[0],size=detail.shape[-2:],mode='bilinear',align_corners=False)
   delta=delta+self.evidence_inject(field)
   if self.ordinal_enabled:
    delta=delta+self.ordinal_inject(self.fields[1])
  self.relative_change=(delta.square().mean()/(base.square().mean()+1e-12)).sqrt().detach()
  self.mean_residual=F.avg_pool2d(delta.detach(),4,4).abs().max()
  return delta
 def texture_evidence(self,rgb,native):
  # Display-RGB texture cue, not an assumed linear-reflectance inversion.
  with torch.no_grad():
   y=(rgb*rgb.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
   y=F.avg_pool2d(y,4,4)
   p=F.pad(y,(1,1,1,1),mode='replicate')
   gx=(p[:,:,1:-1,2:]-p[:,:,1:-1,:-2])*.5
   gy=(p[:,:,2:,1:-1]-p[:,:,:-2,1:-1])*.5
   def avg(t,k):return F.avg_pool2d(F.pad(t,(k//2,)*4,mode='replicate'),k,1)
   if bool(self.normalized_mode):
    denominator=avg(y,3)+.02;gx=gx/denominator;gy=gy/denominator
   cues=[]
   for k in [3,7]:
    xx=avg(gx.square(),k);yy=avg(gy.square(),k);xy=avg(gx*gy,k)
    energy=xx+yy;u=(xx-yy)/(energy+1e-5);v=2*xy/(energy+1e-5)
    # Compare the local texture tensor to its surrounding annulus.
    outer=2*k+1
    ring=lambda t:(avg(t,outer)*outer**2-avg(t,k)*k**2)/(outer**2-k**2)
    e=torch.log1p(100*energy)
    brightness=(y-ring(y))/(ring(y)+.02)
    cues.extend([u,v,e,u-ring(u),v-ring(v),brightness])
   return torch.cat(cues,1)
 def ordinal_evidence(self,rgb):
  # Compute microtexture BEFORE 4x pooling. A retains order; B retains bounded amplitude.
  # No segmentation threshold or learned decision rule is changed.
  with torch.no_grad():
   y=(rgb*rgb.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
   h,w=y.shape[-2:];cues=[]
   for radius in [1,2,4]:
    p=F.pad(y,(radius,)*4,mode='replicate')
    for dy,dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
     neighbor=p[:,:,radius+dy*radius:radius+dy*radius+h,radius+dx*radius:radius+dx*radius+w]
     diff=neighbor-y
     cue=diff.sign() if bool(self.ordinal_mode) else torch.tanh(20*diff)
     cues.append(F.avg_pool2d(cue,4,4))
   local=torch.cat(cues,1)
   def avg(x,k):return F.avg_pool2d(F.pad(x,(k//2,)*4,mode='replicate'),k,1)
   ring=(49*avg(local,7)-9*avg(local,3))/40
   return torch.cat([local,local-ring],1)
 def auxiliary(self,y):
  valid=F.avg_pool2d((y!=255)[:,None].float(),4,4)
  target=F.avg_pool2d((y==2)[:,None].float(),4,4)/valid.clamp_min(1e-6)
  assert target.shape==self.shadow_logits.shape
  return (F.binary_cross_entropy_with_logits(self.shadow_logits,target,reduction='none')*valid).sum()/valid.sum().clamp_min(1)
