"""Shadow-fraction-guided subpatch detail with optional patch-mean conservation."""
import torch
from torch import nn
from torch.nn import functional as F

class ShadowRedistribution(nn.Module):
 def __init__(self,conserve=True,contrast=False,normalized=False,ordinal=False,feedback_slot=0):
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
  self.register_buffer('feedback_slot',torch.tensor(feedback_slot,dtype=torch.int64))
  self.feedback_inject=nn.Conv2d(16,384,1,bias=False);nn.init.zeros_(self.feedback_inject.weight)
  self.feedback_enabled=True;self.ordinal_field=None;self.feedback_relative=0.;self.feedback_calls=0
  self.enabled=True;self.cnn=None;self.native=None;self.shadow_logits=None;self.relative_change=None;self.mean_residual=None
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
   cue=self.texture_evidence(self.rgb,native)
   field=F.interpolate(self.evidence(cue),size=detail.shape[-2:],mode='bilinear',align_corners=False)
   delta=delta+self.evidence_inject(field)
   if self.ordinal_enabled:
    delta=delta+self.ordinal_inject(self.ordinal_field)
  self.relative_change=(delta.square().mean()/(base.square().mean()+1e-12)).sqrt().detach()
  self.mean_residual=F.avg_pool2d(delta.detach(),4,4).abs().max()
  return delta
 def prepare_rgb(self,rgb):
  self.rgb=rgb;self.feedback_calls=0;self.feedback_relative=0.
  self.ordinal_field=self.ordinal(self.ordinal_evidence(rgb)) if self.enabled and self.evidence_enabled and self.ordinal_enabled else None
 def feedback(self,out,slot,h,w):
  if not self.enabled or not self.feedback_enabled or self.ordinal_field is None or slot!=int(self.feedback_slot):return out
  delta=self.feedback_inject(F.adaptive_avg_pool2d(self.ordinal_field,(h,w))).flatten(2).transpose(1,2)
  # Match token units without differentiating through the scale estimate.
  scale=out.detach().float().square().mean(-1,keepdim=True).sqrt().clamp_min(1e-4)
  update=delta*scale
  self.feedback_relative=(update.detach().square().mean()/(out.detach().square().mean()+1e-12)).sqrt()
  self.feedback_calls+=1
  return out+update
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

def install(backbone,module):
 def rgb_hook(m,args):
  x=args[0];mean=x.new_tensor([123.675,116.28,103.53])[None,:,None,None];std=x.new_tensor([58.395,57.12,57.375])[None,:,None,None]
  module.prepare_rgb(((x*std+mean)/255).clamp(0,1))
 backbone.register_forward_pre_hook(rgb_hook)
 backbone.refinement.register_forward_hook(lambda m,args,out:module.feedback(out,args[2],args[3],args[4]))
 backbone.spatial.register_forward_hook(lambda m,a,o:setattr(module,'cnn',o[1]))
 def native(m,args,out):
  if args[1]==0:module.native=out
 backbone.output_residual.register_forward_hook(native)
 backbone.register_forward_hook(lambda m,a,o:(o[0]+module(module.native,module.cnn),*o[1:]),prepend=True)
