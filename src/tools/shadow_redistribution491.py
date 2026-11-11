"""Shadow-fraction-guided subpatch detail with optional patch-mean conservation."""
import torch
from torch import nn
from torch.nn import functional as F

class ShadowRedistribution(nn.Module):
 def __init__(self,conserve=True,contrast=False,normalized=False,cross_depth=True):
  super().__init__();self.register_buffer('conserve_mode',torch.tensor(conserve,dtype=torch.bool))
  self.register_buffer('contrast_mode',torch.tensor(contrast,dtype=torch.bool));self.contrast_enabled=True
  self.detail=nn.Conv2d(64,16,1);self.semantic=nn.Conv2d(384,16,1)
  self.mix=nn.Sequential(nn.Conv2d(32,32,1),nn.GELU(),nn.Conv2d(32,32,3,padding=1,groups=32),nn.GELU(),nn.Conv2d(32,16,1),nn.GELU())
  self.shadow=nn.Conv2d(16,1,1);self.inject=nn.Conv2d(16,384,1,bias=False);nn.init.zeros_(self.inject.weight)
  self.register_buffer('normalized_mode',torch.tensor(normalized,dtype=torch.bool))
  self.evidence=nn.Sequential(nn.Conv2d(12,16,3,padding=1,bias=False),nn.GELU(),nn.Conv2d(16,16,3,padding=1,groups=16,bias=False),nn.GELU(),nn.Conv2d(16,16,1,bias=False),nn.GELU())
  self.evidence_inject=nn.Conv2d(16,384,1,bias=False);nn.init.zeros_(self.evidence_inject.weight)
  self.evidence_enabled=True;self.rgb=None;self.last_direction=None
  self.register_buffer('cross_depth_mode',torch.tensor(cross_depth,dtype=torch.bool))
  self.shallow_project=nn.Conv2d(384,16,1,bias=False)
  self.deep_project=nn.Conv2d(384,16,1,bias=False)
  self.depth_mix=nn.Sequential(nn.Conv2d(32,16,1,bias=False),nn.GELU(),nn.Conv2d(16,16,3,padding=1,groups=16,bias=False),nn.GELU())
  self.depth_inject=nn.Conv2d(16,384,1,bias=False);nn.init.zeros_(self.depth_inject.weight)
  self.deep=None;self.cross_depth_enabled=True
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
   if self.cross_depth_enabled:
    assert self.deep is not None and self.deep.shape==native.shape
    normalize=lambda x:F.layer_norm(x.permute(0,2,3,1),(384,)).permute(0,3,1,2)
    shallow=self.shallow_project(normalize(native));deep=self.deep_project(normalize(self.deep))
    if bool(self.cross_depth_mode):cue=torch.cat([shallow-deep,shallow*deep],1)
    else:cue=torch.cat([shallow,deep],1)
    field=F.interpolate(self.depth_mix(cue),size=detail.shape[-2:],mode='bilinear',align_corners=False)
    delta=delta+self.depth_inject(field)
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
 def auxiliary(self,y):
  valid=F.avg_pool2d((y!=255)[:,None].float(),4,4)
  target=F.avg_pool2d((y==2)[:,None].float(),4,4)/valid.clamp_min(1e-6)
  assert target.shape==self.shadow_logits.shape
  return (F.binary_cross_entropy_with_logits(self.shadow_logits,target,reduction='none')*valid).sum()/valid.sum().clamp_min(1)

def install(backbone,module):
 def rgb_hook(m,args):
  x=args[0];mean=x.new_tensor([123.675,116.28,103.53])[None,:,None,None];std=x.new_tensor([58.395,57.12,57.375])[None,:,None,None]
  module.rgb=((x*std+mean)/255).clamp(0,1)
 backbone.register_forward_pre_hook(rgb_hook)
 backbone.spatial.register_forward_hook(lambda m,a,o:setattr(module,'cnn',o[1]))
 def native(m,args,out):
  if args[1]==0:module.native=out
  if args[1]==3:module.deep=out
 backbone.output_residual.register_forward_hook(native)
 backbone.register_forward_hook(lambda m,a,o:(o[0]+module(module.native,module.cnn),*o[1:]),prepend=True)
