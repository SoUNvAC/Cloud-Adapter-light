"""Shadow-fraction-guided subpatch detail with optional patch-mean conservation."""
import torch
from torch import nn
from torch.nn import functional as F

class ShadowRedistribution(nn.Module):
 def __init__(self,conserve=True,contrast=False,shift_pixels=8):
  super().__init__();self.register_buffer('conserve_mode',torch.tensor(conserve,dtype=torch.bool))
  self.register_buffer('contrast_mode',torch.tensor(contrast,dtype=torch.bool));self.contrast_enabled=True
  self.detail=nn.Conv2d(64,16,1);self.semantic=nn.Conv2d(384,16,1)
  self.mix=nn.Sequential(nn.Conv2d(32,32,1),nn.GELU(),nn.Conv2d(32,32,3,padding=1,groups=32),nn.GELU(),nn.Conv2d(32,16,1),nn.GELU())
  self.shadow=nn.Conv2d(16,1,1);self.inject=nn.Conv2d(16,384,1,bias=False);nn.init.zeros_(self.inject.weight)
  assert shift_pixels in [8,16]
  self.register_buffer('shift_mode',torch.tensor(shift_pixels,dtype=torch.int64))
  self.phase_reduce=nn.Conv2d(384,16,1,bias=False)
  self.phase_mix=nn.Sequential(nn.Conv2d(16,16,3,padding=1,groups=16,bias=False),nn.GELU(),nn.Conv2d(16,16,1,bias=False),nn.GELU())
  self.phase_inject=nn.Conv2d(16,384,1,bias=False);nn.init.zeros_(self.phase_inject.weight)
  self.evidence_enabled=True;self.auxiliary=None
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
   assert self.auxiliary is not None and self.auxiliary.shape==base.shape
   difference=self.auxiliary-base
   field=self.phase_mix(self.phase_reduce(difference))
   extra=self.phase_inject(field)
   # Same 16px exclusion in both arms; padded view edges never inject.
   valid=torch.ones_like(extra[:,:1]);valid[:,:,:4]=0;valid[:,:,-4:]=0;valid[:,:,:,:4]=0;valid[:,:,:,-4:]=0
   delta=delta+extra*valid
  self.relative_change=(delta.square().mean()/(base.square().mean()+1e-12)).sqrt().detach()
  self.mean_residual=F.avg_pool2d(delta.detach(),4,4).abs().max()
  return delta
 def auxiliary(self,y):
  valid=F.avg_pool2d((y!=255)[:,None].float(),4,4)
  target=F.avg_pool2d((y==2)[:,None].float(),4,4)/valid.clamp_min(1e-6)
  assert target.shape==self.shadow_logits.shape
  return (F.binary_cross_entropy_with_logits(self.shadow_logits,target,reduction='none')*valid).sum()/valid.sum().clamp_min(1)

def align_shifted(feature,shift_cells):
 return F.pad(feature[:,:,shift_cells:,shift_cells:],(0,shift_cells,0,shift_cells),mode='replicate')

def install(backbone,module):
 def second_view(m,args):
  if not module.enabled or not module.evidence_enabled:return
  image=args[0];s=int(module.shift_mode);h,w=image.shape[-2:]
  shifted=F.pad(image,(s,0,s,0),mode='replicate')[:,:,:h,:w]
  # Direct forward bypasses this module's pre/post hooks. Submodule captures
  # are overwritten by the subsequent original-view forward before fusion.
  with torch.no_grad():other=m.forward(shifted)[0]
  module.auxiliary=align_shifted(other,s//4).detach()
 backbone.register_forward_pre_hook(second_view)
 backbone.spatial.register_forward_hook(lambda m,a,o:setattr(module,'cnn',o[1]))
 def native(m,args,out):
  if args[1]==0:module.native=out
 backbone.output_residual.register_forward_hook(native)
 backbone.register_forward_hook(lambda m,a,o:(o[0]+module(module.native,module.cnn),*o[1:]),prepend=True)
