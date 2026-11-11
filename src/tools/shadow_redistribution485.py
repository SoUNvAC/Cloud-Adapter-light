"""Shadow-fraction-guided subpatch detail with optional patch-mean conservation."""
import torch
from torch import nn
from torch.nn import functional as F

class ShadowRedistribution(nn.Module):
 def __init__(self,conserve=True,contrast=False,ordered=True):
  super().__init__();self.register_buffer('conserve_mode',torch.tensor(conserve,dtype=torch.bool))
  self.register_buffer('contrast_mode',torch.tensor(contrast,dtype=torch.bool));self.contrast_enabled=True
  self.detail=nn.Conv2d(64,16,1);self.semantic=nn.Conv2d(384,16,1)
  self.mix=nn.Sequential(nn.Conv2d(32,32,1),nn.GELU(),nn.Conv2d(32,32,3,padding=1,groups=32),nn.GELU(),nn.Conv2d(32,16,1),nn.GELU())
  self.shadow=nn.Conv2d(16,1,1);self.inject=nn.Conv2d(16,384,1,bias=False);nn.init.zeros_(self.inject.weight)
  self.register_buffer('ordered_mode',torch.tensor(ordered,dtype=torch.bool))
  self.profile=nn.Sequential(nn.Conv3d(6,8,(3,1,1),padding=(1,0,0),bias=False),nn.GELU(),nn.Conv3d(8,8,(3,1,1),padding=(1,0,0),bias=False),nn.GELU())
  self.profile_fuse=nn.Sequential(nn.Conv2d(72,16,1,bias=False),nn.GELU())
  self.evidence_inject=nn.Conv2d(32,384,1,bias=False);nn.init.zeros_(self.evidence_inject.weight)
  self.evidence_enabled=True;self.rgb=None
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
   field=self.profile_evidence(self.rgb)
   field=F.interpolate(field,size=detail.shape[-2:],mode='bilinear',align_corners=False)
   delta=delta+self.evidence_inject(field)
  self.relative_change=(delta.square().mean()/(base.square().mean()+1e-12)).sqrt().detach()
  self.mean_residual=F.avg_pool2d(delta.detach(),4,4).abs().max()
  return delta
 def profile_evidence(self,rgb):
  assert rgb is not None
  rgb=F.avg_pool2d(rgb,8,8);logrgb=(rgb+.02).log()
  lum=(rgb*rgb.new_tensor([.2126,.7152,.0722])[None,:,None,None]).sum(1,keepdim=True)
  cue=torch.cat([logrgb,(lum+.02).log(),logrgb[:,0:1]-logrgb[:,1:2],logrgb[:,2:3]-logrgb[:,1:2]],1)
  b,c,h,w=cue.shape;padded=F.pad(cue,(4,4,4,4),mode='replicate');fields=[]
  for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]:
   samples=torch.stack([padded[:,:,4+s*dy:4+s*dy+h,4+s*dx:4+s*dx+w] for s in range(-4,5)],2)
   samples=samples-cue[:,:,None]
   if not bool(self.ordered_mode):samples=samples[:,:,[4,0,8,2,6,1,7,3,5]]
   enc=self.profile(samples).reshape(b,72,h,w);fields.append(self.profile_fuse(enc))
  fields=torch.stack(fields,1)
  return torch.cat([fields.mean(1),fields.max(1).values],1)
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
 backbone.output_residual.register_forward_hook(native)
 backbone.register_forward_hook(lambda m,a,o:(o[0]+module(module.native,module.cnn),*o[1:]),prepend=True)
