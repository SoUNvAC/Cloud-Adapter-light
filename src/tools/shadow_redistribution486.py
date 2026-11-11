"""Shadow-fraction-guided subpatch detail with optional patch-mean conservation."""
import torch
from torch import nn
from torch.nn import functional as F

class ShadowRedistribution(nn.Module):
 def __init__(self,conserve=True,contrast=False,adaptive=True):
  super().__init__();self.register_buffer('conserve_mode',torch.tensor(conserve,dtype=torch.bool))
  self.register_buffer('contrast_mode',torch.tensor(contrast,dtype=torch.bool));self.contrast_enabled=True
  self.detail=nn.Conv2d(64,16,1);self.semantic=nn.Conv2d(384,16,1)
  self.mix=nn.Sequential(nn.Conv2d(32,32,1),nn.GELU(),nn.Conv2d(32,32,3,padding=1,groups=32),nn.GELU(),nn.Conv2d(32,16,1),nn.GELU())
  self.shadow=nn.Conv2d(16,1,1);self.inject=nn.Conv2d(16,384,1,bias=False);nn.init.zeros_(self.inject.weight)
  self.register_buffer('adaptive_mode',torch.tensor(adaptive,dtype=torch.bool))
  self.evidence=nn.Sequential(nn.Conv2d(12,16,3,padding=1,bias=False),nn.GELU(),nn.Conv2d(16,16,3,padding=1,groups=16,bias=False),nn.GELU(),nn.Conv2d(16,16,1,bias=False),nn.GELU())
  self.evidence_inject=nn.Conv2d(16,384,1,bias=False);nn.init.zeros_(self.evidence_inject.weight)
  self.evidence_enabled=True;self.rgb=None;self.last_direction=None
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
   cue=self.consensus_evidence(self.rgb,native)
   field=F.interpolate(self.evidence(cue),size=detail.shape[-2:],mode='bilinear',align_corners=False)
   delta=delta+self.evidence_inject(field)
  self.relative_change=(delta.square().mean()/(base.square().mean()+1e-12)).sqrt().detach()
  self.mean_residual=F.avg_pool2d(delta.detach(),4,4).abs().max()
  return delta
 def consensus_evidence(self,rgb,native):
  with torch.no_grad():
   rgb=F.avg_pool2d(rgb,8,8);x=(rgb+.02).log();b,c,h,w=x.shape
   semantic=F.normalize(F.interpolate(native,size=(h,w),mode='bilinear',align_corners=False),dim=1)
   xp=F.pad(x,(4,4,4,4),mode='replicate');sp=F.pad(semantic,(4,4,4,4),mode='replicate')
   vp=F.pad(torch.ones_like(x[:,:1]),(4,4,4,4));deltas=[];similarities=[];weights=[]
   for dy,dx in [(-4,-4),(-4,0),(-4,4),(0,-4),(0,4),(4,-4),(4,0),(4,4)]:
    d=xp[:,:,4+dy:4+dy+h,4+dx:4+dx+w]-x
    sim=(sp[:,:,4+dy:4+dy+h,4+dx:4+dx+w]*semantic).sum(1,keepdim=True).clamp(0,1)
    valid=vp[:,:,4+dy:4+dy+h,4+dx:4+dx+w]
    norm=d.square().sum(1,keepdim=True).sqrt();chroma=d-d.mean(1,keepdim=True)
    weight=valid*sim.pow(4)*torch.sigmoid(8*d.mean(1,keepdim=True))*torch.exp(-3*chroma.square().mean(1,keepdim=True))*norm/(norm+.05)
    deltas.append(d*valid);similarities.append(sim);weights.append(weight)
   d=torch.stack(deltas,1);sim=torch.stack(similarities,1);weight=torch.stack(weights,1)
   default=F.normalize(x.new_ones((b,3)),dim=1)
   direction=(d*weight).sum((1,3,4))/weight.sum((1,3,4)).clamp_min(1e-8)
   length=direction.norm(dim=1,keepdim=True);direction=torch.where(length>1e-5,F.normalize(direction,dim=1),default)
   direction=torch.where(direction.sum(1,keepdim=True)>=0,direction,-direction)
   if not bool(self.adaptive_mode):direction=default
   self.last_direction=direction.detach()
   parallel=(d*direction[:,None,:,None,None]).sum(2,keepdim=True)
   residual=(d-parallel*direction[:,None,:,None,None]).square().sum(2,keepdim=True).clamp_min(0).sqrt()
   norm=d.square().sum(2,keepdim=True).sqrt();chroma=(d-d.mean(2,keepdim=True)).square().mean(2,keepdim=True).sqrt()
   cue=torch.cat([parallel,residual,d.mean(2,keepdim=True),chroma,sim*norm,weight*norm],2)
   return torch.cat([cue.mean(1),cue.max(1).values],1)
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
