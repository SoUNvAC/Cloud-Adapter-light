"""Centered material relation residual and optional positive-light support gate."""
import os,math
import torch
import torch.nn.functional as F
from phase421_material import PhysicsMsRE as OriginalMsRE
from phase411_msre import LayerwiseMsRE
MODES=['material_control','centered','centered_supported']

def anchor(z):
 a=z.clone();a[:,16:32]=a[:,:16];a[:,32:53]=0
 return a

def support(z):
 # Descriptor channel48 is log(reference luminance / query luminance).
 return ((z[:,48:49]-math.log(1.05))/math.log(1.5)).clamp(0,1)

class PhysicsMsRE(OriginalMsRE):
 def __init__(self,mode=None,**kwargs):
  variant=mode or os.environ['PHASE413_MODE'];assert variant in MODES
  super().__init__(mode='material',**kwargs);self.variant=variant
 def forward(self,features,block_index,height,width):
  if self.variant=='material_control':return super().forward(features,block_index,height,width)
  out=LayerwiseMsRE.forward(self,features,block_index,height,width)
  if not self.enabled or block_index!=self.injection_indices[-1]:return out
  x=F.layer_norm(features[:,1:],(self.embed_dims,))
  feature=self.local_projection(x).transpose(1,2).reshape(x.shape[0],16,height,width)
  rgb=F.adaptive_avg_pool2d(self.raw_rgb,(height,width)).clamp_min(0)
  z,confidence=self.context(feature,rgb)
  delta=self.local_output(z.flatten(2).transpose(1,2))-self.local_output(anchor(z).flatten(2).transpose(1,2))
  if self.variant=='centered_supported':confidence=confidence*support(z)
  delta=delta*confidence.flatten(2).transpose(1,2)
  return torch.cat([out[:,:1],out[:,1:]+self.scale[-1]*delta],1)

def self_test():
 torch.set_num_threads(4);records={};states=[]
 for mode in MODES:
  torch.manual_seed(4090101);base=LayerwiseMsRE(embed_dims=8,token_length=4)
  torch.manual_seed(4090101);model=PhysicsMsRE(mode=mode,embed_dims=8,token_length=4)
  assert all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items())
  model.raw_rgb=torch.rand(2,3,16,16);x=torch.randn(2,65,8)
  for i in [2,5,8,11]:torch.testing.assert_close(model(x,i,8,8),base(x,i,8,8),rtol=0,atol=0)
  if mode=='material_control':
   torch.manual_seed(4090101);old=OriginalMsRE(mode='material',embed_dims=8,token_length=4);old.raw_rgb=model.raw_rgb
   assert all(torch.equal(v,old.state_dict()[k]) for k,v in model.state_dict().items())
   torch.testing.assert_close(model(x,11,8,8),old(x,11,8,8),rtol=0,atol=0)
  with torch.no_grad():model.local_output[-1].weight.normal_(0,.01);model.local_output[-1].bias.fill_(.37)
  if mode!='material_control':
   z=torch.randn(2,54,8,8);a=anchor(z)
   assert torch.equal(anchor(a),a)
   f=lambda d:model.local_output(d.flatten(2).transpose(1,2))
   assert torch.count_nonzero(f(a)-f(anchor(a)))==0
   assert torch.count_nonzero(support(a))==0
   z[:,48]=math.log(2);assert torch.all(support(z)==1)
   z[:,48]=math.log(.5);assert torch.count_nonzero(support(z))==0
  model.zero_grad();model(x,11,8,8).square().sum().backward()
  assert model.local_output[-1].weight.grad.abs().sum()>0
  assert model.local_projection.weight.grad.abs().sum()>0
  if mode!='material_control':assert torch.count_nonzero(model.local_output[-1].bias.grad)==0
  model.enabled=False;assert torch.equal(model(x,11,8,8),x)
  torch.manual_seed(4090101);prod=PhysicsMsRE(mode=mode);states.append(prod.state_dict())
  records[mode]={'passed':True,'parameters':sum(p.numel() for p in prod.parameters()),'zero_initial_identity':True,'zero_relation_zero_residual':mode!='material_control','zero_or_negative_light_support_closed':mode=='centered_supported','algebraically_cancelled_output_bias_parameters':384 if mode!='material_control' else 0}
 assert all(torch.equal(v,s[k]) for s in states[1:] for k,v in states[0].items())
 return records
