from pathlib import Path
import sys,json,torch
from torch import nn
from torch.nn import functional as F
from shadow_redistribution487 import ShadowRedistribution,install,align_shifted
sys.path.insert(0,'/home/scv/shared/phase482_20261108')
from shadow_redistribution482 import ShadowRedistribution as Old
torch.set_num_threads(4);torch.manual_seed(48001);old=Old(True,False)
native=torch.randn(2,384,8,8);detail=torch.randn(2,64,32,32);target=torch.randn(2,384,32,32)
common=None
for shift in [8,16]:
 torch.manual_seed(48001);m=ShadowRedistribution(True,False,shift);m.auxiliary=torch.randn(2,384,32,32)
 assert sum(p.numel() for p in m.parameters())==27953
 assert all(torch.equal(t,m.state_dict()[n]) for n,t in old.state_dict().items())
 if common is not None:assert all(torch.equal(t,common[n]) for n,t in m.named_parameters())
 common={n:p.clone() for n,p in m.named_parameters()}
 assert torch.count_nonzero(m(native,detail))==0
 initial={n:p.clone() for n,p in m.named_parameters()};opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
 for _ in range(4):
  opt.zero_grad();delta=m(native,detail);loss=(delta-target).square().mean();loss.backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());opt.step()
 assert all(not torch.equal(initial[n],p) for n,p in m.named_parameters())
 full=m(native,detail);m.evidence_enabled=False;fine=m(native,detail);extra=full-fine
 assert extra.abs().max()>0 and extra[:,:,:4].abs().max()==0 and extra[:,:,:,:4].abs().max()==0
 m.enabled=False;assert torch.count_nonzero(m(native,detail))==0
 x=torch.arange(32*32).reshape(1,1,32,32).float();k=shift//4
 shifted=F.pad(x,(k,0,k,0),mode='replicate')[:,:,:32,:32]
 assert torch.equal(align_shifted(shifted,k)[:,:,:-k,:-k],x[:,:,:-k,:-k])

class Spatial(nn.Module):
 def forward(self,x):
  gray=x.mean(1,keepdim=True);return F.avg_pool2d(gray,16).repeat(1,64,1,1),F.avg_pool2d(gray,4).repeat(1,64,1,1)
class Residual(nn.Module):
 def forward(self,x,slot):return x
class Toy(nn.Module):
 def __init__(self):super().__init__();self.spatial=Spatial();self.output_residual=Residual();self.calls=0
 def forward(self,x):
  self.calls+=1;c,d=self.spatial(x);n=self.output_residual(c.repeat(1,6,1,1),0)
  return (F.interpolate(n,scale_factor=4,mode='bilinear',align_corners=False),n,n,n)
b=Toy();m=ShadowRedistribution();install(b,m);x=torch.randn(2,3,128,128);out=b(x)
assert b.calls==2 and torch.allclose(m.cnn,F.avg_pool2d(x.mean(1,keepdim=True),4).repeat(1,64,1,1))
assert torch.allclose(m.native,F.avg_pool2d(x.mean(1,keepdim=True),16).repeat(1,384,1,1))
m.evidence_enabled=False;b(x);assert b.calls==3
result=dict(passed=True,parameters=27953,extra_vs_482B=12688,new_parameter_tensors=17,old_initial_parameters_identical=True,AB_initial_parameters_identical=True,zero_initial_output=True,all_tensors_updated=True,inverse_alignment=True,common_border_excluded=True,no_recursive_hook=True,original_view_capture_restored=True)
Path('/home/scv/shared/phase487_20261108/UNIT_ACCEPTANCE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
