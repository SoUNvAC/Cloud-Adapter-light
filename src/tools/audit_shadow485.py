from pathlib import Path
import sys,json,torch
from shadow_redistribution485 import ShadowRedistribution
sys.path.insert(0,'/home/scv/shared/phase482_20261108')
from shadow_redistribution482 import ShadowRedistribution as Old
torch.set_num_threads(4);torch.manual_seed(48001);old=Old(True,False)
native=torch.randn(2,384,8,8);detail=torch.randn(2,64,32,32);target=torch.randn(2,384,32,32);rgb=torch.rand(2,3,128,128)
common=None
for ordered in [True,False]:
 torch.manual_seed(48001);m=ShadowRedistribution(True,False,ordered);m.rgb=rgb
 assert sum(p.numel() for p in m.parameters())==29041
 assert all(torch.equal(t,m.state_dict()[n]) for n,t in old.state_dict().items())
 if common is not None:assert all(torch.equal(t,common[n]) for n,t in m.named_parameters())
 common={n:p.clone() for n,p in m.named_parameters()}
 assert torch.count_nonzero(m(native,detail))==0
 assert m.profile_evidence(torch.ones_like(rgb)*.3).abs().max()==0
 evidence=m.profile_evidence(rgb);assert evidence.abs().max()>0
 assert (evidence-m.profile_evidence(torch.roll(rgb,16,-1))).abs().max()>0
 initial={n:p.clone() for n,p in m.named_parameters()};opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
 for _ in range(4):
  opt.zero_grad();delta=m(native,detail);loss=(delta-target).square().mean();loss.backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());opt.step()
 assert all(not torch.equal(initial[n],p) for n,p in m.named_parameters())
 full=m(native,detail);m.evidence_enabled=False;fine=m(native,detail);assert (full-fine).abs().max()>0
 m.enabled=False;assert torch.count_nonzero(m(native,detail))==0
result=dict(passed=True,parameters=29041,extra_vs_482B=13776,new_parameter_tensors=17,old_initial_parameters_identical=True,AB_initial_parameters_identical=True,zero_initial_output=True,all_tensors_updated=True,constant_rgb_zero_evidence=True,spatial_input_sensitive=True)
Path('/home/scv/shared/phase485_20261108/UNIT_ACCEPTANCE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
