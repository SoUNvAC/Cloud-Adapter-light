from pathlib import Path
import sys,json,torch
from shadow_redistribution491 import ShadowRedistribution
sys.path.insert(0,'/home/scv/shared/phase488_20261108')
from shadow_redistribution488 import ShadowRedistribution as Old
torch.set_num_threads(4);torch.manual_seed(48001);old=Old(True,False,False)
native=torch.randn(2,384,8,8);deep=torch.randn_like(native);detail=torch.randn(2,64,32,32);target=torch.randn(2,384,32,32);rgb=torch.rand(2,3,128,128)
common=None;outputs=[]
for cross_depth in [True,False]:
 torch.manual_seed(48001);m=ShadowRedistribution(True,False,False,cross_depth);m.rgb=rgb;m.deep=deep
 assert sum(p.numel() for p in m.parameters())==42625
 assert all(torch.equal(t,m.state_dict()[n]) for n,t in old.state_dict().items())
 if common is not None:assert all(torch.equal(t,common[n]) for n,t in m.named_parameters())
 common={n:p.clone() for n,p in m.named_parameters()}
 assert torch.count_nonzero(m(native,detail))==0
 initial={n:p.clone() for n,p in m.named_parameters()};opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
 for _ in range(4):
  opt.zero_grad();loss=(m(native,detail)-target).square().mean();loss.backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());opt.step()
 assert all(not torch.equal(initial[n],p) for n,p in m.named_parameters())
 full=m(native,detail);outputs.append(full.clone());m.deep=deep.flip(0);assert (full-m(native,detail)).abs().max()>0;m.deep=deep
 m.cross_depth_enabled=False;assert (full-m(native,detail)).abs().max()>0
 m.enabled=False;assert torch.count_nonzero(m(native,detail))==0
assert (outputs[0]-outputs[1]).abs().max()>1e-6
result=dict(passed=True,parameters=42625,extra_vs_488B=19088,new_parameter_tensors=22,old_initial_parameters_identical=True,AB_initial_parameters_identical=True,zero_initial_output=True,all_tensors_updated=True,deep_feature_sensitive=True,interaction_control_different=True)
Path('/home/scv/shared/phase491_20261108/UNIT_ACCEPTANCE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
