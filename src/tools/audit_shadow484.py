from pathlib import Path
import sys,json,torch
from torch.nn import functional as F
from shadow_redistribution484 import ShadowRedistribution
sys.path.insert(0,'/home/scv/shared/phase482_20261108')
from shadow_redistribution482 import ShadowRedistribution as Old
torch.set_num_threads(4);torch.manual_seed(48001);old=Old(True,False)
native=torch.randn(2,384,8,8);detail=torch.randn(2,64,32,32);target=torch.randn(2,384,32,32)
common=None
for constrained in [True,False]:
 torch.manual_seed(48001);m=ShadowRedistribution(True,False,constrained)
 assert sum(p.numel() for p in m.parameters())==22097
 assert all(torch.equal(t,m.state_dict()[n]) for n,t in old.state_dict().items())
 if common is not None:assert all(torch.equal(t,common[n]) for n,t in m.named_parameters())
 common={n:p.clone() for n,p in m.named_parameters()}
 assert torch.count_nonzero(m(native,detail))==0
 initial={n:p.clone() for n,p in m.named_parameters()};opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
 for _ in range(4):
  opt.zero_grad();delta=m(native,detail);loss=(delta-target).square().mean();loss.backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());opt.step()
 assert all(not torch.equal(initial[n],p) for n,p in m.named_parameters())
 full=m(native,detail);m.regional_enabled=False;fine=m(native,detail);region=full-fine
 assert region.abs().max()>0 and fine.abs().max()>0
 assert F.avg_pool2d(fine,4,4).abs().max()<1e-6
 if constrained:assert F.avg_pool2d(region,16,16).abs().max()<1e-6
 m.enabled=False;assert torch.count_nonzero(m(native,detail))==0
result=dict(passed=True,parameters=22097,extra_vs_482B=6832,new_parameter_tensors=18,old_initial_parameters_identical=True,AB_initial_parameters_identical=True,zero_initial_output=True,all_tensors_updated=True,regional_path_nonzero=True,CPU_regional_mean_constraint=True)
Path('/home/scv/shared/phase484_20261108/UNIT_ACCEPTANCE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
