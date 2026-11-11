from pathlib import Path
import json,torch
from torch.nn import functional as F
from shadow_redistribution480 import ShadowRedistribution
torch.set_num_threads(4);torch.manual_seed(48001)
native=torch.randn(2,384,8,8);detail=torch.randn(2,64,32,32);y=torch.randint(0,3,(2,128,128));target=torch.randn(2,384,32,32)
a=ShadowRedistribution(False);b=ShadowRedistribution(True);b.load_state_dict({**a.state_dict(),'conserve_mode':torch.tensor(True)})
for m in [a,b]:
 assert torch.count_nonzero(m(native,detail))==0
 assert torch.isfinite(m.auxiliary(y))
 assert m.auxiliary(torch.full_like(y,255)).item()==0
 initial={n:p.clone() for n,p in m.named_parameters()};optimizer=torch.optim.AdamW(m.parameters(),lr=1e-3)
 for _ in range(3):
  optimizer.zero_grad();delta=m(native,detail);loss=(delta-target).square().mean()+.1*m.auxiliary(y);loss.backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());optimizer.step()
 assert all(not torch.equal(initial[n],p) for n,p in m.named_parameters())
 delta=m(native,detail);assert torch.count_nonzero(delta)>0
 if bool(m.conserve_mode):assert F.avg_pool2d(delta,4,4).abs().max()<1e-6
 m.enabled=False;assert torch.count_nonzero(m(native,detail))==0
out=dict(passed=True,new_parameters=sum(p.numel() for p in a.parameters()),new_parameter_tensors=len(list(a.parameters())),zero_initial_residual=True,patch_mean_conservation=True,ignored_label_loss_zero=True,gradient_and_update=True)
Path('/home/scv/shared/phase480_20261108/UNIT_ACCEPTANCE.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
