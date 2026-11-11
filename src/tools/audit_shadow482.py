from pathlib import Path
import json,torch
from torch.nn import functional as F
from shadow_redistribution482 import ShadowRedistribution
torch.set_num_threads(4);torch.manual_seed(48001)
native=torch.randn(2,384,8,8);detail=torch.randn(2,64,32,32);y=torch.randint(0,3,(2,128,128));target=torch.randn(2,384,32,32)
for contrast in [False,True]:
 m=ShadowRedistribution(True,contrast)
 assert sum(p.numel() for p in m.parameters())==15265
 assert torch.count_nonzero(m(native,detail))==0
 assert m.auxiliary(torch.full_like(y,255)).item()==0
 initial={n:p.clone() for n,p in m.named_parameters()};opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
 for _ in range(3):
  opt.zero_grad();delta=m(native,detail);loss=(delta-target).square().mean()+(.1 if contrast else 0)*m.auxiliary(y);loss.backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());opt.step()
 assert all(not torch.equal(initial[n],p) for n,p in m.named_parameters())
 assert F.avg_pool2d(m(native,detail),4,4).abs().max()<1e-6
 if contrast:
  with torch.no_grad():m.shadow.weight.zero_();m.shadow.bias.fill_(1.)
  assert m(native,detail).abs().max()<1e-7
  m.contrast_enabled=False;assert m(native,detail).abs().max()>1e-7
 m.enabled=False;assert torch.count_nonzero(m(native,detail))==0
out=dict(passed=True,parameters=15265,zero_initial_output=True,uniform_coverage_zero_contrast=True,patch_mean_constraint_cpu=True,ignored_labels_zero_loss=True,all_trainable_tensors_updated=True)
Path('/home/scv/shared/phase482_20261108/UNIT_ACCEPTANCE.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
