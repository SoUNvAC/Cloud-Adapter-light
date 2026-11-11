from pathlib import Path
import json,torch
from shadow_projection467 import OcclusionProjection,ring
torch.set_num_threads(4);torch.manual_seed(46701)
m=OcclusionProjection();m.rgb=torch.rand(2,3,128,128);x=torch.randn(2,384,8,8);target=torch.randn_like(x)
assert m(x).count_nonzero()==0
assert torch.allclose(ring(torch.ones(1,2,8,8),9,5),torch.ones(1,2,8,8))
opt=torch.optim.AdamW(m.parameters(),lr=.001)
for i in range(2):
 opt.zero_grad();loss=(m(x)-target).square().mean();loss.backward()
 if i==1:assert all(p.grad is not None and p.grad.abs().sum()>0 and torch.isfinite(p.grad).all() for p in m.parameters())
 opt.step()
y=m(x);m.ring_enabled=False;assert (y-m(x)).abs().max()>0;m.enabled=False;assert m(x).count_nonzero()==0
out=dict(passed=True,new_parameters=sum(p.numel() for p in m.parameters()),zero_init=True,all_new_parameters_receive_gradients=True,ring_changes_output=True,disabled_zero=True,boundary_counts_correct=True)
Path('/home/scv/shared/phase467_20261107/MODULE_AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
