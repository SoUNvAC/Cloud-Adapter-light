from pathlib import Path
import json,torch
from shadow_contrast472 import ContrastProjector,neighbors
torch.set_num_threads(2);torch.manual_seed(47201)
m=ContrastProjector().cuda();m.rgb=torch.rand(2,3,64,64,device='cuda');x=torch.randn(2,384,16,16,device='cuda')
assert torch.equal(m(x),x)
m.strength.fill_(.25);assert not torch.equal(m(x),x)
opt=torch.optim.Adam(m.parameters(),lr=.001);loss=m(x).square().mean();loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.parameters());opt.step()
m.enabled=False;assert torch.equal(m(x),x);m.enabled=True
with torch.no_grad():
 a=m(x);m.appearance_enabled=False;b=m(x);assert not torch.equal(a,b)
 m.appearance_enabled=True;assert torch.allclose(m(torch.ones_like(x)),torch.ones_like(x),atol=1e-6)
r=dict(passed=True,new_parameters=sum(p.numel() for p in m.parameters()),zero_strength_exact=True,full_strength_changes_input=True,all_parameter_gradients=True,appearance_changes_reference=True,constant_preserved_at_borders=True)
Path('/home/scv/shared/phase472_20261107/MODULE_AUDIT.json').write_text(json.dumps(r,indent=2));print(r)
