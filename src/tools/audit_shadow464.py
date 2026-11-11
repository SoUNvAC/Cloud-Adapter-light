from pathlib import Path
import sys,json,torch
sys.path.insert(0,'/home/scv/shared/phase431_20261102/code')
from phase431.network import ContextInteraction
from shadow_routing464 import RegionRouting
torch.set_num_threads(4);torch.manual_seed(46401)
layer=ContextInteraction().requires_grad_(False);r=RegionRouting();x=torch.randn(2,64,384);ctx=torch.randn(2,64,8,8);y=torch.randint(0,3,(2,128,128));y[:,:16]=255
base=layer(x,ctx);out=r(layer,x,ctx);assert torch.allclose(out,base,atol=1e-6,rtol=1e-6)
loss=out.square().mean()+r.auxiliary(y);loss.backward()
assert all(p.grad is not None and p.grad.abs().sum()>0 and torch.isfinite(p.grad).all() for p in r.parameters())
with torch.no_grad():r.relation.normal_(0,1)
out=r(layer,x,ctx);assert (out-base).abs().max()>1e-5
r.enabled=False;off=r(layer,x,ctx);assert torch.allclose(off,base,atol=1e-6,rtol=1e-6)
assert r.auxiliary(torch.full_like(y,255)).item()==0
info=dict(passed=True,parameters_four_layers=4*sum(p.numel() for p in r.parameters()),zero_init=True,nonzero_gradients=True,routing_changes_attention=True,disabled_recovers_original=True,all_ignore_zero_loss=True)
Path('/home/scv/shared/phase464_20261107/MODULE_AUDIT.json').write_text(json.dumps(info,indent=2));print(json.dumps(info))
