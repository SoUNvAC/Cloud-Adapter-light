from pathlib import Path
import sys,json,torch
sys.path.insert(0,'/home/scv/shared/phase431_20261102/code')
from phase431.network import ContextInteraction
from shadow_routing465 import RegionRouting
torch.set_num_threads(4);torch.manual_seed(46501)
layer=ContextInteraction().requires_grad_(False);r=RegionRouting();x=torch.randn(2,64,384)*8;ctx=torch.randn(2,64,8,8)*8;y=torch.randint(0,3,(2,128,128));y[:,:16]=255
base=layer(x,ctx);out=r(layer,x,ctx);assert torch.equal(out,base)
r.gain.fill_(.5);out=r(layer,x,ctx);assert (out-base).abs().max()>1e-5
loss=out.square().mean()+r.auxiliary(y);loss.backward();assert all(p.grad is not None and p.grad.abs().sum()>0 and torch.isfinite(p.grad).all() for p in r.parameters())
r.enabled=False;assert torch.equal(r(layer,x,ctx),base)
r.enabled=True;r.class_enabled=False;normalized=r(layer,x,ctx);r.class_mode.fill_(False);r.class_enabled=True;assert torch.equal(r(layer,x,ctx),normalized)
assert r.auxiliary(torch.full_like(y,255)).item()==0
out=dict(passed=True,parameters_four_layers=4*sum(p.numel() for p in r.parameters()),zero_gain_exact=True,nonzero_gradients=True,active_path_changes_output=True,disabled_exact=True,class_off_matches_control=True,all_ignore_zero_loss=True)
Path('/home/scv/shared/phase465_20261107/MODULE_AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
