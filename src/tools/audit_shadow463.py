from pathlib import Path
import json,torch
from shadow_spatial463 import AttenuationSpatialPrior
torch.set_num_threads(4);torch.manual_seed(46301)
m=AttenuationSpatialPrior();image=torch.randn(2,3,128,128);context=torch.randn(2,64,8,8);detail=torch.randn(2,64,32,32)
o=m(image,(context,detail));assert torch.equal(o[0],context) and torch.equal(o[1],detail)
y=torch.randint(0,3,(2,128,128));y[:,:16]=255
loss=sum(t.square().mean() for t in o)+m.auxiliary(y);loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
opt=torch.optim.AdamW(m.parameters(),lr=1e-3);opt.step();opt.zero_grad()
o=m(image,(context,detail));loss=sum(t.square().mean() for t in o)+m.auxiliary(y);loss.backward()
assert all(p.grad.abs().sum()>0 for p in m.parameters())
m.enabled=False;o=m(image,(context,detail));assert o[0] is context and o[1] is detail
m.enabled=True;m(image,(context,detail));assert m.auxiliary(torch.full_like(y,255)).item()==0
constant=torch.ones(1,3,128,128)*.3;cues=m.cues(constant,(32,32));assert cues[:,6:].abs().max()==0
out=dict(passed=True,parameters=sum(p.numel() for p in m.parameters()),zero_init=True,all_parameter_gradients=True,all_ignore_zero_loss=True,constant_image_directional_differences_zero=True)
Path('/home/scv/shared/phase463_20261106/MODULE_AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
