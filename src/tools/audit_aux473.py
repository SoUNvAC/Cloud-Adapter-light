import json,torch
from pathlib import Path
from shadow_contrast473 import ContrastProjector
torch.manual_seed(47301);torch.set_num_threads(2)
m=ContrastProjector();m.rgb=torch.rand(2,3,32,32);m.strength.fill_(.25)
x=torch.randn(2,384,4,4);m(x)
y=torch.randint(0,3,(2,32,32));y[:,:8]=255
loss=m.auxiliary(y);assert torch.isfinite(loss) and loss>0
loss.backward();assert m.classes.weight.grad.abs().sum()>0
m.class_logits=[torch.zeros(2,3,4,4)]*4
assert abs(m.auxiliary(y).item()-torch.tensor(3.).log().item())<1e-6
assert m.auxiliary(torch.full_like(y,255)).item()==0
r=dict(passed=True,uniform_CE_is_log3=True,all_ignore_zero=True,class_head_aux_gradient=True,ignore255_excluded=True)
Path('/home/scv/shared/phase473_20261107/AUX_AUDIT.json').write_text(json.dumps(r,indent=2));print(r)
