from pathlib import Path
import json,torch
from shadow_envelope479 import ShadowEnvelope,closing,average
torch.set_num_threads(4);torch.manual_seed(47901)
x=torch.ones(1,3,41,41);x[:,:,19:22,19:22]=.2
assert torch.allclose(closing(x,5),torch.ones_like(x))
assert average(x,5)[0,0,20,20]<1
a=ShadowEnvelope(False);b=ShadowEnvelope(True);b.load_state_dict({**a.state_dict(),'morphology_mode':torch.tensor(True)})
rgb=torch.rand(2,3,128,128)*.3+.2;detail=torch.randn(2,64,32,32);context=torch.randn(2,64,8,8)
assert torch.allclose(b.descriptor(rgb,(32,32)),b.descriptor(rgb*.7,(32,32)),atol=2e-6)
for m in [a,b]:
 m.rgb=rgb;c,d=m(context,detail);assert torch.equal(c,context) and torch.equal(d,detail)
 optimizer=torch.optim.AdamW(m.parameters(),lr=1e-3)
 initial={n:p.clone() for n,p in m.named_parameters()}
 for _ in range(3):
  optimizer.zero_grad();c,d=m(context,detail);loss=(d-.1).square().mean();loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());optimizer.step()
 assert all(not torch.equal(initial[n],p) for n,p in m.named_parameters())
 assert not torch.equal(m(context,detail)[1],detail)
 m.enabled=False;assert torch.equal(m(context,detail)[1],detail)
out=dict(passed=True,new_parameters=sum(p.numel() for p in a.parameters()),zero_initial_residual=True,dark_valley_fixture=True,no_clipping_exposure_descriptor_invariance=True,gradient_and_update=True)
Path('/home/scv/shared/phase479_20261108/UNIT_ACCEPTANCE.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
