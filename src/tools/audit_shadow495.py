from pathlib import Path
import sys,json,torch
from torch.nn import functional as F
from shadow_redistribution495 import ShadowRedistribution
sys.path.insert(0,'/home/scv/shared/phase493_20261109')
from shadow_redistribution493 import ShadowRedistribution as Old
torch.set_num_threads(4);torch.manual_seed(48001);old=Old(True,False,False,False)
native=torch.randn(2,384,8,8);detail=torch.randn(2,64,32,32);rgb=torch.rand(2,3,128,128);target=torch.randn(2,384,32,32);common=None
for slot in [0,3]:
 torch.manual_seed(48001);m=ShadowRedistribution(feedback_slot=slot);m.prepare_rgb(rgb)
 assert sum(p.numel() for p in m.parameters())==43137
 assert all(torch.equal(t,m.state_dict()[n]) for n,t in old.state_dict().items())
 if common is not None:assert all(torch.equal(t,common[n]) for n,t in m.named_parameters())
 common={n:p.clone() for n,p in m.named_parameters()}
 x=native.flatten(2).transpose(1,2)
 assert torch.equal(m.feedback(x,slot,8,8),x) and torch.count_nonzero(m(native,detail))==0
 assert m.feedback(x,(slot+1)%4,8,8) is x
 initial={n:p.clone() for n,p in m.named_parameters()};opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
 for _ in range(4):
  m.prepare_rgb(rgb);v=m.feedback(x,slot,8,8).transpose(1,2).reshape_as(native)
  pred=F.interpolate(v,size=(32,32),mode='bilinear',align_corners=False)+m(v,detail)
  opt.zero_grad();loss=(pred-target).square().mean();loss.backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
  assert m.feedback_inject.weight.grad.abs().max()>0;opt.step()
 assert all(not torch.equal(initial[n],p) for n,p in m.named_parameters())
 m.prepare_rgb(rgb);assert not torch.equal(m.feedback(x,slot,8,8),x)
 m.feedback_enabled=False;assert m.feedback(x,slot,8,8) is x
 reference=Old(True,False,False,False);reference.load_state_dict({n:m.state_dict()[n] for n in old.state_dict()},strict=True);reference.rgb=rgb
 assert torch.allclose(m(native,detail),reference(native,detail),atol=1e-6,rtol=1e-6)
 m.enabled=False;m.prepare_rgb(rgb);assert m.ordinal_field is None and m.feedback(x,slot,8,8) is x and torch.count_nonzero(m(native,detail))==0
result=dict(passed=True,parameters=43137,extra_vs_493B=6144,parameter_tensors=22,old_initial_parameters_identical=True,AB_initial_parameters_identical=True,zero_initial_output=True,all_tensors_updated=True,feedback_gradient_nonzero=True,slot_routing_correct=True,disabled_identity=True,cache_equivalent_to_old_path=True)
Path('/home/scv/shared/phase495_20261109/UNIT_ACCEPTANCE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
