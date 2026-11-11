from pathlib import Path
import sys,json,torch
from shadow_redistribution493 import ShadowRedistribution
sys.path.insert(0,'/home/scv/shared/phase488_20261108')
from shadow_redistribution488 import ShadowRedistribution as Old
torch.set_num_threads(4);torch.manual_seed(48001);old=Old(True,False,False)
native=torch.randn(2,384,8,8);detail=torch.randn(2,64,32,32);target=torch.randn(2,384,32,32);rgb=torch.rand(2,3,128,128)
common=None;cues=[]
for ordinal in [True,False]:
 torch.manual_seed(48001);m=ShadowRedistribution(True,False,False,ordinal);m.rgb=rgb
 assert sum(p.numel() for p in m.parameters())==36993
 assert all(torch.equal(t,m.state_dict()[n]) for n,t in old.state_dict().items())
 if common is not None:assert all(torch.equal(t,common[n]) for n,t in m.named_parameters())
 common={n:p.clone() for n,p in m.named_parameters()}
 assert torch.count_nonzero(m(native,detail))==0
 cue=m.ordinal_evidence(rgb);cues.append(cue)
 assert cue.shape==(2,48,32,32) and torch.isfinite(cue).all()
 assert m.ordinal_evidence(torch.zeros_like(rgb)).abs().max()==0
 assert m.ordinal_evidence(torch.ones_like(rgb)*.3).abs().max()==0
 assert torch.allclose(cue,torch.cat([m.ordinal_evidence(rgb[i:i+1]) for i in range(2)]),atol=1e-6)
 initial={n:p.clone() for n,p in m.named_parameters()};opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
 for _ in range(4):
  opt.zero_grad();loss=(m(native,detail)-target).square().mean();loss.backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());opt.step()
 assert all(not torch.equal(initial[n],p) for n,p in m.named_parameters())
 full=m(native,detail);m.ordinal_enabled=False;assert (full-m(native,detail)).abs().max()>0
 m.enabled=False;assert torch.count_nonzero(m(native,detail))==0
assert (cues[0]-cues[1]).abs().max()>.01
# Positive affine illumination change preserves order, without clipping or quantization.
# Use grayscale integer steps to avoid accidental near-equality from float arithmetic.
y=(torch.arange(128*128).reshape(1,1,128,128)%251).float()/300
x=y.expand(-1,3,-1,-1)
m=ShadowRedistribution(ordinal=True)
assert torch.equal(m.ordinal_evidence(x),m.ordinal_evidence(.5*x+.1))
m.ordinal_mode.fill_(False)
assert (m.ordinal_evidence(x)-m.ordinal_evidence(.5*x+.1)).abs().max()>.01
result=dict(passed=True,parameters=36993,extra_vs_488B=13456,new_parameter_tensors=21,old_initial_parameters_identical=True,AB_initial_parameters_identical=True,zero_initial_output=True,all_tensors_updated=True,constant_and_black_finite=True,per_image_independent=True,positive_affine_order_preserved=True,control_different=True)
Path('/home/scv/shared/phase493_20261109/UNIT_ACCEPTANCE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
