from pathlib import Path
import json,torch
from shadow_material476 import MaterialConvolution
torch.set_num_threads(2);torch.manual_seed(47601)
for chroma in [False,True]:
 m=MaterialConvolution(chroma).cuda();m.rgb=torch.rand(2,3,64,64,device='cuda')*.6+.2
 d=torch.randn(2,64,16,16,device='cuda');c=torch.randn(2,64,4,4,device='cuda')
 out=m(c,d);assert torch.equal(out[0],c) and torch.equal(out[1],d)
 g=m.guide(m.rgb,(16,16));scaled=m.guide(m.rgb*.6,(16,16))
 if chroma:assert torch.allclose(g,scaled,atol=1e-6)
 else:assert not torch.allclose(g,scaled)
 opt=torch.optim.Adam(m.parameters(),lr=.001)
 for _ in range(3):
  opt.zero_grad();out=m(c,d);loss=out[0].square().mean()+out[1].square().mean();loss.backward();opt.step()
 assert all(p.grad is not None and p.grad.abs().sum()>0 and torch.isfinite(p.grad).all() for p in m.parameters())
 m.enabled=False;out=m(c,d);assert torch.equal(out[0],c) and torch.equal(out[1],d)
 m.enabled=True;ones=torch.ones_like(d);assert torch.equal(m(c,ones)[1],ones)
r=dict(passed=True,new_parameters=sum(p.numel() for p in m.parameters()),zero_kernel_exact=True,constant_features_preserved=True,all_parameter_gradients=True,gray_gain_chroma_guide_invariant=True,note='Gray gain test avoids clipping/floor; RGB not physical reflectance')
Path('/home/scv/shared/phase476_20261107/MODULE_AUDIT.json').write_text(json.dumps(r,indent=2));print(r)
