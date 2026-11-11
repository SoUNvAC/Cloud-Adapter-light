from pathlib import Path
import json,torch
from shadow_conditional474 import ConditionalOcclusion
torch.set_num_threads(2);torch.manual_seed(47401)
x=torch.randn(2,3,64,64,device='cuda');d=torch.randn(2,2,64,64,device='cuda',requires_grad=True)
a=ConditionalOcclusion.compose(x,d,True)
assert torch.allclose(a.exp().sum(1),torch.ones_like(x[:,0]),atol=1e-6)
g=torch.autograd.grad(a.exp()[:,1].sum(),d)[0];assert torch.count_nonzero(g[:,1])==0 and g[:,0].abs().sum()>0
for mode in [False,True]:
 m=ConditionalOcclusion(mode).cuda();detail=torch.randn(2,64,16,16,device='cuda');features=[torch.randn(2,384,k,k,device='cuda') for k in [16,8,4,2]];rgb=torch.rand(2,3,64,64,device='cuda')
 out=m(detail,features,rgb,x);assert torch.allclose(out.softmax(1),x.softmax(1),atol=1e-6)
 opt=torch.optim.Adam(m.parameters(),lr=.001);y=torch.randint(0,3,(2,64,64),device='cuda')
 for _ in range(3):
  opt.zero_grad();loss=torch.nn.functional.cross_entropy(m(detail,features,rgb,x),y);loss.backward();opt.step()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.parameters())
 m.enabled=False;assert torch.equal(m(detail,features,rgb,x),x)
r=dict(passed=True,new_parameters=sum(p.numel() for p in m.parameters()),initial_probability_identity=True,conditional_normalized=True,shadow_path_cloud_probability_derivative_zero=True,all_parameter_gradients=True,disabled_exact_identity=True)
Path('/home/scv/shared/phase474_20261107/MODULE_AUDIT.json').write_text(json.dumps(r,indent=2));print(r)
