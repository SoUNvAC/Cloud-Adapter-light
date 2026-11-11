from pathlib import Path
import json,torch
from shadow_transport471 import ContourTransport,distance_target
torch.set_num_threads(2);states={};params={}
for normal in [False,True]:
 torch.manual_seed(47101);m=ContourTransport(normal).cuda();states[normal]={k:t.clone() for k,t in m.state_dict().items() if k!='normal_mode'}
 m.cnn=torch.randn(2,64,16,16,device='cuda');m.rgb=torch.rand(2,3,64,64,device='cuda')
 features=[torch.randn(2,384,s,s,device='cuda') for s in [16,8,4,2]]
 y=torch.zeros((2,64,64),device='cuda',dtype=torch.long);y[:,16:48,16:48]=2
 assert torch.equal(m(features),torch.zeros_like(features[0]))
 assert m.auxiliary(torch.full_like(y,255)).item()==0
 target,valid=distance_target(y,(16,16));assert (target[:,0,8,8]>0).all() and (target[:,0,0,0]<0).all() and valid.all()
 for value,sign in [(0,-1),(2,1)]:
  t,v=distance_target(torch.full_like(y,value),(16,16));assert (t==sign).all() and v.all()
 opt=torch.optim.Adam(m.parameters(),lr=.001)
 for step in range(2):
  z=m(features);loss=(z-torch.randn_like(z)).square().mean()+m.auxiliary(y);opt.zero_grad();loss.backward();opt.step()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.parameters())
 assert m.flow.square().sum(1).sqrt().max()<=2.00001
 m.enabled=False;assert not m(features).any();m.enabled=True
 if normal:
  z=m(features);m.normal_enabled=False;assert not torch.equal(z,m(features))
 params[normal]=sum(p.numel() for p in m.parameters())
assert all(torch.equal(t,states[True][k]) for k,t in states[False].items())
r=dict(passed=True,parameters=params,zero_init_exact=True,all_ignore_zero_loss=True,distance_sign_and_empty_cases=True,all_parameter_gradients=True,common_initialization_identical=True,normal_projection_changes_output=True,max_flow_grid=2)
Path('/home/scv/shared/phase471_20261107/MODULE_AUDIT.json').write_text(json.dumps(r,indent=2));print(r)
