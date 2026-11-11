from pathlib import Path
import json,torch
from shadow_experts470 import ErrorExperts
torch.set_num_threads(2)
states={};params={}
for routed in [False,True]:
 torch.manual_seed(47001);m=ErrorExperts(routed).cuda();states[routed]={k:t.clone() for k,t in m.state_dict().items() if k!='routed_mode'}
 m.cnn=torch.randn(2,64,16,16,device='cuda');m.rgb=torch.rand(2,3,64,64,device='cuda')
 features=[torch.randn(2,384,s,s,device='cuda') for s in [16,8,4,2]]
 y=torch.randint(0,3,(2,64,64),device='cuda');y[:,:2]=255;ref=torch.randn(2,3,64,64,device='cuda')
 assert torch.equal(m(features),torch.zeros_like(features[0]))
 assert m.auxiliary(torch.full_like(y,255),ref).item()==0
 opt=torch.optim.Adam(m.parameters(),lr=.001)
 for step in range(2):
  z=m(features);loss=(z-1).square().mean()+m.auxiliary(y,ref);opt.zero_grad();loss.backward();opt.step()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.parameters())
 m.enabled=False;assert not m(features).any();m.enabled=True
 if routed:
  z=m(features);m.routing_enabled=False;assert not torch.equal(z,m(features))
 params[routed]=sum(p.numel() for p in m.parameters())
assert all(torch.equal(t,states[True][k]) for k,t in states[False].items())
y=torch.tensor([[[0,1,2,2,0,255]]]);pred=torch.tensor([[[2,2,0,2,0,2]]]);ref=torch.nn.functional.one_hot(pred,3).permute(0,3,1,2).float()
assert ErrorExperts.targets(y,ref).tolist()==[[[2,3,1,0,0,255]]]
r=dict(passed=True,parameters=params,zero_init_exact=True,all_ignore_zero_loss=True,error_targets_correct=True,all_parameter_gradients=True,common_initialization_identical=True,routing_changes_output=True)
Path('/home/scv/shared/phase470_20261107/MODULE_AUDIT.json').write_text(json.dumps(r,indent=2));print(r)
