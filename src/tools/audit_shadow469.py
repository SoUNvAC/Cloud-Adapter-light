from pathlib import Path
import json,torch
from shadow_vfm469 import OcclusionAttenuation
torch.set_num_threads(4);counts={};shared={}
for relative in [False,True]:
 torch.manual_seed(46901);m=OcclusionAttenuation(relative);x=torch.randn(2,1029,384);qkv=torch.randn(2,1029,1152);target=torch.randn_like(qkv);y=torch.randint(0,3,(2,512,512));y[:,:40]=255
 if not relative:shared={n:p.detach().clone() for n,p in m.named_parameters()}
 else:assert all(torch.equal(p,dict(m.named_parameters())[n]) for n,p in shared.items())
 assert torch.equal(m(x,qkv),qkv);assert m.auxiliary(torch.full_like(y,255)).item()==0
 opt=torch.optim.AdamW(m.parameters(),lr=.001)
 for i in range(2):
  opt.zero_grad();out=m(x,qkv);loss=(out-target).square().mean()+.05*m.auxiliary(y);loss.backward()
  if i==1:assert all(p.grad is not None and p.grad.abs().sum()>0 and torch.isfinite(p.grad).all() for p in m.parameters())
  opt.step()
 out=m(x,qkv);assert torch.equal(out[:,:5],qkv[:,:5])
 if relative:m.relative_enabled=False;assert (out-m(x,qkv)).abs().max()>0
 m.enabled=False;assert torch.equal(m(x,qkv),qkv)
 counts[str(relative)]=4*sum(p.numel() for p in m.parameters())
v=dict(passed=True,parameters=counts,zero_init_exact=True,prefix_unchanged=True,all_ignore_zero_loss=True,all_parameter_gradients=True,common_initialization_identical=True,operator_changes_output=True)
Path('/home/scv/shared/phase469_20261107/MODULE_AUDIT.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
