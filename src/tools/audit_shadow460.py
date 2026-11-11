import json,hashlib
from pathlib import Path
import torch
from shadow_reference460 import GroundReferenceDecoder
torch.set_num_threads(4);torch.manual_seed(46001)
m=GroundReferenceDecoder().cuda();xs=[torch.randn(2,384,s,s,device='cuda') for s in [128,64,32,16]]
ys=torch.randint(0,3,(2,512,512),device='cuda');ys[:,0]=255
out=m(xs);assert all(torch.equal(a,b) for a,b in zip(out,xs))
optimizer=torch.optim.AdamW(m.parameters(),lr=5e-5);grads={n:0. for n,p in m.named_parameters()}
for step in range(3):
 out=m(xs);loss=sum(x.square().mean() for x in out)+m.auxiliary(ys);assert torch.isfinite(loss)
 optimizer.zero_grad();loss.backward()
 for n,p in m.named_parameters():
  assert p.grad is not None and torch.isfinite(p.grad).all(),n
  grads[n]+=float(p.grad.abs().sum())
 optimizer.step()
assert all(v>0 for v in grads.values()),grads
m(xs);assert m.auxiliary(torch.full_like(ys,255)).item()==0
m.enabled=False;assert all(torch.equal(a,b) for a,b in zip(m(xs),xs))
r=dict(passed=True,zero_initialization_exact=True,all_parameters_received_gradient=True,ignore_all_loss_zero=True,disabled_exact=True,parameters=sum(p.numel() for p in m.parameters()),code_sha=hashlib.sha256(Path(__file__).with_name('shadow_reference460.py').read_bytes()).hexdigest())
Path('/home/scv/shared/phase460_20261106/ARCHITECTURE_AUDIT.json').write_text(json.dumps(r,indent=2));print(r)
