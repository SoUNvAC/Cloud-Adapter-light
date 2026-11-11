from pathlib import Path
import json,hashlib
import torch
from query_loss462 import targets,matched_query_loss
torch.set_num_threads(4);torch.manual_seed(46201)
y=torch.zeros(2,64,64,dtype=torch.long,device='cuda');y[:,8:28,8:35]=1;y[:,35:58,17:50]=2;y[:,:8,:]=255
for split in [False,True]:
 labels,t,w=targets(y[0],(16,16),split)
 assert (t>=0).all() and (t<=1+1e-6).all()
 assert torch.allclose(t.sum(0)[w>0],torch.ones_like(w[w>0]),atol=1e-6)
 assert len(labels)==(4 if split else 3)
 cls=[torch.randn(2,25,4,device='cuda',requires_grad=True) for _ in range(3)]
 masks=[torch.randn(2,25,16,16,device='cuda',requires_grad=True) for _ in range(3)]
 loss,parts=matched_query_loss(cls,masks,y,split);assert torch.isfinite(loss);loss.backward()
 assert all(torch.isfinite(t.grad).all() and t.grad.abs().sum()>0 for t in cls+masks)
 assert all(m.grad[:,:,:2,:].abs().sum()==0 for m in masks)
 changed=[m.detach().clone() for m in masks]
 for m in changed:m[:,:,:2,:]=123
 repeated,_=matched_query_loss([t.detach() for t in cls],changed,y,split)
 assert torch.equal(loss.detach(),repeated)
 zero,_=matched_query_loss(cls,masks,torch.full_like(y,255),split);assert zero.item()==0
# Empty shadow and thin-only shadow must remain valid.
for truth in [torch.zeros_like(y),torch.full_like(y,255)]:
 for split in [False,True]:
  value,_=matched_query_loss(cls,masks,truth,split);assert torch.isfinite(value)
result=dict(passed=True,valid_occupancy_conserved=True,shadow_split_four_targets=True,finite_gradients=True,ignored_pixel_gradient_zero=True,ignored_prediction_changes_no_effect=True,all_ignore_zero=True,code_sha=hashlib.sha256(Path(__file__).with_name('query_loss462.py').read_bytes()).hexdigest())
Path('/home/scv/shared/phase462_20261106/LOSS_AUDIT.json').write_text(json.dumps(result,indent=2));print(result)
