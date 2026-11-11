from pathlib import Path
import json,torch
from torch import nn
from shadow_detail468 import DetailBridge,bands,install
torch.set_num_threads(4);torch.manual_seed(46801)
assert bands(torch.ones(2,3,16,16)).abs().max()==0
shared={};counts={}
for frequency in [False,True]:
 torch.manual_seed(46801);m=DetailBridge(frequency);m.rgb=torch.rand(2,3,64,64);m.cnn=torch.randn(2,64,16,16);x=torch.randn(2,384,16,16);target=torch.randn_like(x)
 if not frequency:shared={n:p.detach().clone() for n,p in m.named_parameters()}
 else:assert all(torch.equal(p,dict(m.named_parameters())[n]) for n,p in shared.items())
 assert m(x).count_nonzero()==0
 opt=torch.optim.AdamW(m.parameters(),lr=.001)
 for i in range(2):
  opt.zero_grad();loss=(m(x)-target).square().mean();loss.backward()
  if i==1:assert all(p.grad is not None and p.grad.abs().sum()>0 and torch.isfinite(p.grad).all() for p in m.parameters())
  opt.step()
 y=m(x)
 if frequency:m.frequency_enabled=False;assert (y-m(x)).abs().max()>0
 m.enabled=False;assert m(x).count_nonzero()==0
 counts[str(frequency)]=sum(p.numel() for p in m.parameters())
class Spatial(nn.Module):
 def forward(self,x):return x,torch.randn(x.shape[0],64,16,16)
class Backbone(nn.Module):
 def __init__(self):super().__init__();self.spatial=Spatial()
 def forward(self,x):self.spatial(x);return (torch.ones(x.shape[0],384,16,16),)*4
b=Backbone();seen={};b.register_forward_hook(lambda m,a,o:seen.update(out=o))
m=DetailBridge(True);m.mix[-1].bias.data.fill_(.1);install(b,m);out=b(torch.rand(2,3,64,64))
assert torch.equal(out[0],seen['out'][0]) and torch.allclose(out[0],torch.full_like(out[0],1.1))
assert all(torch.equal(x,torch.ones_like(x)) for x in out[1:])
v=dict(passed=True,parameters=counts,constant_input_zero_bands=True,common_initialization_identical=True,zero_init_exact=True,all_parameter_gradients=True,frequency_changes_output=True,hook_capture_matches_decoder=True)
Path('/home/scv/shared/phase468_20261107/MODULE_AUDIT.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
