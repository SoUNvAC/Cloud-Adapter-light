"""The sole Phase208 ordinary control; M imports immutable Phase207 code."""
import torch
from torch import nn
from phase207_msre_interface import ParallelMsREInterface,original_core,FrozenR4WithParallelMsRE

class OrdinaryCore(nn.Module):
 def __init__(self):
  super().__init__();self.convs=nn.ModuleList([nn.Conv2d(384,384,k,padding=k//2,groups=384) for k in (3,5,7)]);self.first=nn.Conv2d(384,384,1);self.gelu=nn.GELU();self.second=nn.Conv2d(384,384,1);self.scale=nn.Parameter(torch.tensor(1e-3))
 def forward(self,z):return self.scale*self.second(self.gelu(self.first(torch.stack([m(z) for m in self.convs],dim=0).mean(0))))

class OrdinaryInterface(nn.Module):
 def __init__(self):
  super().__init__();self.core=OrdinaryCore();channels=(48,104,208,440);self.input_projections=nn.ModuleList([nn.Conv2d(c,384,1) for c in channels]);self.output_projections=nn.ModuleList([nn.Conv2d(384,c,1,bias=False) for c in channels]);self.enabled=True
 def forward(self,features):
  if not self.enabled:return features
  result=list(features)
  for i in range(4):raw=features[len(features)-4+i];z=self.input_projections[i](raw);result[len(features)-4+i]=raw+self.output_projections[i](self.core(z))
  return result

def paired_modules(source_path):
 torch.manual_seed(20801);M=ParallelMsREInterface((48,104,208,440),original_core(source_path));torch.manual_seed(20801);B=OrdinaryInterface();shared=[]
 for family in ('input_projections','output_projections'):
  src=getattr(M,family);dst=getattr(B,family);dst.load_state_dict(src.state_dict(),strict=True)
  for k,v in src.state_dict().items():shared.append((family+'.'+k,v,dst.state_dict()[k]))
 B.core.convs.load_state_dict(M.core.convs.state_dict(),strict=True)
 for k,v in M.core.convs.state_dict().items():shared.append(('DW.'+k,v,B.core.convs.state_dict()[k]))
 assert all(torch.equal(a,b) and a.data_ptr()!=b.data_ptr() for _,a,b in shared)
 assert not ({p.data_ptr() for p in M.parameters()} & {p.data_ptr() for p in B.parameters()})
 assert sum(p.numel() for p in M.parameters())==970369 and sum(p.numel() for p in B.parameters())==944641
 assert all(not isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for mod in (M,B) for m in mod.modules())
 return M,B,[name for name,_,_ in shared]
