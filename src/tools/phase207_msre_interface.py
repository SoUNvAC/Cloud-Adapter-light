"""One declared CNN-interface variant; original MsRE class body remains unchanged."""
import ast,math
from typing import Iterable
from pathlib import Path
import torch
from torch import nn,Tensor
from torch.nn import functional as F

def original_core(path):
 tree=ast.parse(Path(path).read_text(encoding='utf8'));node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SparseMsREResidual');scope=dict(math=math,Iterable=Iterable,torch=torch,Tensor=Tensor,nn=nn,F=F)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
 return scope['SparseMsREResidual']

class ParallelMsREInterface(nn.Module):
 def __init__(self,channels,core_type):
  super().__init__();assert tuple(channels)==(48,104,208,440);self.slots=(2,5,8,11)
  self.core=core_type(embed_dims=384,injection_indices=self.slots,token_length=16,scale_init=1e-3)
  self.input_projections=nn.ModuleList([nn.Conv2d(c,384,1) for c in channels]);self.output_projections=nn.ModuleList([nn.Conv2d(384,c,1,bias=False) for c in channels]);self.enabled=True
 def forward(self,features):
  if not self.enabled:return features
  out=list(features);start=len(out)-4
  for i,slot in enumerate(self.slots):
   raw=features[start+i];z=self.input_projections[i](raw);b,c,h,w=z.shape;patches=z.flatten(2).transpose(1,2);dummy=z.new_zeros((b,1,c));tokens=torch.cat([dummy,patches],dim=1)
   adapted=self.core(tokens,slot,h,w)[:,1:];delta=(adapted-patches).transpose(1,2).reshape(b,c,h,w);out[start+i]=raw+self.output_projections[i](delta)
  return out

class FrozenR4WithParallelMsRE(nn.Module):
 def __init__(self,base,adapter):
  super().__init__();self.base=base;self.adapter=adapter
  for p in base.parameters():p.requires_grad_(False)
  base.eval();self.handle=base.encoder.register_forward_hook(lambda module,args,features:self.adapter(features))
 def train(self,mode=True):
  super().train(mode);self.base.eval();return self
 def forward(self,x):return self.base(x)
