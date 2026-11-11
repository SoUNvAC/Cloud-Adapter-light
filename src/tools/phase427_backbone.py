"""Independent switches with unchanged legacy module/state_dict names."""
import torch
from torch import nn
from mmseg.registry import MODELS
from cloud_adapter.models.backbones.target_msre_dinov2 import TargetMsRECloudAdapterDinoVisionTransformer,SparseMsREResidual
from cloud_adapter.models.backbones.utils import set_requires_grad,set_train
from phase427_fused import SpatialMsRE as SparseMsREResidual, attach
import os

@MODELS.register_module()
class Phase400TargetMsRECloudAdapter(TargetMsRECloudAdapterDinoVisionTransformer):
 def __init__(self,msre_enabled=True,head_delta_enabled=True,msre_init_seed=None,head_init_seed=None,**kwargs):
  super().__init__(**kwargs);self.msre_enabled=bool(msre_enabled);self.head_delta_enabled=bool(head_delta_enabled)
  # Explicit independent initialization does not advance the data/global RNG.
  if msre_init_seed is not None:
   config=dict(kwargs.get('target_msre_config') or {});config.setdefault('embed_dims',self.embed_dim);config['enabled']=self.target_enabled and self.msre_enabled
   with torch.random.fork_rng(devices=[]):torch.manual_seed(msre_init_seed);self.target_msre=SparseMsREResidual(**config)
  if head_init_seed is not None:
   with torch.random.fork_rng(devices=[]):
    torch.manual_seed(head_init_seed);heads=nn.ModuleList()
    for _ in self.out_indices:
     head=nn.Sequential(nn.Conv2d(self.embed_dim,kwargs.get('head_delta_rank',8),1),nn.GELU(),nn.Conv2d(kwargs.get('head_delta_rank',8),self.embed_dim,1));nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias);heads.append(head)
    self.target_head_delta=heads
  self.mode=os.environ['PHASE427_MODE']
  msre=self.target_msre
  del self.target_msre
  attach(self.cloud_adapter,msre,self.mode)
  self.set_target_enabled(self.target_enabled)
 def set_target_enabled(self,enabled):
  self.target_enabled=bool(enabled);self.target_msre.enabled=bool(enabled) and self.msre_enabled
 def __getattr__(self,name):
  if name=='target_msre':
   modules=self.__dict__.get('_modules',{})
   if 'cloud_adapter' in modules and hasattr(modules['cloud_adapter'],'msre'):
    return modules['cloud_adapter'].msre
  return super().__getattr__(name)
 def forward_features(self,x,masks=None):
  b,_,height,width=x.shape;h=height//self.patch_size;w=width//self.patch_size
  cache,detail=self.cloud_adapter.prepare(x)
  x=self.prepare_tokens_with_masks(x,masks);outputs=[];slot=0
  for index,block in enumerate(self.blocks):
   if index not in self._active_block_index_set:continue
   x=block(x)
   if index in self.adapter_index:
    x,cache,detail=self.cloud_adapter.interact(x,slot,index,h,w,cache,detail);slot+=1
   if index in self.out_indices:
    outputs.append(x[:,1:].transpose(1,2).reshape(b,-1,h,w).contiguous())
  return outputs,cache
 def train(self,mode=True):
  nn.Module.train(self,False)
  if not mode:return self
  active=['cloud_adapter.msre.']
  if self.mode!='baseline':active.append('cloud_adapter.net.')
  if self.mode=='integrated':active.append('cloud_adapter.detail.')
  for name,p in self.named_parameters():p.requires_grad=any(name.startswith(k) for k in active)
  self.cloud_adapter.msre.train()
  if self.mode!='baseline':self.cloud_adapter.net.train()
  if self.mode=='integrated':self.cloud_adapter.detail.train()
  return self
