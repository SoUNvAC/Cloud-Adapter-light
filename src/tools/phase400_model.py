"""Independent switches with unchanged legacy module/state_dict names."""
import torch
from torch import nn
from mmseg.registry import MODELS
from cloud_adapter.models.backbones.target_msre_dinov2 import TargetMsRECloudAdapterDinoVisionTransformer,SparseMsREResidual
from cloud_adapter.models.backbones.utils import set_requires_grad,set_train

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
  self.set_target_enabled(self.target_enabled)
 def set_target_enabled(self,enabled):
  self.target_enabled=bool(enabled);self.target_msre.enabled=bool(enabled) and self.msre_enabled
 def forward_features(self,x,masks=None):
  batch,_,height,width=x.shape;cache=self.cloud_adapter.cnn(x);grid_h,grid_w=height//self.patch_size,width//self.patch_size;x=self.prepare_tokens_with_masks(x,masks);outputs=[];adapter_slot=0
  for index,block in enumerate(self.blocks):
   if index not in self._active_block_index_set:continue
   x=block(x)
   if index in self.adapter_index:x=self.cloud_adapter.forward(x,adapter_slot,batch_first=True,has_cls_token=True,cache=cache);adapter_slot+=1
   x=self.target_msre(x,index,grid_h,grid_w)
   if index in self.out_indices:
    feature=x[:,1:].permute(0,2,1).reshape(batch,-1,grid_h,grid_w).contiguous()
    if self.target_enabled and self.head_delta_enabled:feature=feature+self.target_head_delta[len(outputs)](feature)
    outputs.append(feature)
  return outputs,cache
 def train(self,mode=True):
  if not mode:return nn.Module.train(self,False)
  keys=[]
  if self.target_enabled and self.msre_enabled:keys.append('target_msre')
  if self.target_enabled and self.head_delta_enabled:keys.append('target_head_delta')
  set_requires_grad(self,keys);set_train(self,keys);return self
