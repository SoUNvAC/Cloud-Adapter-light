"""Task-routed low-rank adapters inside DINOv3 QKV, final four blocks."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class OcclusionAttenuation(nn.Module):
    def __init__(self,relative=False):
        super().__init__()
        self.cloud=nn.Sequential(nn.Linear(384,4,bias=False),nn.Linear(4,1152,bias=False))
        self.shadow=nn.Sequential(nn.Linear(384,4,bias=False),nn.Linear(4,384,bias=False))
        self.route=nn.Linear(384,3)
        nn.init.zeros_(self.cloud[-1].weight);nn.init.zeros_(self.shadow[-1].weight)
        self.register_buffer('relative_mode',torch.tensor(relative,dtype=torch.bool))
        self.enabled=True;self.relative_enabled=True;self.route_logits=None

    def forward(self,x,qkv):
        assert x.ndim==3 and x.shape[1]==1029
        if not self.enabled:return qkv
        patch=x[:,5:];z=F.layer_norm(patch,(384,))
        logits=self.route(z);self.route_logits=logits
        prob=logits.softmax(-1)
        cloud=self.cloud(z)*prob[...,1:2]
        shadow=self.shadow(z)
        if self.relative_mode and self.relative_enabled:
            shadow=.5*shadow.tanh()*qkv[:,5:,768:]
        shadow=shadow*prob[...,2:3]
        change=cloud+F.pad(shadow,(768,0))
        return qkv+F.pad(change,(0,0,5,0))

    def auxiliary(self,y):
        valid=y!=255;safe=y.clamp(0,2)
        gt=F.one_hot(safe,3).permute(0,3,1,2).float()*valid[:,None]
        count=F.adaptive_avg_pool2d(valid[:,None].float(),(32,32))
        target=F.adaptive_avg_pool2d(gt,(32,32))/count.clamp_min(1e-8)
        logits=self.route_logits.transpose(1,2).reshape(y.shape[0],3,32,32)
        return -(target*logits.log_softmax(1)*count).sum()/count.sum().clamp_min(1)

def install(backbone,modules):
    assert len(modules)==4
    for index,adapter in zip((8,9,10,11),modules):
        backbone.encoder.blocks[index].attn.qkv.register_forward_hook(lambda m,a,o,adapter=adapter:adapter(a[0],o))
