"""MsRE-owned Cloud-Adapter with local detail reads and recurrent spatial feedback."""
import os
import torch
from torch import nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from cloud_adapter.models.backbones.cloud_adapter import CloudAdapter
from phase411_msre import LayerwiseMsRE

MODES=['baseline','unfrozen','integrated']

class SpatialMsRE(LayerwiseMsRE):
    def forward(self,features,block_index,height,width,detail=None):
        if detail is None:return super().forward(features,block_index,height,width)
        if not self.enabled or block_index not in self.layer_slots:return features
        cls,x=features[:,:1],features[:,1:];b,n,c=x.shape
        assert n==height*width and detail.shape==x.shape
        tokens=self.learnable_tokens[self.layer_slots[block_index]]
        spatial=x.transpose(1,2).reshape(b,c,height,width)
        multi=torch.stack([conv(spatial) for conv in self.convs],1)
        multi=multi.permute(0,1,3,4,2).reshape(b,3,n,c)
        multi=(multi+detail[:,None])*self.level_encoding.weight[None,:,None,:]
        attn=torch.einsum('bsnc,mc->bsnm',multi,tokens).mul(c**-.5).softmax(-1)
        delta=torch.einsum('bsnm,mc->bsnc',attn[...,1:],self.mlp_token2feat(tokens[1:])).mean(1)+x+detail
        projected=self.projector(delta.transpose(1,2).reshape(b,c,height,width)).flatten(2).transpose(1,2)
        return torch.cat([cls,x+self.scale[self.layer_slots[block_index]]*(projected+delta)],1)

class LocalDetail(nn.Module):
    def __init__(self,channels=384,context=64,rank=16):
        super().__init__();self.rank=rank
        self.q=nn.Linear(channels,rank,bias=False)
        self.kv=nn.Conv2d(context,2*rank,1,bias=False)
        self.out=nn.Linear(rank,channels,bias=False)
        self.feedback=nn.Linear(channels,context,bias=False)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.feedback.weight)

    def read(self,x,detail,h,w):
        # Exactly 4x4 higher-resolution samples per patch; no global detail attention.
        assert detail.shape[-2:]==(4*h,4*w)
        b,n,c=x.shape;r=self.rank
        kv=F.unfold(self.kv(detail),kernel_size=4,stride=4).reshape(b,2,r,16,n)
        k,v=kv[:,0].permute(0,3,2,1),kv[:,1].permute(0,3,2,1)
        q=self.q(F.layer_norm(x,(c,)))
        weight=((q[:,:,None]*k).sum(-1)*(r**-.5)).softmax(-1)
        return self.out((weight[...,None]*v).sum(-2))

    def write(self,x,detail,h,w):
        update=self.feedback(F.layer_norm(x,(x.shape[-1],))).transpose(1,2).reshape(x.shape[0],-1,h,w)
        return detail+.01*F.interpolate(update,size=detail.shape[-2:],mode='bilinear',align_corners=False)

@MODELS.register_module()
class Phase427CloudAdapter(CloudAdapter):
    def prepare(self,image):
        # Match PMAAConvBlock.forward's order exactly, retaining stage 2 before pooling.
        cnn=self.cnn
        values=[cnn.spp_dw[0](cnn.proj_1x1(image))]
        for block in cnn.spp_dw[1:]:values.append(block(values[-1]))
        cache=torch.zeros_like(values[-1])
        for value in values:cache=cache+F.adaptive_avg_pool2d(value,values[-1].shape[-2:])
        return cache,values[2]

    def interact(self,features,slot,index,h,w,cache,detail):
        x=super().forward(features,slot,batch_first=True,has_cls_token=True,cache=cache)
        if self.mode=='integrated':
            local=self.detail.read(x[:,1:],detail,h,w)
            x=self.msre(x,index,h,w,detail=local)
            if slot<len(self.msre.injection_indices)-1:
                updated=self.detail.write(x[:,1:],detail,h,w)
                # Feed semantic update into both retained detail and next global context.
                cache=cache+F.adaptive_avg_pool2d(updated-detail,cache.shape[-2:])
                detail=updated
        else:x=self.msre(x,index,h,w)
        return x,cache,detail

def attach(adapter,msre,mode):
    adapter.mode=mode;adapter.msre=msre
    if mode=='integrated':
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4270101);adapter.detail=LocalDetail(msre.embed_dims,64)

def self_test():
    torch.set_num_threads(4);records={}
    cfg=dict(cnn_type='pmaa',int_type='convnext',emd_dim=8,num_layers=4,
             context_dim=64,hidden_channels=64,depth=4,return_multi_feats=False,
             return_last_feature=False,rank_dim=8)
    for mode in MODES:
        torch.manual_seed(7);adapter=Phase427CloudAdapter(**cfg).eval()
        torch.manual_seed(4090101);msre=SpatialMsRE(embed_dims=8)
        rng=torch.get_rng_state().clone();attach(adapter,msre,mode)
        assert torch.equal(rng,torch.get_rng_state())
        rgb=torch.randn(1,3,64,64);cache,detail=adapter.prepare(rgb)
        torch.testing.assert_close(cache,adapter.cnn(rgb),atol=0,rtol=0)
        x=torch.randn(1,17,8);a=x;b=x
        for slot,index in enumerate([2,5,8,11]):
            b=CloudAdapter.forward(adapter,b,slot,cache=cache)
            b=LayerwiseMsRE.forward(msre,b,index,4,4)
            a,cache,detail=adapter.interact(a,slot,index,4,4,cache,detail)
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
        if mode=='integrated':
            with torch.no_grad():adapter.detail.out.weight.normal_(0,.01);adapter.detail.feedback.weight.normal_(0,.01)
            cache,detail=adapter.prepare(rgb);before=detail.clone()
            y,cache,detail=adapter.interact(x,0,2,4,4,cache,detail)
            z,_,_=adapter.interact(y,1,5,4,4,cache,detail)
            z.square().mean().backward()
            assert not torch.equal(before,detail)
            for p in adapter.detail.parameters():assert p.grad is not None and p.grad.abs().sum()>0
            # Local reading depends on location, not just a pooled detail summary.
            random_detail=torch.randn(1,64,16,16)
            assert not torch.equal(adapter.detail.read(x[:,1:],random_detail,4,4),adapter.detail.read(x[:,1:],random_detail.roll(4,-1),4,4))
        prod_msre=SpatialMsRE()
        count=sum(p.numel() for p in prod_msre.parameters())
        if mode!='baseline':
            prod_adapter=Phase427CloudAdapter(**{**cfg,'emd_dim':384})
            count+=sum(p.numel() for p in prod_adapter.net.parameters())
        if mode=='integrated':count+=sum(p.numel() for p in LocalDetail().parameters())
        records[mode]={'passed':True,'parameters':count,'cache_exact':True,'initial_output_close':True,
                       'feedback_and_locality_gradient_test':mode=='integrated'}
    return records
