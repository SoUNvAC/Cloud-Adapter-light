"""Three semantic token banks and competitive residual routing inside MsRE."""
import os
import torch
from torch import nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from cloud_adapter.models.segmentors.frozen_encoder_decoder import FrozenHeadEncoderDecoder
from phase411_msre import LayerwiseMsRE

MODES=['baseline','competitive']

def soft_targets(labels, size):
    valid=labels!=255
    one=F.one_hot(labels.masked_fill(~valid,0).long(),3).permute(0,3,1,2).float()
    mass=F.adaptive_avg_pool2d(one*valid[:,None],size)
    support=mass.sum(1)
    return mass/support[:,None].clamp_min(1e-8),support

class CompetitiveMsRE(LayerwiseMsRE):
    def __init__(self,mode=None,**kwargs):
        super().__init__(**kwargs)
        self.mode=mode or os.environ['PHASE426_MODE']
        assert self.mode in MODES and self.token_length==16
        self.semantic_logits=[]
        if self.mode=='competitive':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(4260101)
                # Scores + shadow/cloud and shadow/surface margins -> three gates.
                self.router=nn.Linear(5,3,bias=False)
                nn.init.zeros_(self.router.weight)

    def forward(self,features,block_index,height,width):
        if self.mode=='baseline':return super().forward(features,block_index,height,width)
        if not self.enabled or block_index not in self.layer_slots:return features
        slot=self.layer_slots[block_index]
        x=features[:,1:];b,n,c=x.shape
        assert n==height*width
        tokens=self.learnable_tokens[slot]
        spatial=x.transpose(1,2).reshape(b,c,height,width)
        multi=torch.stack([conv(spatial) for conv in self.convs],1)
        multi=multi.permute(0,1,3,4,2).reshape(b,3,n,c)*self.level_encoding.weight[None,:,None,:]
        logits=torch.einsum('bsnc,mc->bsnm',multi,tokens)*(c**-.5)
        grouped=logits[...,1:].reshape(b,3,n,3,5)
        evidence=torch.logsumexp(grouped,-1)
        centered=evidence-evidence.mean(-1,keepdim=True)
        descriptor=torch.cat([centered,(evidence[...,2]-evidence[...,1])[...,None],
                              (evidence[...,2]-evidence[...,0])[...,None]],-1)
        weights=(evidence+self.router(descriptor)).softmax(-1)
        values=self.mlp_token2feat(tokens[1:]).reshape(3,5,c)
        candidates=torch.einsum('bsngk,gkc->bsngc',grouped.softmax(-1),values)
        # Preserve reserved-token mass: zero router recovers original attention algebra.
        mass=1-logits.softmax(-1)[...,0:1]
        delta=(weights[...,None]*candidates).sum(-2)*mass
        delta=delta.mean(1)+x
        projected=self.projector(delta.transpose(1,2).reshape(b,c,height,width)).flatten(2).transpose(1,2)
        if self.training and torch.is_grad_enabled():
            self.semantic_logits.append(evidence.mean(1).transpose(1,2).reshape(b,3,height,width))
        return torch.cat([features[:,:1],x+self.scale[slot]*(projected+delta)],1)

    def semantic_loss(self,labels):
        losses=[]
        for logits in self.semantic_logits:
            target,support=soft_targets(labels,logits.shape[-2:])
            logp=logits.log_softmax(1)
            # Detached difficulty for shadow/nonshadow confusion, bounded [1,3].
            mismatch=(logp.detach().exp()[:,2]-target[:,2]).abs()
            weight=support*(1+2*mismatch)
            losses.append((-(target*logp).sum(1)*weight).sum()/weight.sum().clamp_min(1e-8))
        assert losses
        return torch.stack(losses).mean()*.05

@MODELS.register_module()
class Phase426Segmentor(FrozenHeadEncoderDecoder):
    def loss(self,inputs,data_samples):
        module=self.backbone.target_msre
        module.semantic_logits=[]
        result=super().loss(inputs,data_samples)
        if module.mode=='competitive':
            labels=torch.stack([s.gt_sem_seg.data[0] for s in data_samples])
            result['loss_token_semantics']=module.semantic_loss(labels)
        module.semantic_logits=[]
        return result

def self_test():
    torch.set_num_threads(4);records={}
    for mode in MODES:
        torch.manual_seed(4090101);base=LayerwiseMsRE(embed_dims=8)
        rng=torch.get_rng_state().clone()
        torch.manual_seed(4090101);model=CompetitiveMsRE(mode=mode,embed_dims=8)
        assert torch.equal(rng,torch.get_rng_state())
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items())
        x=torch.randn(2,17,8)
        for index in [2,5,8,11]:
            torch.testing.assert_close(model(x,index,4,4),base(x,index,4,4),rtol=1e-5,atol=1e-6)
        if mode=='competitive':
            labels=torch.randint(0,3,(2,16,16));labels[:,0]=255
            loss=model.semantic_loss(labels)+model(x,11,4,4).square().mean()
            loss.backward()
            assert model.router.weight.grad.abs().sum()>0
            assert model.learnable_tokens.grad.abs().sum()>0
            model.semantic_logits=[];model(x,11,4,4)
            zero=model.semantic_loss(torch.full_like(labels,255))
            assert torch.isfinite(zero) and zero==0
            model.eval();model.semantic_logits=[];model(x,11,4,4)
            assert not model.semantic_logits
        model.enabled=False;assert torch.equal(model(x,11,4,4),x)
        target,support=soft_targets(torch.tensor([[[0,1],[2,255]]]),(1,1))
        torch.testing.assert_close(target.flatten(),torch.ones(3)/3)
        assert support.item()==.75
        prod=CompetitiveMsRE(mode=mode)
        records[mode]={'passed':True,'parameters':sum(p.numel() for p in prod.parameters()),
                       'common_initialization_equal':True,'initial_output_close':True,
                       'ignore_and_mixed_target_test':True}
    return records
