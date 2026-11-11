"""RGB cloud/shadow mechanism probes. Approximate operators, not physical retrieval."""
import os
from phase416_optimizer import optimizer_self_test
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from mmcv.transforms import BaseTransform
from mmengine.structures import PixelData
from mmseg.datasets.transforms import PackSegInputs
from mmseg.registry import TRANSFORMS, MODELS
from cloud_adapter.models.segmentors.frozen_encoder_decoder import FrozenHeadEncoderDecoder
from phase411_msre import LayerwiseMsRE

MODES=['link_generic', 'link_cloud', 'field_spatial', 'field_illumination', 'field_shuffled']

@TRANSFORMS.register_module()
class Phase413KeepRaw(BaseTransform):
    def transform(self, results):
        # Loader is BGR in linear TOA*255 units. Copy AFTER crop/flip/resize,
        # BEFORE photometric distortion. No clipping of reflectance >1.
        results['phase413_raw']=np.ascontiguousarray(results['img'][...,::-1].transpose(2,0,1)/255.)
        return results

@TRANSFORMS.register_module()
class Phase413Pack(PackSegInputs):
    def transform(self, results):
        packed=super().transform(results)
        packed['data_samples'].raw_rgb=PixelData(data=torch.from_numpy(results['phase413_raw'].copy()))
        return packed

@MODELS.register_module()
class Phase413Segmentor(FrozenHeadEncoderDecoder):
    def forward(self, inputs, data_samples=None, mode='tensor'):
        assert data_samples and all(hasattr(s,'raw_rgb') for s in data_samples)
        raw=torch.stack([s.raw_rgb.data for s in data_samples]).to(inputs)
        assert raw.shape==inputs.shape and torch.isfinite(raw).all()
        self.backbone.target_msre.raw_rgb=raw
        self.backbone.target_msre.aux_logits=None
        return super().forward(inputs,data_samples,mode)

    def loss(self, inputs, data_samples):
        losses=super().loss(inputs,data_samples)
        logits=self.backbone.target_msre.aux_logits
        labels=torch.stack([s.gt_sem_seg.data for s in data_samples]).squeeze(1)
        labels=F.interpolate(labels[:,None].float(),size=logits.shape[-2:],mode='nearest')[:,0].long()
        valid=labels!=255
        losses['mechanism.loss_semantic']=.1*(F.cross_entropy(logits,labels,ignore_index=255) if valid.any() else logits.sum()*0)
        return losses

class PhysicsMsRE(LayerwiseMsRE):
    def __init__(self, mode=None, **kwargs):
        super().__init__(**kwargs)
        self.mode=mode or os.environ['PHASE413_MODE'];assert self.mode in MODES
        c=self.embed_dims
        self.semantic=nn.Linear(c,3)
        self.material=nn.Linear(c,16)
        self.operator=nn.Linear(16,10)
        self.assignment=nn.Linear(3,16)
        self.offset=nn.Linear(16,8)
        self.unmatched=nn.Linear(16,1)
        self.output=nn.Sequential(nn.Linear(16,16),nn.GELU(),nn.Linear(16,c))
        nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)
        # The same modules exist in every variant, but inactive family-specific
        # parameters are frozen: scope counts actual trainable parameters.
        family=self.mode.split('_')[0]
        active={'dual':{'semantic','material','operator','output'},
                'link':{'semantic','material','offset','unmatched','output'},
                'field':{'semantic','material','operator','assignment','output'}}[family]
        self.inactive_prefixes=tuple(n+'.' for n in ['semantic','material','operator','assignment','offset','unmatched','output'] if n not in active)
        self.raw_rgb=None;self.aux_logits=None

    def freeze_inactive(self):
        for n,p in self.named_parameters():
            if n.startswith(self.inactive_prefixes):p.requires_grad_(False)

    def references(self, rgb, feature, surface, h, w):
        # 4x4 candidate material regions; exclude the query's entire own region.
        def pool(x):return F.adaptive_avg_pool2d(x,(4,4)).flatten(2).transpose(1,2)
        p=surface.reshape(-1,1,h,w)
        mass=pool(p)
        keys=pool(feature.transpose(1,2).reshape(-1,16,h,w)*p)/mass.clamp_min(1e-6)
        colours=pool(rgb*p)/mass.clamp_min(1e-6)
        q=F.normalize(feature,dim=-1);k=F.normalize(keys,dim=-1)
        score=torch.einsum('bnc,bkc->bnk',q,k)*4+mass[...,0].clamp_min(1e-6).log()[:,None,:]
        y,x=torch.meshgrid(torch.arange(h,device=rgb.device),torch.arange(w,device=rgb.device),indexing='ij')
        own=((y*4//h).clamp_max(3)*4+(x*4//w).clamp_max(3)).flatten()
        score=score.masked_fill(F.one_hot(own,16)[None].bool(),-1e4)
        weights=score.softmax(-1)
        ref=torch.einsum('bnk,bkc->bnc',weights,colours)
        confidence=torch.einsum('bnk,bkc->bnc',weights,mass).clamp(0,1)
        return ref,confidence

    def forward(self, features, block_index, height, width):
        if not self.enabled or block_index not in self.layer_slots:return features
        out=super().forward(features,block_index,height,width)
        if block_index!=self.injection_indices[-1]:return out
        assert self.raw_rgb is not None
        x=F.layer_norm(features[:,1:],(self.embed_dims,));b,n,c=x.shape
        rgb=F.adaptive_avg_pool2d(self.raw_rgb,(height,width));colour=rgb.flatten(2).transpose(1,2)
        logits=self.semantic(x.detach());prob=logits.softmax(-1)
        self.aux_logits=logits.transpose(1,2).reshape(b,3,height,width)
        material=self.material(x)
        if self.mode.startswith('link'):
            # Displacements are image coordinates, NOT solar geometry/cloud heights.
            grid_y,grid_x=torch.meshgrid(torch.linspace(-1,1,height,device=x.device),torch.linspace(-1,1,width,device=x.device),indexing='ij')
            grid=torch.stack([grid_x,grid_y],-1).to(x.dtype)[None].expand(b,-1,-1,-1)
            offsets=.5*torch.tanh(self.offset(material.mean(1))).reshape(b,4,2)
            support=prob[...,1:2] if self.mode=='link_cloud' else prob.max(-1,keepdim=True).values
            fmap=(material*support).transpose(1,2).reshape(b,16,height,width)
            confidence=support.transpose(1,2).reshape(b,1,height,width)
            values=[];masses=[]
            for k in range(4):
                displaced=grid+offsets[:,k,None,None,:]
                values.append(F.grid_sample(fmap,displaced,align_corners=True,padding_mode='zeros'))
                masses.append(F.grid_sample(confidence,displaced,align_corners=True,padding_mode='zeros'))
            value=torch.stack(values).mean(0).flatten(2).transpose(1,2)
            mass=torch.stack(masses).mean(0).flatten(2).transpose(1,2)
            # Explicit unmatched route, including out-of-frame samples. No cloud
            # evidence never hard-negates shadow: the local appearance survives.
            matched=mass*torch.sigmoid(self.unmatched(material))
            z=matched*value/mass.clamp_min(1e-5)+(1-matched)*material
            self.diagnostics={'offsets':offsets.detach(),'matched_mean':matched.detach().mean()}
        else:
            ref,confidence=self.references(rgb,material,prob[...,0:1],height,width)
            if self.mode=='dual_shuffled':ref=ref.roll(n//2,1)
            if self.mode.startswith('dual'):
                # Region-shared coefficients avoid an unconstrained per-pixel fit.
                shared=F.adaptive_avg_pool2d(material.transpose(1,2).reshape(b,16,height,width),(4,4))
                coeff=self.operator(shared.flatten(2).transpose(1,2)).transpose(1,2).reshape(b,10,4,4)
                coeff=F.interpolate(coeff,size=(height,width),mode='nearest').flatten(2).transpose(1,2)
                t=coeff[...,:1].sigmoid();air=coeff[...,1:4].sigmoid()
                attenuation=coeff[...,4:7].sigmoid();bias=.1*coeff[...,7:10].tanh()
                cloud=t*ref+(1-t)*air;shadow=attenuation*ref+bias
                if self.mode=='dual_generic':z=torch.cat([colour,ref,coeff],-1)
                else:z=torch.cat([colour,ref,cloud,shadow,(colour-cloud).abs().mean(-1,keepdim=True),(colour-shadow).abs().mean(-1,keepdim=True),confidence,t],-1)
            else:
                # Material-normalized appearance is only a proxy for illumination.
                ratio=((colour.clamp_min(.01)/ref.clamp_min(.01)).log()).clamp(-3,3)
                if self.mode=='field_spatial':
                    yy,xx=torch.meshgrid(torch.linspace(-1,1,height,device=x.device),torch.linspace(-1,1,width,device=x.device),indexing='ij')
                    assignment_input=torch.stack([yy,xx,yy*xx],-1).reshape(1,n,3).expand(b,-1,-1).to(x.dtype)
                else:assignment_input=ratio
                assignment=self.assignment(assignment_input).softmax(-1)
                if self.mode=='field_shuffled':assignment=assignment.roll(n//2,1)
                support=assignment*confidence
                pooled=torch.einsum('bnk,bnc->bkc',support,material)/support.sum(1).unsqueeze(-1).clamp_min(1e-5)
                coeff=self.operator(pooled)
                # Only 3 attenuations + 3 biases used; remaining operator channels
                # serve the routing descriptor in every field control as well.
                coefficients=torch.einsum('bnk,bkc->bnc',assignment,coeff)
                a=coefficients[...,:3].sigmoid();bias=.1*coefficients[...,3:6].tanh()
                pred=a*ref+bias
                z=torch.cat([colour,ref,pred,colour-pred,coefficients[...,6:10]],-1)
            # Abstain continuously when no visible-surface reference is reliable.
            z=z*confidence
            self.diagnostics={'reference_confidence':confidence.detach().mean()}
        delta=self.output(z)
        if not self.mode.startswith('link'):delta=delta*confidence
        return torch.cat([out[:,:1],out[:,1:]+self.scale[-1]*delta],1)

def self_test():
    optimizer_self_test()
    torch.set_num_threads(4)
    records={}
    for mode in MODES:
        torch.manual_seed(4090101);base=LayerwiseMsRE(embed_dims=8,token_length=4)
        torch.manual_seed(4090101);model=PhysicsMsRE(mode=mode,embed_dims=8,token_length=4)
        model.raw_rgb=torch.rand(2,3,16,16);x=torch.randn(2,65,8)
        for index in [2,5,8,11]:torch.testing.assert_close(model(x,index,8,8),base(x,index,8,8),rtol=0,atol=0)
        model.freeze_inactive()
        x.requires_grad_(True);model(x,11,8,8)
        model.aux_logits.square().mean().backward()
        assert x.grad is None
        assert model.semantic.weight.grad is not None and model.semantic.weight.grad.abs().sum()>0
        assert all(p.grad is None for name,p in model.named_parameters() if not name.startswith('semantic.'))
        model.zero_grad(set_to_none=True)
        model(x,11,8,8).square().sum().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if not mode.startswith('link'):
            # A learned output bias must not bypass the no-reference fallback.
            with torch.no_grad():model.output[-1].bias.fill_(.3)
            model.references=lambda rgb,feature,surface,h,w:(torch.zeros(feature.shape[0],h*w,3),torch.zeros(feature.shape[0],h*w,1))
            torch.testing.assert_close(model(x,11,8,8),base(x,11,8,8),rtol=0,atol=0)
        model.enabled=False;assert torch.equal(model(x,11,8,8),x)
        torch.manual_seed(4090101);production=PhysicsMsRE(mode=mode);production.freeze_inactive()
        records[mode]={'passed':True,'parameters':sum(p.numel() for p in production.parameters() if p.requires_grad),'allocated_parameters':sum(p.numel() for p in production.parameters())}
    # Raw copy alignment and independence from later color mutations.
    a=np.arange(8*8*3,dtype=np.float32).reshape(8,8,3)
    r=Phase413KeepRaw().transform({'img':a.copy()});raw=r['phase413_raw'].copy();r['img']*=0
    assert np.array_equal(raw,r['phase413_raw']) and np.array_equal(raw[0],a[...,2]/255.)
    return records
