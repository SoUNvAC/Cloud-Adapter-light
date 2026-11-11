"""Local material-reference residual inside MsRE; no auxiliary semantic loss."""
import os
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
MODES=['baseline','neighborhood','material']
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
        return super().forward(inputs,data_samples,mode)


class PhysicsMsRE(LayerwiseMsRE):
    def __init__(self, mode=None, **kwargs):
        super().__init__(**kwargs)
        self.mode=mode or os.environ['PHASE413_MODE'];assert self.mode in MODES
        self.raw_rgb=None
        if self.mode!='baseline':
            # Additional modules do not alter the common initialization/RNG.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(4210101)
                self.local_projection=nn.Linear(self.embed_dims,16)
                self.local_output=nn.Sequential(nn.Linear(54,16),nn.GELU(),nn.Linear(16,self.embed_dims))
                nn.init.zeros_(self.local_output[-1].weight);nn.init.zeros_(self.local_output[-1].bias)

    def neighbors(self, value):
        b,c,h,w=value.shape
        padded=F.pad(value,(3,3,3,3))
        valid=F.pad(torch.ones_like(value[:,:1]),(3,3,3,3))
        offsets=[(dy*r,dx*r) for r in (1,3) for dy,dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]]
        return (torch.stack([padded[:,:,3+dy:3+dy+h,3+dx:3+dx+w] for dy,dx in offsets],2),
                torch.stack([valid[:,:,3+dy:3+dy+h,3+dx:3+dx+w] for dy,dx in offsets],2))

    def context(self, feature, rgb):
        lum=(rgb*rgb.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True).clamp_min(.01)
        chrom=rgb/rgb.sum(1,keepdim=True).clamp_min(.03)
        loglum=lum.log()
        avg=lambda x:F.avg_pool2d(F.pad(x,(1,1,1,1),mode='replicate'),3,stride=1)
        texture=(avg(loglum.square())-avg(loglum).square()).clamp_min(0).add(1e-6).sqrt()
        values=torch.cat([feature,lum,chrom,texture],1)
        neighbor,valid=self.neighbors(values)
        nf=neighbor[:,:16];nl=neighbor[:,16:17];nc=neighbor[:,17:20];nt=neighbor[:,20:21]
        sim=(F.normalize(feature,dim=1).unsqueeze(2)*F.normalize(nf,dim=1)).sum(1,keepdim=True)*4
        brighter=torch.sigmoid((nl.clamp_min(.01).log()-loglum.unsqueeze(2)-.05)/.1)
        compatibility=torch.exp(-((nc-chrom.unsqueeze(2)).square().sum(1,keepdim=True)/.02+(nt-texture.unsqueeze(2)).abs()/.2))
        if self.mode=='material':sim=sim+(brighter*compatibility).clamp_min(1e-8).log()
        weight=sim.masked_fill(valid==0,-1e4).softmax(2)*valid
        weight=weight/weight.sum(2,keepdim=True).clamp_min(1e-8)
        ref=(weight*neighbor).sum(2)
        confidence=(weight*brighter*compatibility).sum(2) if self.mode=='material' else valid.amax(2)
        descriptor=torch.cat([feature,ref[:,:16],ref[:,:16]-feature,
            (ref[:,16:17].clamp_min(.01).log()-loglum).clamp(-3,3),ref[:,17:20]-chrom,ref[:,20:21]-texture,confidence],1)
        self.diagnostics={'confidence':confidence.detach().mean()}
        return descriptor,confidence

    def forward(self, features, block_index, height, width):
        out=super().forward(features,block_index,height,width)
        if not self.enabled or self.mode=='baseline' or block_index!=self.injection_indices[-1]:return out
        x=F.layer_norm(features[:,1:],(self.embed_dims,))
        feature=self.local_projection(x).transpose(1,2).reshape(x.shape[0],16,height,width)
        rgb=F.adaptive_avg_pool2d(self.raw_rgb,(height,width)).clamp_min(0)
        descriptor,confidence=self.context(feature,rgb)
        delta=self.local_output(descriptor.flatten(2).transpose(1,2))*confidence.flatten(2).transpose(1,2)
        return torch.cat([out[:,:1],out[:,1:]+self.scale[-1]*delta],1)

def self_test():
    torch.set_num_threads(4);records={}
    states=[]
    for mode in MODES:
        torch.manual_seed(4090101);base=LayerwiseMsRE(embed_dims=8,token_length=4);rng=torch.get_rng_state()
        torch.manual_seed(4090101);model=PhysicsMsRE(mode=mode,embed_dims=8,token_length=4)
        assert torch.equal(rng,torch.get_rng_state())
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items())
        model.raw_rgb=torch.rand(2,3,16,16);x=torch.randn(2,65,8)
        for i in [2,5,8,11]:torch.testing.assert_close(model(x,i,8,8),base(x,i,8,8),rtol=0,atol=0)
        model(x,11,8,8).square().sum().backward()
        if mode!='baseline':
            assert model.local_output[-1].weight.grad.abs().sum()>0
            with torch.no_grad():model.local_output[-1].weight.normal_(0,.01)
            model.zero_grad();model(x,11,8,8).square().sum().backward()
            assert model.local_projection.weight.grad.abs().sum()>0
            for h,w in [(1,1),(2,3),(8,8)]:
                d,c=model.context(torch.randn(2,16,h,w),torch.zeros(2,3,h,w))
                assert torch.isfinite(d).all() and torch.isfinite(c).all()
            if mode=='material':
                _,c=model.context(torch.ones(1,16,5,5),torch.ones(1,3,5,5)*.3)
                uniform=float(c.mean())
                rgb=torch.ones(1,3,5,5)*.6;rgb[:,:,2,2]=.1
                _,c=model.context(torch.ones(1,16,5,5),rgb)
                assert c[0,0,2,2]>uniform
        model.enabled=False;assert torch.equal(model(x,11,8,8),x)
        torch.manual_seed(4090101);prod=PhysicsMsRE(mode=mode)
        records[mode]={'passed':True,'parameters':sum(p.numel() for p in prod.parameters()),'zero_initial_identity':True,'common_weights_and_RNG_equal':True}
        if mode!='baseline':states.append(prod.state_dict())
    assert all(torch.equal(v,states[1][k]) for k,v in states[0].items())
    assert records['neighborhood']['parameters']==records['material']['parameters']
    return records
