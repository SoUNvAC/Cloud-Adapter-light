"""Dense 1/4-resolution decoder bridge with optional achromatic/chromatic bands."""
import torch
from torch import nn
from torch.nn import functional as F


def bands(rgb):
    x=(rgb+.02).log()
    result=[]
    for d in (1,2,4):
        p=F.pad(x,(0,d,0,d),mode='replicate');h,w=x.shape[-2:]
        a,b,c,e=p[...,:h,:w],p[...,:h,d:d+w],p[...,d:d+h,:w],p[...,d:d+h,d:d+w]
        result.extend([(a-b+c-e)/2,(a+b-c-e)/2,(a-b-c+e)/2])
    return torch.stack(result,1)  # B,9,RGB,H,W; undecimated Haar-like details


class DetailBridge(nn.Module):
    def __init__(self,frequency=False):
        super().__init__()
        self.detail=nn.Conv2d(64,24,1)
        self.semantic=nn.Conv2d(384,16,1)
        self.appearance=nn.Sequential(nn.Conv2d(3,24,1),nn.GELU(),nn.Conv2d(24,24,3,padding=1,groups=24),nn.GELU())
        self.mix=nn.Sequential(nn.Conv2d(64,32,1),nn.GELU(),nn.Conv2d(32,32,3,padding=1,groups=32),nn.GELU(),nn.Conv2d(32,384,1))
        nn.init.zeros_(self.mix[-1].weight);nn.init.zeros_(self.mix[-1].bias)
        self.register_buffer('frequency_mode',torch.tensor(frequency,dtype=torch.bool))
        if frequency:
            self.band_gate=nn.Sequential(nn.Conv2d(58,24,1),nn.GELU(),nn.Conv2d(24,9,1))
            self.band_project=nn.Conv2d(36,24,1)
        self.enabled=True;self.frequency_enabled=True;self.rgb=None;self.cnn=None

    def forward(self,feature):
        if not self.enabled:return torch.zeros_like(feature)
        assert feature.shape[-2:]==self.cnn.shape[-2:]
        detail=self.detail(self.cnn)
        sem=self.semantic(F.layer_norm(feature.permute(0,2,3,1),(384,)).permute(0,3,1,2))
        rgb=F.adaptive_avg_pool2d(self.rgb,feature.shape[-2:])
        appearance=self.appearance(rgb)
        if self.frequency_mode and self.frequency_enabled:
            coeff=bands(rgb);common=coeff.mean(2,keepdim=True);chroma=coeff-common
            energy=(chroma.square().mean(2)+1e-8).sqrt()
            gate=self.band_gate(torch.cat([detail,sem,common[:, :, 0].abs(),energy],1)).sigmoid()
            separated=torch.cat([common,chroma],2)*gate[:,:,None]
            appearance=appearance+self.band_project(separated.flatten(1,2))
        return self.mix(torch.cat([detail,sem,appearance],1))


def install(backbone,module):
    def prepare(m,args):
        x=args[0];mean=x.new_tensor([123.675,116.28,103.53])[None,:,None,None];std=x.new_tensor([58.395,57.12,57.375])[None,:,None,None]
        module.rgb=((x*std+mean)/255).clamp(0,1)
    backbone.register_forward_pre_hook(prepare)
    backbone.spatial.register_forward_hook(lambda m,a,o:setattr(module,'cnn',o[1]))
    # Run before existing feature capture hooks, so decoder and450 see identical features.
    backbone.register_forward_hook(lambda m,a,o:(o[0]+module(o[0]),*o[1:]),prepend=True)
