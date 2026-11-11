"""Semantic material-reference evidence, with an explicit learnable null match.

Appearance cues are display-RGB descriptors, not reflectance estimates.
All segmentation decisions remain the original three-class argmax.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .texture_base import ShadowRedistribution as Base


def shift(x, dy, dx):
    """Read a neighbor at (y+dy,x+dx); missing neighbors are zero, never wrapped."""
    h,w=x.shape[-2:]
    pad=max(abs(dy),abs(dx))
    return F.pad(x,(pad,pad,pad,pad))[...,pad+dy:pad+dy+h,pad+dx:pad+dx+w]


class ShadowRedistribution(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Preserve initialization of all retained Phase496 parameters.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(49701)
            self.reference_projection=nn.Conv2d(384,16,1,bias=False)
            self.log_temperature=nn.Parameter(torch.tensor(math.log(8.0)))
            self.null_logit=nn.Parameter(torch.tensor(6.4))
            self.evidence[0]=nn.Conv2d(20,16,3,padding=1,bias=False)
        directions=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        self.offsets=[(r*y,r*x) for r in (1,2) for y,x in directions]
        self.uniform_reference=False
        self.last_confidence=None

    def routing(self,native,size):
        q=F.normalize(self.reference_projection(F.layer_norm(
            native.permute(0,2,3,1),(384,)).permute(0,3,1,2)),dim=1,eps=1e-6)
        temperature=self.log_temperature.exp().clamp(max=40)
        one=torch.ones_like(q[:,:1])
        scores=[]
        for dy,dx in self.offsets:
            valid=shift(one,dy,dx)>0
            score=(q*shift(q,dy,dx)).sum(1,keepdim=True)*temperature
            scores.append(score.masked_fill(~valid,-1e4))
        scores=torch.cat([self.null_logit.expand_as(one),*scores],1)
        weights=F.interpolate(scores.softmax(1),size=size,mode='bilinear',align_corners=False)
        valid=torch.cat([torch.ones_like(weights[:,:1])]+[
            shift(torch.ones_like(weights[:,:1]),dy*4,dx*4) for dy,dx in self.offsets],1)
        weights=weights*valid
        weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-8)
        if self.uniform_reference:
            # Preserve learned null mass, replace only neighbor preference.
            confidence=1-weights[:,:1]
            neighbors=valid[:,1:]/valid[:,1:].sum(1,keepdim=True).clamp_min(1)
            weights=torch.cat([weights[:,:1],confidence*neighbors],1)
        return weights

    @staticmethod
    def avg(x,k):
        return F.avg_pool2d(F.pad(x,(k//2,)*4,mode='replicate'),k,1)

    def descriptors(self,rgb):
        with torch.no_grad():
            y=(rgb*rgb.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
            # Identical pre-pooling amplitude calculation to Phase496.
            h,w=y.shape[-2:];micro=[]
            for r in (1,2,4):
                p=F.pad(y,(r,)*4,mode='replicate')
                for dy,dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                    neighbor=p[:,:,r+dy*r:r+dy*r+h,r+dx*r:r+dx*r+w]
                    micro.append(F.avg_pool2d(torch.tanh(20*(neighbor-y)),4,4))
            micro=torch.cat(micro,1)
            rgb4=F.avg_pool2d(rgb,4,4)
            chroma=rgb4/(rgb4.sum(1,keepdim=True)+.02)
            y=F.avg_pool2d(y,4,4)
            p=F.pad(y,(1,1,1,1),mode='replicate')
            gx=(p[:,:,1:-1,2:]-p[:,:,1:-1,:-2])*.5
            gy=(p[:,:,2:,1:-1]-p[:,:,:-2,1:-1])*.5
            levels=[]
            for k in (3,7):
                xx=self.avg(gx.square(),k);yy=self.avg(gy.square(),k);xy=self.avg(gx*gy,k)
                energy=xx+yy
                levels.append(torch.cat([(xx-yy)/(energy+1e-5),2*xy/(energy+1e-5),torch.log1p(100*energy)],1))
            return torch.cat([*levels,y,chroma,micro],1) # 6+1+3+24 = 34 channels

    def forward(self,native,detail):
        if not self.enabled:
            return torch.zeros_like(F.interpolate(native,size=detail.shape[-2:],mode='bilinear',align_corners=False))
        local=self.descriptors(self.rgb)
        weights=self.routing(native,local.shape[-2:])
        confidence=weights[:,1:].sum(1,keepdim=True)
        reference=torch.zeros_like(local)
        for i,(dy,dx) in enumerate(self.offsets):
            reference=reference+weights[:,i+1:i+2]*shift(local,dy*4,dx*4)
        reference=reference/confidence.clamp_min(1e-6)
        contrast=confidence*(local-reference)
        brightness=confidence*(local[:,6:7]-reference[:,6:7])/(reference[:,6:7]+.02)
        self._texture=torch.cat([
            torch.cat([local[:,i:i+3],contrast[:,i:i+3],brightness,contrast[:,7:10]],1)
            for i in (0,3)],1)
        self._micro=torch.cat([local[:,10:],contrast[:,10:]],1)
        self.last_confidence=confidence.detach().mean()
        try:
            return super().forward(native,detail)
        finally:
            self._texture=None;self._micro=None

    def texture_evidence(self,rgb,native):
        return self._texture

    def ordinal_evidence(self,rgb):
        return self._micro
