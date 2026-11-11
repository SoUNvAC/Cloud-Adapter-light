"""Half-resolution three-class mask features with learned paired-side reading."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class PairedBoundary(nn.Module):
    def __init__(self):
        super().__init__()
        self.detail=nn.Conv2d(64,16,1)
        self.semantic=nn.Conv2d(128,16,1)
        self.center=nn.Sequential(nn.Conv2d(35,16,1),nn.GELU())
        self.angles=nn.Conv2d(16,4,1)
        nn.init.zeros_(self.angles.weight);nn.init.zeros_(self.angles.bias)
        self.pair=nn.Sequential(nn.Conv2d(54,16,1),nn.GELU(),nn.Conv2d(16,16,1),nn.GELU())
        self.score=nn.Conv2d(16,1,1)
        self.mix=nn.Sequential(nn.Conv2d(32,32,3,padding=1,groups=32),nn.GELU(),nn.Conv2d(32,16,1),nn.GELU())
        self.inject=nn.Conv2d(16,128,1,bias=False);nn.init.zeros_(self.inject.weight)
        self.enabled=True;self.pair_enabled=True

    @staticmethod
    def sample(x,dx,dy):
        b,c,h,w=x.shape
        yy,xx=torch.meshgrid(torch.arange(h,device=x.device,dtype=x.dtype),torch.arange(w,device=x.device,dtype=x.dtype),indexing='ij')
        gx=2*(xx[None]+dx+.5)/w-1;gy=2*(yy[None]+dy+.5)/h-1
        valid=(gx>=-1+1/w)&(gx<=1-1/w)&(gy>=-1+1/h)&(gy<=1-1/h)
        out=F.grid_sample(x,torch.stack([gx,gy],-1),mode='bilinear',padding_mode='border',align_corners=False)
        return out,valid

    def forward(self,coarse,detail,rgb):
        high=F.interpolate(coarse,size=detail.shape[-2:],mode='bilinear',align_corners=False)
        if not self.enabled:return high
        sem=F.interpolate(self.semantic(coarse),size=detail.shape[-2:],mode='bilinear',align_corners=False)
        z=self.center(torch.cat([self.detail(detail),sem,rgb],1))
        angles=self.angles(z).tanh()*(math.pi/8)+z.new_tensor([0,math.pi/4,math.pi/2,3*math.pi/4])[None,:,None,None]
        source=torch.cat([z,rgb],1);codes=[];scores=[];valids=[]
        for i in range(4):
            for radius in (1,3,6):
                dx=angles[:,i].cos()*radius;dy=angles[:,i].sin()*radius
                a,va=self.sample(source,dx,dy);b,vb=self.sample(source,-dx,-dy)
                code=self.pair(torch.cat([z,(a+b)*.5,(a-b).abs()],1))
                valid=(va&vb)[:,None]
                codes.append(code);scores.append(self.score(code));valids.append(valid)
        logits=torch.stack(scores,1);valid=torch.stack(valids,1)
        # Explicit null candidate: corners with no valid pair get a zero pair code.
        logits=logits.masked_fill(~valid,-1e4)
        weights=torch.cat([logits,torch.zeros_like(logits[:,:1])],1).softmax(1)[:,:-1]
        paired=(torch.stack(codes,1)*weights).sum(1)
        if not self.pair_enabled:paired=torch.zeros_like(paired)
        return high+self.inject(self.mix(torch.cat([z,paired],1)))

