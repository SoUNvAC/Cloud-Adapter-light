"""Learned surface/illumination consistency references for cloud-shadow segmentation."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from shadow_branch450 import box

def shift(x,dy,dx):
    h,w=x.shape[-2:];p=max(abs(dy),abs(dx))
    padded=F.pad(x,(p,p,p,p),mode='replicate')
    return padded[...,p+dy:p+dy+h,p+dx:p+dx+w]

class SurfaceIlluminationConsistency(nn.Module):
    def __init__(self):
        super().__init__()
        self.reference_score=nn.Sequential(nn.Conv2d(22,16,1),nn.GELU(),nn.Conv2d(16,1,1))
        nn.init.zeros_(self.reference_score[-1].weight);nn.init.zeros_(self.reference_score[-1].bias)
        self.fuse=nn.Sequential(nn.Conv2d(57,32,1),nn.GELU(),nn.Conv2d(32,32,3,padding=1,groups=32),nn.GELU(),nn.Conv2d(32,1,1))
        nn.init.zeros_(self.fuse[-1].weight);nn.init.zeros_(self.fuse[-1].bias)

    def forward(self,rgb,semantic,prob,feature):
        # Quarter-resolution patch statistics; no labels or thresholds at inference.
        mean=box(rgb,5);var=(box(rgb.square(),5)-mean.square()).clamp_min(0)
        chroma=mean/(mean.sum(1,keepdim=True)+.02)
        sem=F.normalize(semantic,dim=1,eps=1e-6)
        h,w=rgb.shape[-2:];yy,xx=torch.meshgrid(torch.arange(h,device=rgb.device),torch.arange(w,device=rgb.device),indexing='ij')
        evidence=[];scores=[]
        for radius in (4,12,24):
            for dy0,dx0 in ((0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)):
                dy=round(radius*dy0/math.sqrt(dy0*dy0+dx0*dx0));dx=round(radius*dx0/math.sqrt(dy0*dy0+dx0*dx0))
                ref=shift(rgb,dy,dx);ref_mean=shift(mean,dy,dx);ref_var=shift(var,dy,dx)
                covariance=box(rgb*ref,5)-mean*ref_mean
                correlation=(covariance/((var+1e-4)*(ref_var+1e-4)).sqrt()).clamp(-1,1)
                # Residual after per-channel local affine illumination compensation.
                residual=(1-covariance.square()/((var+1e-4)*(ref_var+1e-4))).clamp(0,1)
                ratio=(mean+.02).log()-(ref_mean+.02).log()
                color=chroma-shift(chroma,dy,dx)
                texture=(var+1e-4).log()-(ref_var+1e-4).log()
                similarity=(sem*shift(sem,dy,dx)).sum(1,keepdim=True)
                ref_prob=shift(prob,dy,dx)
                distance=torch.full_like(similarity,radius/24)
                pair=torch.cat([ratio,color,correlation,residual,texture,similarity,ref_prob[:,:2],prob,distance],1)
                valid=((yy+dy>=2)&(yy+dy<h-2)&(xx+dx>=2)&(xx+dx<w-2))[None,None]
                score=2*similarity+self.reference_score(pair)+(.1+ref_prob[:,:1]).log()
                scores.append(score.masked_fill(~valid,-1e4));evidence.append(pair)
        weights=torch.stack(scores,1).softmax(1)
        summary=(torch.stack(evidence,1)*weights).sum(1)
        cloud=prob[:,1:2]
        context=torch.cat([box(cloud,15),box(cloud,31),cloud.mean((2,3),keepdim=True).expand_as(cloud)],1)
        return self.fuse(torch.cat([feature,summary,context],1))
