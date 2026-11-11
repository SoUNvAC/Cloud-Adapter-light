"""Scene-shared soft cloud/shadow displacement evidence injected into features."""
import math
import torch
from torch import nn
from torch.nn import functional as F

def shift(x,dy,dx):
    h,w=x.shape[-2:];p=max(abs(dy),abs(dx))
    z=F.pad(x,(p,p,p,p))
    return z[...,p+dy:p+dy+h,p+dx:p+dx+w]

class CloudShadowGeometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.scene=nn.Sequential(nn.Linear(67,64),nn.GELU(),nn.Linear(64,33))
        nn.init.zeros_(self.scene[-1].weight);nn.init.zeros_(self.scene[-1].bias)
        self.inject=nn.Sequential(nn.Conv2d(13,32,1),nn.GELU(),nn.Conv2d(32,32,3,padding=1,groups=32,padding_mode='replicate'),nn.GELU(),nn.Conv2d(32,64,1,bias=False))
        nn.init.zeros_(self.inject[-1].weight)

    def forward(self,prob,feature):
        # 1/4 image grid. Offsets are hypotheses, never supplied solar geometry.
        cloud=prob[:,1:2];shadow=prob[:,2:3];maps=[];valids=[];distances=[]
        for dy0,dx0 in ((0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)):
            for radius in (8,16,32,48):
                norm=math.sqrt(dy0*dy0+dx0*dx0);dy=round(radius*dy0/norm);dx=round(radius*dx0/norm)
                maps.append(shift(cloud,dy,dx));valids.append(shift(torch.ones_like(cloud),dy,dx));distances.append(radius/48)
        maps=torch.cat(maps,1);valid=torch.cat(valids,1)
        # Normalized spatial compatibility prevents cloud area alone dominating.
        denominator=(maps.square().sum((2,3))*(shadow.square()*valid).sum((2,3))+1e-6).sqrt()
        compatibility=(maps*shadow).sum((2,3))/denominator
        learned=self.scene(torch.cat([feature.mean((2,3)),prob.mean((2,3)),compatibility],1))
        weights=(learned+torch.cat([4*compatibility,torch.zeros_like(compatibility[:,:1])],1)).softmax(1)
        weighted=maps*weights[:,:32,None,None]
        b,_,h,w=maps.shape
        directional=weighted.reshape(b,8,4,h,w).sum(2)
        projected=weighted.sum(1,keepdim=True)
        distance=(weighted*maps.new_tensor(distances)[None,:,None,None]).sum(1,keepdim=True)
        availability=(valid*weights[:,:32,None,None]).sum(1,keepdim=True)
        null=weights[:,32:,None,None].expand(b,1,h,w)
        global_cloud=cloud.mean((2,3),keepdim=True).expand_as(cloud)
        # Local content and raw class probabilities cannot bypass the geometry path.
        evidence=torch.cat([directional,projected,distance,availability,null,global_cloud],1)
        modulation=self.inject(evidence)
        modulation=modulation-modulation.mean((2,3),keepdim=True)
        scale,offset=modulation.chunk(2,1)
        scale=.5*torch.tanh(scale)
        scale=scale-scale.mean((2,3),keepdim=True)
        normalized=F.layer_norm(feature.permute(0,2,3,1),(32,)).permute(0,3,1,2)
        return scale*normalized+offset
