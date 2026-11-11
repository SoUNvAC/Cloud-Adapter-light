"""Supervised shadow seeds and region connections, with anchored propagation."""
import torch
from torch import nn
from torch.nn import functional as F

def shift(x,dy,dx):
    h,w=x.shape[-2:];pad=max(abs(dy),abs(dx));z=F.pad(x,(pad,pad,pad,pad),mode='replicate')
    return z[...,pad+dy:pad+dy+h,pad+dx:pad+dx+w]

class ShadowRegionPropagation(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(54,32,1),nn.GELU(),nn.Conv2d(32,32,3,padding=1,groups=32),nn.GELU())
        self.seed=nn.Conv2d(32,1,1);nn.init.zeros_(self.seed.weight);nn.init.zeros_(self.seed.bias)
        self.semantic=nn.Conv2d(16,8,1)
        self.edge=nn.Sequential(nn.Conv2d(12,16,1),nn.GELU(),nn.Conv2d(16,1,1))
        self.strength=nn.Conv2d(32,1,1);nn.init.zeros_(self.strength.weight);nn.init.zeros_(self.strength.bias)
        self.fuse=nn.Sequential(nn.Conv2d(36,32,1),nn.GELU(),nn.Conv2d(32,32,1))
        nn.init.zeros_(self.fuse[-1].weight);nn.init.zeros_(self.fuse[-1].bias)
        self.propagation_enabled=True

    def forward(self,rgb,semantic,prob,feature):
        z=self.stem(torch.cat([feature,semantic,rgb,prob],1))
        prior=prob[:,2:3].clamp(1e-5,1-1e-5)
        seed_logit=torch.logit(prior)+self.seed(z);seed=seed_logit.sigmoid();q=seed
        embed=F.normalize(self.semantic(semantic),dim=1,eps=1e-6)
        chroma=rgb/(rgb.sum(1,keepdim=True)+.02);luma=(rgb.mean(1,keepdim=True)+.02).log()
        cues=torch.cat([embed,chroma,luma],1);strength=self.strength(z).sigmoid()
        h,w=rgb.shape[-2:];yy,xx=torch.meshgrid(torch.arange(h,device=rgb.device),torch.arange(w,device=rgb.device),indexing='ij')
        self.edge_training=[]
        for distance in (1,2,4,8):
            numerator=torch.zeros_like(q);denominator=torch.zeros_like(q)
            for dy,dx in ((0,distance),(distance,0),(0,-distance),(-distance,0)):
                valid=((yy+dy>=0)&(yy+dy<h)&(xx+dx>=0)&(xx+dx<w))[None,None]
                logit=self.edge((cues-shift(cues,dy,dx)).abs())
                weight=logit.sigmoid()*valid
                numerator=numerator+weight*shift(q,dy,dx);denominator=denominator+weight
                if distance==1:self.edge_training.append((logit,dy,dx,valid))
            propagated=numerator/denominator.clamp_min(1e-6)
            q=(1-strength)*seed+strength*propagated
        if not self.propagation_enabled:q=seed
        self.seed_training=seed_logit
        self.propagated_training=q
        return self.fuse(torch.cat([z,seed,q,q-prior,q*(1-q)],1))

    def auxiliary_loss(self,y):
        valid=(y!=255).float()[:,None];shadow=(y==2).float()[:,None]
        coverage=F.avg_pool2d(valid,4);target=F.avg_pool2d(shadow,4)/coverage.clamp_min(1e-6)
        usable=coverage>.999
        seed_loss=(F.binary_cross_entropy_with_logits(self.seed_training,target,reduction='none')*usable).sum()/usable.sum().clamp_min(1)
        propagated_loss=(F.binary_cross_entropy(self.propagated_training.clamp(1e-6,1-1e-6),target,reduction='none')*usable).sum()/usable.sum().clamp_min(1)
        positive=usable&(target>=.9);negative=usable&(target<=.1);terms=[]
        for logit,dy,dx,bounds in self.edge_training:
            neighbor_positive=shift(positive.float(),dy,dx)>.5;neighbor_negative=shift(negative.float(),dy,dx)>.5
            same=positive&neighbor_positive&bounds
            different=((positive&neighbor_negative)|(negative&neighbor_positive))&bounds
            if same.any():terms.append(F.softplus(-logit)[same].mean())
            if different.any():terms.append(F.softplus(logit)[different].mean())
        edge_loss=torch.stack(terms).mean() if terms else self.seed_training.sum()*0
        return .05*(seed_loss+propagated_loss)+.05*edge_loss,dict(seed=float(seed_loss.detach()),propagated=float(propagated_loss.detach()),edge=float(edge_loss.detach()))
