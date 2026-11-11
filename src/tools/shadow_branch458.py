"""Content-matched annular references plus pixel-scale signed shadow correction."""
import torch
from torch import nn
from torch.nn import functional as F
from shadow_region458 import ShadowRegionPropagation
def box(x,k):
    x=F.pad(x,(0,0,k//2,k//2),mode='replicate');x=F.avg_pool2d(x,(k,1),stride=1)
    return F.avg_pool2d(F.pad(x,(k//2,k//2,0,0),mode='replicate'),(1,k),stride=1)
def ring(x,outer,inner):return (box(x,outer)*outer**2-box(x,inner)*inner**2)/(outer**2-inner**2)
def boundary_features(rgb,prob,guide=None):
    # The frozen shadow probability defines the local normal, not a label mask.
    shadow=box(prob[:,2:3],3) if guide is None else guide
    dx=F.pad((shadow[...,2:]-shadow[...,:-2])*.5,(1,1,0,0),mode='replicate')
    dy=F.pad((shadow[...,2:,:]-shadow[...,:-2,:])*.5,(0,0,1,1),mode='replicate')
    magnitude=(dx.square()+dy.square()+1e-12).sqrt()
    nx=dx/magnitude;ny=dy/magnitude
    chroma=rgb/(rgb.sum(1,keepdim=True)+.02)
    texture=((rgb-box(rgb,3)).square().mean(1,keepdim=True)+1e-12).sqrt()
    signal=torch.cat([(rgb+.02).log(),chroma,texture,prob[:,:1],prob[:,2:3]],1)
    b,_,h,w=rgb.shape
    yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=rgb.device),torch.linspace(-1,1,w,device=rgb.device),indexing='ij')
    grid=torch.stack([xx,yy],-1)[None].expand(b,-1,-1,-1)
    normal=torch.cat([nx*2/(w-1),ny*2/(h-1)],1).permute(0,2,3,1)
    features=[]
    for distance in [2.,6.,12.]:
        inside=F.grid_sample(signal,grid+distance*normal,mode='bilinear',padding_mode='border',align_corners=True)
        outside=F.grid_sample(signal,grid-distance*normal,mode='bilinear',padding_mode='border',align_corners=True)
        features.extend([inside-outside,signal-(inside+outside)*.5])
    return torch.cat([*features,magnitude],1)

class LocalOcclusionBranch(nn.Module):
    def __init__(self):
        super().__init__();self.detail=nn.Conv2d(64,24,1)
        self.semantic=nn.ModuleList([nn.Conv2d(384,16,1) for _ in range(4)])
        self.mix=nn.Sequential(nn.Conv2d(114,64,1),nn.GELU(),nn.Conv2d(64,64,3,padding=1,groups=64),nn.GELU(),nn.Conv2d(64,32,1),nn.GELU())
        self.gate=nn.Conv2d(32,1,1);nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,-1.)
        self.pixel=nn.Sequential(nn.Conv2d(41,16,1),nn.GELU(),nn.Conv2d(16,1,1));nn.init.zeros_(self.pixel[-1].weight);nn.init.zeros_(self.pixel[-1].bias)
        # Preserve common initialization and downstream RNG sequence used by445.
        with torch.random.fork_rng(devices=[]):
            self.boundary=nn.Sequential(nn.Conv2d(55,16,1),nn.GELU(),nn.Conv2d(16,16,3,padding=1,groups=16),nn.GELU(),nn.Conv2d(16,1,1))
            nn.init.zeros_(self.boundary[-1].weight);nn.init.zeros_(self.boundary[-1].bias)
        with torch.random.fork_rng(devices=[]):
            self.appearance=nn.Sequential(nn.Conv2d(55,16,1),nn.GELU(),nn.Conv2d(16,16,3,padding=1,groups=16),nn.GELU(),nn.Conv2d(16,1,1))
            nn.init.zeros_(self.appearance[-1].weight);nn.init.zeros_(self.appearance[-1].bias)
        with torch.random.fork_rng(devices=[]):
            self.region=ShadowRegionPropagation()
    def forward(self,detail,features,rgb,base):
        size=detail.shape[-2:];z=self.detail(detail)
        semantic=sum(F.interpolate(p(x),size=size,mode='bilinear',align_corners=False) for p,x in zip(self.semantic,features))/4
        prob=base.softmax(1);p=F.interpolate(prob,size=size,mode='bilinear',align_corners=False)
        # Nonzero floor permits reference retrieval even when the teacher is wrong.
        w=.1+p[:,0:1];references=[];light=[];low=F.interpolate(rgb,size=size,mode='area')
        for outer,inner in [(7,3),(15,7),(31,15)]:
            denominator=ring(w,outer,inner).clamp_min(.05)
            references.append(ring(z*w,outer,inner)/denominator)
            light.append(ring(low*w,outer,inner)/denominator)
        refs=torch.stack(references,1);weights=(F.normalize(z,dim=1)[:,None]*F.normalize(refs,dim=2)).sum(2).mul(2).softmax(1)
        ref=(refs*weights[:,:,None]).sum(1);ref_rgb=(torch.stack(light,1)*weights[:,:,None]).sum(1)
        contrast=torch.log(low.clamp_min(0)+.02)-torch.log(ref_rgb.clamp_min(0)+.02)
        chroma=low/(low.sum(1,keepdim=True)+.02);texture=(low-box(low,3)).square().mean(1,keepdim=True).sqrt()
        scene=semantic.mean((2,3),keepdim=True).expand_as(semantic)
        f=self.mix(torch.cat([z,ref,z-ref,semantic,p,contrast,chroma,texture,scene],1))
        f=f+self.region(low,semantic,p,f)
        full=F.interpolate(f,size=rgb.shape[-2:],mode='bilinear',align_corners=False)
        gate=F.interpolate(self.gate(f).sigmoid(),size=rgb.shape[-2:],mode='bilinear',align_corners=False)
        # Image-derived direction can expose omitted shadows with no teacher edge.
        luminance=(rgb*rgb.new_tensor([.2126,.7152,.0722])[None,:,None,None]).sum(1,keepdim=True)
        dark_guide=-box((luminance+.02).log(),5)
        appearance=self.appearance(boundary_features(rgb,prob,guide=dark_guide))
        raw=self.pixel(torch.cat([full,rgb,rgb-box(rgb,9),prob],1))+self.boundary(boundary_features(rgb,prob))+appearance;delta=4*torch.tanh(raw/4)*gate
        logits=torch.cat([base[:,:2],base[:,2:3]+delta],1)
        return logits,delta,gate
