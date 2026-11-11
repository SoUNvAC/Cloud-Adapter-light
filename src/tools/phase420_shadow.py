"""Shadow-only odds correction with a frozen non-shadow conditional distribution."""
import torch
from torch import nn
import torch.nn.functional as F

def normalize(scores):
    scores=scores.clamp_min(0)
    return scores/scores.sum(1,keepdim=True).clamp_min(1e-12)

def correct(prob,delta):
    weight=torch.exp(4*torch.tanh(delta/4))
    weighted=torch.cat([prob[:,:2],prob[:,2:3]*weight],1)
    return weighted/weighted.sum(1,keepdim=True).clamp_min(1e-12)

class ShadowCorrection(nn.Module):
    def __init__(self,mode):
        super().__init__();assert mode in ['global','spatial'];self.mode=mode
        self.projection=nn.Conv2d(384,16,1)
        self.local=nn.Sequential(nn.Conv2d(24,24,3,padding=1,groups=24),nn.GELU(),nn.Conv2d(24,16,1),nn.GELU(),nn.Conv2d(16,1,1))
        nn.init.zeros_(self.local[-1].weight);nn.init.zeros_(self.local[-1].bias)
    def forward(self,feature,prob,rgb):
        size=(rgb.shape[-2]//4,rgb.shape[-1]//4)
        feature=F.layer_norm(feature.permute(0,2,3,1),(384,)).permute(0,3,1,2)
        feat=F.interpolate(self.projection(feature),size=size,mode='bilinear',align_corners=False)
        colour=F.interpolate(rgb,size=size,mode='bilinear',align_corners=False)
        lum=(colour*colour.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
        small=F.avg_pool2d(F.pad(lum,(2,2,2,2),mode='replicate'),5,stride=1)
        large=F.avg_pool2d(F.pad(lum,(8,8,8,8),mode='replicate'),17,stride=1)
        dx=F.pad(lum[:,:,:,1:]-lum[:,:,:,:-1],(0,1,0,0));dy=F.pad(lum[:,:,1:]-lum[:,:,:-1],(0,0,0,1))
        cues=torch.cat([lum,lum-small,lum-large,dx,dy],1)
        p=F.interpolate(prob,size=size,mode='bilinear',align_corners=False)
        delta=self.local(torch.cat([feat,p,cues],1))
        if self.mode=='global':delta=delta.mean((2,3),keepdim=True).expand_as(delta)
        return F.interpolate(delta,size=rgb.shape[-2:],mode='bilinear',align_corners=False)

def self_test():
    torch.manual_seed(42001)
    p=torch.rand(2,3,19,23);p=normalize(p)
    for d in [torch.zeros(2,1,19,23),torch.randn(2,1,19,23)*100]:
        q=correct(p,d);assert torch.isfinite(q).all()
        assert (q[:,0]*p[:,1]-q[:,1]*p[:,0]).abs().max()<1e-6
        a=p.argmax(1);b=q.argmax(1)
        assert not (((a==0)&(b==1))|((a==1)&(b==0))).any()
    records={}
    for mode in ['global','spatial']:
        torch.manual_seed(4200101);m=ShadowCorrection(mode)
        feat=torch.randn(1,384,4,4);prob=normalize(torch.rand(1,3,64,64));rgb=torch.rand(1,3,64,64)
        delta=m(feat,prob,rgb);assert torch.count_nonzero(delta)==0
        q=correct(prob,delta);torch.testing.assert_close(q,prob,rtol=1e-6,atol=1e-7)
        (-q[:,2].clamp_min(1e-8).log().mean()).backward()
        assert m.local[-1].weight.grad.abs().sum()>0
        count=sum(p.numel() for p in m.parameters());assert count==6817,count
        records[mode]={'parameters':count,'zero_initial_identity':True,'shadow_gradient':True,'nonshadow_order_invariant':True}
    return records
