"""Half-resolution center-detail decoder; no paired reading in the runtime graph."""
import torch
from torch import nn
from torch.nn import functional as F
from mmdet.registry import MODELS
from mmdet.models.layers import MSDeformAttnPixelDecoder
from .init_reference import PairedBoundary

class DetailBoundary(nn.Module):
    def __init__(self):
        super().__init__()
        # Use the exact Phase500 initialization, not its trained checkpoint.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(50001)
            ref=PairedBoundary()
            self.detail=ref.detail;self.semantic=ref.semantic;self.center=ref.center
            self.mix=nn.Sequential(nn.Conv2d(16,16,3,padding=1,groups=16),nn.GELU(),nn.Conv2d(16,16,1),nn.GELU())
            self.inject=ref.inject
            with torch.no_grad():
                self.mix[0].weight.copy_(ref.mix[0].weight[:16])
                self.mix[0].bias.copy_(ref.mix[0].bias[:16])
                self.mix[2].weight.copy_(ref.mix[2].weight[:,:16])
                # Fold the removed zero-input channels' constant contribution into bias.
                constant=F.gelu(ref.mix[0].bias[16:])
                self.mix[2].bias.copy_(ref.mix[2].bias+ref.mix[2].weight[:,16:,0,0]@constant)
        self.enabled=True

    def forward(self,coarse,detail,rgb):
        high=F.interpolate(coarse,size=detail.shape[-2:],mode='bilinear',align_corners=False)
        if not self.enabled:return high
        sem=F.interpolate(self.semantic(coarse),size=detail.shape[-2:],mode='bilinear',align_corners=False)
        z=self.center(torch.cat([self.detail(detail),sem,rgb],1))
        return high+self.inject(self.mix(z))

@MODELS.register_module()
class BoundaryPixelDecoder501(MSDeformAttnPixelDecoder):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.boundary=DetailBoundary()
        self.coarse_mode=False
    def forward(self,feats):
        assert len(feats)==6
        mask,memory=super().forward(feats[:4])
        if self.coarse_mode:return mask,memory
        return self.boundary(mask,feats[4],feats[5]),memory
