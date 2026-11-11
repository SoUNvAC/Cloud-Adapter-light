"""Local/ring appearance contrast inside the MsRE projector, not logit correction."""
import torch
from torch import nn
from torch.nn import functional as F


def ring(x, outer, inner):
    # Exclude padded pixels and the central square from each ring.
    ones = torch.ones_like(x[:, :1])
    def total(v, k):
        return F.avg_pool2d(v, k, 1, k//2, count_include_pad=True) * (k*k)
    return (total(x, outer)-total(x, inner))/(total(ones, outer)-total(ones, inner)).clamp_min(1)


class OcclusionProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce = nn.Conv2d(384, 16, 1)
        self.compare = nn.Sequential(nn.Conv2d(89, 32, 1), nn.GELU(), nn.Conv2d(32, 384, 1))
        nn.init.zeros_(self.compare[-1].weight)
        nn.init.zeros_(self.compare[-1].bias)
        self.enabled = True
        self.ring_enabled = True
        self.rgb = None

    def forward(self, features):
        if not self.enabled:
            return torch.zeros_like(features)
        z = self.reduce(F.layer_norm(features.permute(0,2,3,1), (384,)).permute(0,3,1,2))
        # Display-RGB appearance descriptors: no physical-reflectance claim.
        rgb = F.adaptive_avg_pool2d(self.rgb, features.shape[-2:]).clamp_min(1/255).log()
        if self.ring_enabled:
            r1, r2 = ring(z,5,3), ring(z,9,5)
            a1, a2 = ring(rgb,5,3), ring(rgb,9,5)
        else:
            r1, r2, a1, a2 = z, z, rgb, rgb
        inputs = torch.cat([z,r1,r2,z-r1,z-r2,rgb,rgb-a1,rgb-a2],1)
        return self.compare(inputs)


def install(backbone, module):
    def prepare(m, args):
        x = args[0]
        mean = x.new_tensor([123.675,116.28,103.53])[None,:,None,None]
        std = x.new_tensor([58.395,57.12,57.375])[None,:,None,None]
        module.rgb = ((x*std+mean)/255).clamp(0,1)
    backbone.register_forward_pre_hook(prepare)
    backbone.refinement.projector.register_forward_hook(lambda m,a,o: o+module(a[0]))
