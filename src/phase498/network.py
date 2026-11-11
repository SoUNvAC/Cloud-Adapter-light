"""Shared DINOv3-S+ backbone with baseline/fused spatial adapters.

Cloud-Adapter supplies the spatial-prior/cross-attention idea; MsRE supplies
multi-scale token refinement. The local read/write connection is our extension.
Only PyTorch is needed here; the official DINOv3 implementation is a dependency.
"""
from pathlib import Path
import sys
import math
import torch
from torch import nn
from torch.nn import functional as F


class SpatialPrior(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        def conv(cin, cout, stride=1, groups=1):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, groups=groups, bias=False),
                                 nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.stem = nn.Sequential(nn.Conv2d(3, width, 1, bias=False),
                                  nn.BatchNorm2d(width), nn.ReLU(inplace=True),
                                  conv(width, width, groups=width))
        self.down = nn.ModuleList([
            nn.Sequential(conv(width, width, 2, width),
                          nn.Conv2d(width, width, 1, bias=False), nn.BatchNorm2d(width))
            for _ in range(4)])

    def forward(self, image):
        values = [self.stem(image)]
        for block in self.down:
            values.append(block(values[-1]))
        size = values[-1].shape[-2:]
        context = sum(F.adaptive_avg_pool2d(x, size) for x in values)
        return context, values[2]  # 1/16 context, 1/4 detail


class ContextInteraction(nn.Module):
    def __init__(self, dim=384, width=64, rank=8, heads=8, head_dim=64):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        inner = heads * head_dim
        def linear(cin, cout):
            return nn.Sequential(nn.Linear(cin, rank, bias=False), nn.Linear(rank, cout, bias=False))
        self.q, self.k, self.v = linear(dim, inner), linear(width, inner), linear(width, inner)
        self.out = linear(inner, dim)

    def forward(self, patches, context):
        context = context.flatten(2).transpose(1, 2)
        def split(x):
            return x.reshape(x.shape[0], x.shape[1], self.heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.q(patches)), split(self.k(context)), split(self.v(context))
        update = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2)
        return patches + self.out(update)


class MultiScaleRefinement(nn.Module):
    """MsRE-style refinement, operating exclusively on image patch tokens."""
    def __init__(self, dim=384, layers=4, tokens=16):
        super().__init__()
        self.tokens = nn.Parameter(torch.empty(layers, tokens, dim))
        nn.init.uniform_(self.tokens, -math.sqrt(6 / (3 * 14 * 14 + dim)), math.sqrt(6 / (3 * 14 * 14 + dim)))
        self.convs = nn.ModuleList([nn.Conv2d(dim, dim, k, padding=k//2, groups=dim) for k in (3, 5, 7)])
        self.levels = nn.Embedding(3, dim)
        self.token_proj = nn.Linear(dim, dim)
        self.projector = nn.Conv2d(dim, dim, 1)
        self.scale = nn.Parameter(torch.full((layers,), 0.001))

    def forward(self, x, detail, slot, h, w):
        b, n, c = x.shape
        spatial = x.transpose(1, 2).reshape(b, c, h, w)
        multi = torch.stack([conv(spatial).flatten(2).transpose(1, 2) for conv in self.convs], 1)
        multi = (multi + detail[:, None]) * self.levels.weight[None, :, None]
        tokens = self.tokens[slot]
        attention = (multi @ tokens.T * c**-0.5).softmax(-1)
        delta = (attention[..., 1:] @ self.token_proj(tokens[1:])).mean(1) + x + detail
        projected = self.projector(delta.transpose(1, 2).reshape(b, c, h, w)).flatten(2).transpose(1, 2)
        return x + self.scale[slot] * (delta + projected)


class LocalReadWrite(nn.Module):
    def __init__(self, dim=384, width=64, rank=16):
        super().__init__()
        self.rank = rank
        self.q = nn.Linear(dim, rank, bias=False)
        self.kv = nn.Conv2d(width, rank*2, 1, bias=False)
        self.out = nn.Linear(rank, dim, bias=False)
        self.feedback = nn.Linear(dim, width, bias=False)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.feedback.weight)

    def read(self, x, detail, h, w):
        assert detail.shape[-2:] == (4*h, 4*w)
        b, n, c = x.shape
        kv = F.unfold(self.kv(detail), 4, stride=4).reshape(b, 2, self.rank, 16, n)
        k, v = [kv[:, i].permute(0, 3, 2, 1) for i in range(2)]
        q = self.q(F.layer_norm(x, (c,)))
        attention = (q[:, :, None] * k).sum(-1).mul(self.rank**-0.5).softmax(-1)
        return self.out((attention[..., None] * v).sum(-2))

    def write(self, x, detail, h, w):
        update = self.feedback(F.layer_norm(x, (x.shape[-1],))).transpose(1, 2).reshape(x.shape[0], -1, h, w)
        return detail + 0.01 * F.interpolate(update, size=detail.shape[-2:], mode='bilinear', align_corners=False)


class OutputFeatureResidual(nn.Module):
    """Four independent rank-8 residual branches, one network layer."""
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(384, 8, 1), nn.GELU(), nn.Conv2d(8, 384, 1))
            for _ in range(4)])
        for branch in self.branches:
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    def forward(self, feature, slot):
        return feature + self.branches[slot](feature)


class CloudBackbone(nn.Module):
    def __init__(self, repo=None, weights=None, variant='control', init_seed=42901, encoder=None):
        super().__init__()
        if variant not in ('control', 'texture'):
            raise ValueError(variant)
        self.variant = variant
        if encoder is None:
            if not repo or not weights or not Path(weights).is_file():
                raise FileNotFoundError('Set DINOV3_REPO and DINOV3_WEIGHTS to official local code and S+/16 weights')
            sys.path.insert(0, str(Path(repo).resolve()))
            from dinov3.hub.backbones import dinov3_vits16plus
            encoder = dinov3_vits16plus(pretrained=True, weights=str(Path(weights).resolve()))
        self.encoder = encoder
        assert encoder.embed_dim == 384 and encoder.patch_size == 16 and len(encoder.blocks) == 12
        assert encoder.n_storage_tokens == 4
        self.indices = (2, 5, 8, 11)
        # Isolate initialization so extra fused parameters cannot change head/data RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(init_seed)
            self.spatial = SpatialPrior()
            self.interactions = nn.ModuleList([ContextInteraction() for _ in self.indices])
        if variant == 'control':
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(init_seed + 1)
                self.refinement = MultiScaleRefinement()
                self.local = LocalReadWrite()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(init_seed + 2)
            self.output_residual = OutputFeatureResidual()
        if variant == 'texture':
            from .texture import ShadowRedistribution
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(init_seed + 3)
                self.texture_adapter = ShadowRedistribution(ordinal=False)
        self.encoder.requires_grad_(False)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        return self

    def forward(self, image, adapt=True):
        if image.shape[-2] % 32 or image.shape[-1] % 32:
            raise ValueError('Input dimensions must be multiples of 32 for the four-level decoder pyramid')
        context, detail = self.spatial(image) if adapt else (None, None)
        x, (h, w) = self.encoder.prepare_tokens_with_masks(image)
        prefix = 1 + self.encoder.n_storage_tokens
        rope = self.encoder.rope_embed(H=h, W=w)
        outputs = []
        if adapt and self.variant == 'texture':
            mean=image.new_tensor([123.675,116.28,103.53])[None,:,None,None]
            std=image.new_tensor([58.395,57.12,57.375])[None,:,None,None]
            self.texture_adapter.rgb=((image*std+mean)/255).clamp(0,1)
            self.texture_adapter.prepare_evidence()
        for index, block in enumerate(self.encoder.blocks):
            # No no_grad here: gradients must cross frozen blocks to earlier adapters.
            x = block(x, rope)
            if index not in self.indices:
                continue
            slot = self.indices.index(index)
            patches = x[:, prefix:]
            assert patches.shape[1] == h*w
            if adapt:
                stage_context = self.texture_adapter.early_context(context, slot) if self.variant == 'texture' else context
                patches = self.interactions[slot](patches, stage_context)
                if self.variant == 'control':
                    local = self.local.read(patches, detail, h, w)
                    patches = self.refinement(patches, local, slot, h, w)
                    if slot < 3:
                        updated = self.local.write(patches, detail, h, w)
                        context = context + F.adaptive_avg_pool2d(updated-detail, (h, w))
                        detail = updated
                x = torch.cat((x[:, :prefix], patches), dim=1)
            feature = self.encoder.norm(patches).transpose(1, 2).reshape(image.shape[0], 384, h, w)
            if adapt:
                feature = self.output_residual(feature, slot)
            pyramid = F.interpolate(feature, scale_factor=(4, 2, 1, 0.5)[slot], mode='bilinear', align_corners=False)
            if adapt and self.variant == 'texture' and slot == 0:
                pyramid = pyramid + self.texture_adapter(feature, detail)
            outputs.append(pyramid)
        return tuple(outputs)
