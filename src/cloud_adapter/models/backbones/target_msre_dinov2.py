import math
from typing import Iterable

import torch
from mmseg.models.builder import BACKBONES
from torch import Tensor, nn
import torch.nn.functional as F

from .cloud_adapter_dinov2 import CloudAdapterDinoVisionTransformer
from .utils import set_requires_grad, set_train


class SparseMsREResidual(nn.Module):
    """Sparse DINOv2-S adaptation matching the public MsRE computation.

    Only the requested transformer blocks own tokens.  The three depth-wise
    convolutions and projections are shared across injection points, as in the
    public implementation.  ``enabled=False`` is a strict identity path used
    to preserve the source sensor model.
    """

    def __init__(
        self,
        embed_dims: int = 384,
        injection_indices: Iterable[int] = (2, 5, 8, 11),
        token_length: int = 16,
        scale_init: float = 1e-3,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.injection_indices = tuple(int(index) for index in injection_indices)
        if len(set(self.injection_indices)) != len(self.injection_indices):
            raise ValueError("injection_indices must be unique")
        if token_length < 2:
            raise ValueError("token_length must be at least two")
        self.layer_slots = {index: slot for slot, index in enumerate(self.injection_indices)}
        self.embed_dims = int(embed_dims)
        self.token_length = int(token_length)
        self.enabled = bool(enabled)
        self.learnable_tokens = nn.Parameter(
            torch.empty(len(self.injection_indices), token_length, embed_dims)
        )
        self.level_encoding = nn.Embedding(3, embed_dims)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.mlp_token2feat = nn.Linear(embed_dims, embed_dims)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(embed_dims, embed_dims, kernel_size=k, padding=k // 2,
                          groups=embed_dims)
                for k in (3, 5, 7)
            ]
        )
        self.projector = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)
        val = math.sqrt(6.0 / float(3 * 14 * 14 + embed_dims))
        nn.init.uniform_(self.learnable_tokens, -val, val)
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))

    def forward(self, features: Tensor, block_index: int, height: int, width: int) -> Tensor:
        if not self.enabled or block_index not in self.layer_slots:
            return features
        cls_token, patches = torch.tensor_split(features, [1], dim=1)
        batch, patch_count, channels = patches.shape
        if patch_count != height * width:
            raise ValueError(
                f"patch grid mismatch: {patch_count} tokens for {height}x{width}"
            )
        tokens = self.learnable_tokens[self.layer_slots[block_index]]
        identity = patches
        spatial = patches.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        multi_scale = torch.stack([conv(spatial) for conv in self.convs], dim=1)
        multi_scale = multi_scale.permute(0, 1, 3, 4, 2).reshape(
            batch, 3, patch_count, channels
        )
        multi_scale = multi_scale * self.level_encoding.weight[None, :, None, :]
        attention = torch.einsum("bsnc,mc->bsnm", multi_scale, tokens)
        attention = F.softmax(attention * (channels ** -0.5), dim=-1)
        # The public MsRE implementation reserves token zero and uses the
        # remaining tokens to form the feature residual.
        delta = torch.einsum(
            "bsnm,mc->bsnc",
            attention[..., 1:],
            self.mlp_token2feat(tokens[1:]),
        )
        delta = delta.mean(dim=1) + identity
        projected = self.projector(
            delta.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        )
        projected = projected.permute(0, 2, 3, 1).reshape(batch, patch_count, channels)
        adapted = patches + self.scale * (projected + delta)
        return torch.cat([cls_token, adapted], dim=1)


@BACKBONES.register_module()
class TargetMsRECloudAdapterDinoVisionTransformer(CloudAdapterDinoVisionTransformer):
    """Frozen source Cloud-Adapter plus a sensor-specific sparse MsRE path."""

    def __init__(
        self,
        target_msre_config=None,
        target_enabled=True,
        head_delta_rank=8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        config = dict(target_msre_config or {})
        config.setdefault("embed_dims", self.embed_dim)
        config.setdefault("enabled", target_enabled)
        self.target_msre = SparseMsREResidual(**config)
        self.target_enabled = bool(target_enabled)
        self.target_head_delta = nn.ModuleList()
        for _ in self.out_indices:
            delta = nn.Sequential(
                nn.Conv2d(self.embed_dim, head_delta_rank, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(head_delta_rank, self.embed_dim, kernel_size=1),
            )
            nn.init.zeros_(delta[-1].weight)
            nn.init.zeros_(delta[-1].bias)
            self.target_head_delta.append(delta)

    def set_target_enabled(self, enabled: bool) -> None:
        self.target_enabled = bool(enabled)
        self.target_msre.enabled = bool(enabled)

    def forward_features(self, x, masks=None):
        batch, _, height, width = x.shape
        cache = self.cloud_adapter.cnn(x)
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outputs = []
        adapter_slot = 0
        for index, block in enumerate(self.blocks):
            if index not in self._active_block_index_set:
                continue
            x = block(x)
            if index in self.adapter_index:
                x = self.cloud_adapter.forward(
                    x, adapter_slot, batch_first=True, has_cls_token=True, cache=cache
                )
                adapter_slot += 1
            x = self.target_msre(x, index, grid_h, grid_w)
            if index in self.out_indices:
                feature = x[:, 1:].permute(0, 2, 1).reshape(
                    batch, -1, grid_h, grid_w
                ).contiguous()
                if self.target_enabled:
                    slot = len(outputs)
                    feature = feature + self.target_head_delta[slot](feature)
                outputs.append(feature)
        return outputs, cache

    def train(self, mode: bool = True):
        if not mode:
            return nn.Module.train(self, False)
        set_requires_grad(self, ["target_msre", "target_head_delta"])
        set_train(self, ["target_msre", "target_head_delta"])
        return self
