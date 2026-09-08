from typing import List, Sequence, Tuple

from mmengine.model import BaseModule
from mmseg.registry import MODELS
from torch import Tensor, nn
import torch.nn.functional as F


class _ConvGN(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int = 1,
        activate: bool = True,
        num_groups: int = 32,
    ):
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(num_groups, out_channels),
        ]
        if activate:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class _DepthwiseSeparableRefine(nn.Sequential):
    def __init__(self, channels: int, num_groups: int):
        super().__init__(
            _ConvGN(
                channels,
                channels,
                kernel_size=3,
                groups=channels,
                num_groups=num_groups,
            ),
            _ConvGN(
                channels,
                channels,
                kernel_size=1,
                num_groups=num_groups,
            ),
        )


@MODELS.register_module()
class LiteFPNPixelDecoder(BaseModule):
    """Export-oriented pixel decoder for a Mask2Former query decoder.

    It preserves the interface and low-to-high feature ordering of
    ``MSDeformAttnPixelDecoder`` while replacing deformable attention with a
    lightweight top-down FPN made only from standard tensor operations.
    ``encoder`` is accepted because Mask2FormerHead inspects its configured
    feature-level count before building the pixel decoder.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        strides: Sequence[int],
        feat_channels: int,
        out_channels: int,
        num_outs: int = 3,
        norm_cfg=None,
        act_cfg=None,
        encoder=None,
        positional_encoding=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        del act_cfg, encoder, positional_encoding

        if len(in_channels) != len(strides):
            raise ValueError("in_channels and strides must have the same length")
        if len(in_channels) < 2:
            raise ValueError("LiteFPNPixelDecoder needs at least two feature levels")
        if not 1 <= num_outs < len(in_channels) + 1:
            raise ValueError("num_outs must be between 1 and the input level count")
        if norm_cfg is not None and norm_cfg.get("type", "GN") != "GN":
            raise ValueError("LiteFPNPixelDecoder currently supports GroupNorm only")

        num_groups = 32 if norm_cfg is None else norm_cfg.get("num_groups", 32)
        if feat_channels % num_groups != 0:
            raise ValueError("feat_channels must be divisible by GroupNorm groups")

        self.strides = tuple(strides)
        self.num_outs = int(num_outs)
        self.input_projections = nn.ModuleList(
            [
                _ConvGN(
                    channels,
                    feat_channels,
                    kernel_size=1,
                    activate=False,
                    num_groups=num_groups,
                )
                for channels in in_channels
            ]
        )
        self.refine_blocks = nn.ModuleList(
            [
                _DepthwiseSeparableRefine(feat_channels, num_groups)
                for _ in in_channels
            ]
        )
        self.mask_feature = nn.Conv2d(feat_channels, out_channels, kernel_size=1)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, feats: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
        if len(feats) != len(self.input_projections):
            raise ValueError(
                f"Expected {len(self.input_projections)} feature maps, "
                f"received {len(feats)}"
            )

        laterals = [
            projection(feature)
            for projection, feature in zip(self.input_projections, feats)
        ]
        pyramid = [None] * len(laterals)
        pyramid[-1] = self.refine_blocks[-1](laterals[-1])
        for level in range(len(laterals) - 2, -1, -1):
            top_down = F.interpolate(
                pyramid[level + 1],
                size=laterals[level].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pyramid[level] = self.refine_blocks[level](laterals[level] + top_down)

        # Mask2Former consumes transformer features from low to high
        # resolution, while the mask feature uses the finest FPN level.
        multi_scale_features = list(reversed(pyramid))[: self.num_outs]
        mask_feature = self.mask_feature(pyramid[0])
        return mask_feature, multi_scale_features
