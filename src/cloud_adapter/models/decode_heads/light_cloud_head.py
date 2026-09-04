from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


class _Projection(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class _DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.act(self.depthwise_bn(self.depthwise(x)))
        x = self.pointwise_bn(self.pointwise(x))
        return self.act(x + residual)


@MODELS.register_module()
class LightCloudHead(BaseDecodeHead):
    """Small FPN-like head for edge-oriented cloud segmentation.

    The head projects all backbone outputs to a small common width, resizes
    them to the finest feature resolution, averages them, and applies cheap
    depthwise-separable refinement. It deliberately avoids the deformable
    attention and query decoder used by Mask2Former.
    """

    def __init__(
        self,
        decoder_channels: int = 64,
        num_fuse_blocks: int = 2,
        **kwargs,
    ):
        kwargs.setdefault("input_transform", "multiple_select")
        super().__init__(**kwargs)

        if not isinstance(self.in_channels, Sequence):
            raise TypeError("LightCloudHead expects a sequence of input channels")
        if len(self.in_channels) < 2:
            raise ValueError("LightCloudHead expects at least two feature maps")
        if decoder_channels <= 0:
            raise ValueError("decoder_channels must be positive")
        if num_fuse_blocks < 0:
            raise ValueError("num_fuse_blocks must be non-negative")

        self.projections = nn.ModuleList(
            [_Projection(channels, decoder_channels) for channels in self.in_channels]
        )
        self.fuse = nn.Sequential(
            *[_DepthwiseSeparableBlock(decoder_channels) for _ in range(num_fuse_blocks)]
        )

        if self.channels != decoder_channels:
            raise ValueError(
                "channels and decoder_channels must match because BaseDecodeHead "
                "builds the classifier from channels"
            )

    def forward(self, inputs) -> Tensor:
        features = self._transform_inputs(inputs)
        target_size = features[0].shape[-2:]
        projected = []

        for feature, projection in zip(features, self.projections):
            feature = projection(feature)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            projected.append(feature)

        fused = torch.stack(projected, dim=0).mean(dim=0)
        fused = self.fuse(fused)
        return self.cls_seg(fused)
