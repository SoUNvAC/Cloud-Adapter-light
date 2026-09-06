from typing import Sequence

from torch import Tensor, nn
import torch.nn.functional as F
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS

from .light_cloud_head import _DepthwiseSeparableBlock


class _LateralProjection(nn.Sequential):
    """Project a level without clipping signed features before FPN addition."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )


@MODELS.register_module()
class LightCloudFPNHead(BaseDecodeHead):
    """Lightweight top-down FPN head for cloud segmentation.

    Unlike ``LightCloudHead``, this head does not resize every feature and
    collapse them in one operation. It starts from the deepest feature and
    progressively injects it into each finer level. A cheap separable block
    refines every merge, retaining scale-specific spatial information without
    a Mask2Former pixel or query decoder.
    """

    def __init__(
        self,
        decoder_channels: int = 64,
        num_refine_blocks: int = 1,
        **kwargs,
    ):
        kwargs.setdefault("input_transform", "multiple_select")
        super().__init__(**kwargs)

        if not isinstance(self.in_channels, Sequence):
            raise TypeError("LightCloudFPNHead expects multiple input features")
        if len(self.in_channels) < 2:
            raise ValueError("LightCloudFPNHead expects at least two feature maps")
        if decoder_channels <= 0:
            raise ValueError("decoder_channels must be positive")
        if num_refine_blocks < 0:
            raise ValueError("num_refine_blocks must be non-negative")
        if self.channels != decoder_channels:
            raise ValueError(
                "channels and decoder_channels must match because BaseDecodeHead "
                "builds the classifier from channels"
            )

        self.lateral_projections = nn.ModuleList(
            [
                _LateralProjection(channels, decoder_channels)
                for channels in self.in_channels
            ]
        )
        self.top_down_blocks = nn.ModuleList(
            [
                _DepthwiseSeparableBlock(decoder_channels)
                for _ in range(len(self.in_channels) - 1)
            ]
        )
        self.output_refine = nn.Sequential(
            *[
                _DepthwiseSeparableBlock(decoder_channels)
                for _ in range(num_refine_blocks)
            ]
        )

    def forward(self, inputs) -> Tensor:
        features = self._transform_inputs(inputs)
        laterals = [
            projection(feature)
            for feature, projection in zip(features, self.lateral_projections)
        ]

        fused = laterals[-1]
        # Input order is fine-to-coarse. Each level is merged exactly once so
        # scale identity is preserved instead of being averaged away.
        for level in range(len(laterals) - 2, -1, -1):
            fused = F.interpolate(
                fused,
                size=laterals[level].shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            fused = self.top_down_blocks[level](fused + laterals[level])

        fused = self.output_refine(fused)
        return self.cls_seg(fused)
