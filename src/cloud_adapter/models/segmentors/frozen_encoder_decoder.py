from typing import List
import torch
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.models.segmentors import EncoderDecoder
from typing import Iterable


def detach_everything(everything):
    if isinstance(everything, Tensor):
        return everything.detach()
    elif isinstance(everything, Iterable):
        return [detach_everything(x) for x in everything]
    else:
        return everything


@MODELS.register_module()
class FrozenBackboneEncoderDecoder(EncoderDecoder):
    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        with torch.no_grad():
            x = self.backbone(inputs)
            x = detach_everything(x)
        if self.with_neck:
            x = self.neck(x)
        return x

@MODELS.register_module()
class FrozenHeadEncoderDecoder(EncoderDecoder):
    def train(self, mode=True):
        super().train(mode)

        self.decode_head.eval()
        for param in self.decode_head.parameters():
            param.requires_grad = False
        return self


@MODELS.register_module()
class Phase64BSourceEncoderDecoder(EncoderDecoder):
    """Retrain the existing source adapter for explicit six-band input.

    The frozen DINO transformer is unchanged.  Its six-channel patch stem,
    the existing Cloud-Adapter, and the existing source decode head are
    retrained, matching the source-model training role without introducing a
    new head, loss, or router.  Target adaptation uses
    ``FrozenHeadEncoderDecoder`` and freezes these source-trained components.
    """

    _TRAINABLE_BACKBONE_PREFIXES = (
        "patch_embed",
        "cloud_adapter",
    )

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad = name.startswith(
                self._TRAINABLE_BACKBONE_PREFIXES
            )
        for name, module in self.backbone.named_modules():
            if name.startswith(self._TRAINABLE_BACKBONE_PREFIXES):
                module.train(mode)
        return self
