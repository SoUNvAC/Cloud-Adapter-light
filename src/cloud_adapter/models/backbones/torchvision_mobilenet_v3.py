from typing import Sequence, Tuple

from mmengine.model import BaseModule
from mmseg.registry import MODELS
from torch import Tensor, nn
from torchvision.models import mobilenet_v3_large


@MODELS.register_module()
class TorchvisionMobileNetV3Large(BaseModule):
    """Torchvision MobileNetV3-Large as a four-scale segmentation backbone."""

    def __init__(
        self,
        out_indices: Sequence[int] = (3, 6, 12, 15),
        norm_eval: bool = False,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        if tuple(sorted(set(out_indices))) != tuple(out_indices):
            raise ValueError("out_indices must be unique and increasing")
        if not out_indices or min(out_indices) < 0 or max(out_indices) >= 16:
            raise ValueError("out_indices must select torchvision features 0..15")
        self.out_indices = tuple(int(index) for index in out_indices)
        self.norm_eval = bool(norm_eval)
        self.features = mobilenet_v3_large(weights=None).features

    def forward(self, inputs: Tensor) -> Tuple[Tensor, ...]:
        outputs = []
        for index, layer in enumerate(self.features):
            inputs = layer(inputs)
            if index in self.out_indices:
                outputs.append(inputs)
        return tuple(outputs)

    def train(self, mode: bool = True):
        result = super().train(mode)
        if mode and self.norm_eval:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return result
