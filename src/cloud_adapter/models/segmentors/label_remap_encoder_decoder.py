import torch

from mmseg.models import EncoderDecoder
from mmseg.registry import MODELS


@MODELS.register_module()
class LabelRemapEncoderDecoder(EncoderDecoder):
    """Remap predicted class IDs for fixed-taxonomy cross-dataset evaluation."""

    def __init__(self, label_remap, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer(
            "label_remap",
            torch.as_tensor(label_remap, dtype=torch.long),
            persistent=False,
        )

    def postprocess_result(self, seg_logits, data_samples):
        results = super().postprocess_result(seg_logits, data_samples)
        for result in results:
            prediction = result.pred_sem_seg.data
            result.pred_sem_seg.data = self.label_remap[prediction]
        return results
