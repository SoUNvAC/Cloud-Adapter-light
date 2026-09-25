import torch
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmseg.utils import add_prefix

from .frozen_encoder_decoder import FrozenHeadEncoderDecoder


def partial_label_nll(logits, mask, allowed_classes):
    """Mean -log(sum allowed probabilities) over a boolean pixel mask."""
    if mask.dtype != torch.bool:
        raise TypeError("partial-label mask must be boolean")
    if mask.shape != logits.shape[:1] + logits.shape[-2:]:
        raise ValueError("partial-label mask and logits must have matching N/H/W")
    allowed = torch.as_tensor(allowed_classes, dtype=torch.long, device=logits.device)
    if allowed.ndim != 1 or len(allowed) < 2:
        raise ValueError("allowed_classes must contain at least two classes")
    if int(allowed.min()) < 0 or int(allowed.max()) >= logits.shape[1]:
        raise ValueError("allowed_classes contains an out-of-range class")
    log_probability = F.log_softmax(logits.float(), dim=1)
    pixel_loss = -torch.logsumexp(log_probability[:, allowed], dim=1)
    if not mask.any():
        return logits.sum() * 0.0
    return pixel_loss[mask].mean()


@MODELS.register_module()
class PartialLabelFrozenHeadEncoderDecoder(FrozenHeadEncoderDecoder):
    """Frozen-head adapter training with parameter-free set-valued labels."""

    def __init__(
        self,
        partial_label_index=254,
        partial_classes=(0, 3),
        partial_loss_weight=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.partial_label_index = int(partial_label_index)
        self.partial_classes = tuple(int(value) for value in partial_classes)
        self.partial_loss_weight = float(partial_loss_weight)
        if self.partial_label_index == int(self.decode_head.ignore_index):
            raise ValueError("partial_label_index must differ from ignore_index")
        if len(set(self.partial_classes)) != len(self.partial_classes):
            raise ValueError("partial_classes must be unique")
        if self.partial_loss_weight <= 0:
            raise ValueError("partial_loss_weight must be positive")

    def loss(self, inputs, data_samples):
        features = self.extract_feat(inputs)
        partial_masks = []
        for sample in data_samples:
            labels = sample.gt_sem_seg.data
            partial = labels == self.partial_label_index
            partial_masks.append(partial)
            if partial.any():
                # Native CE/Dice sees only verified thin/thick labels. The
                # set-valued objective below is solely responsible for ID 0.
                sample.gt_sem_seg.data = labels.masked_fill(
                    partial, int(self.decode_head.ignore_index)
                )

        seg_logits = self.decode_head(features)
        losses = add_prefix(
            self.decode_head.loss_by_feat(seg_logits, data_samples), "decode"
        )
        if self.with_auxiliary_head:
            losses.update(self._auxiliary_head_forward_train(features, data_samples))

        partial_mask = torch.stack(partial_masks, dim=0).squeeze(1)
        if partial_mask.shape[-2:] != seg_logits.shape[-2:]:
            seg_logits = F.interpolate(
                seg_logits,
                size=partial_mask.shape[-2:],
                mode="bilinear",
                align_corners=self.decode_head.align_corners,
            )
        losses["partial.loss_set_nll"] = partial_label_nll(
            seg_logits, partial_mask, self.partial_classes
        ) * self.partial_loss_weight
        return losses
