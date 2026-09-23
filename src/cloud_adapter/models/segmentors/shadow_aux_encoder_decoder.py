import torch
import torch.nn.functional as F
from mmseg.registry import MODELS

from .frozen_encoder_decoder import FrozenHeadEncoderDecoder


def balanced_shadow_bce(shadow_logit, labels, ignore_index=255, shadow_index=3):
    """Equal-mass positive/negative BCE for a rare cloud-shadow class."""
    valid = labels != int(ignore_index)
    positive = valid & (labels == int(shadow_index))
    negative = valid & ~positive
    target = positive.to(shadow_logit.dtype)
    loss = F.binary_cross_entropy_with_logits(shadow_logit, target, reduction="none")

    terms = []
    if positive.any():
        terms.append(loss[positive].mean())
    if negative.any():
        terms.append(loss[negative].mean())
    if not terms:
        return shadow_logit.sum() * 0.0
    return torch.stack(terms).mean()


@MODELS.register_module()
class ShadowAuxFrozenHeadEncoderDecoder(FrozenHeadEncoderDecoder):
    """Phase 53: add one parameter-free target shadow supervision mechanism."""

    def __init__(self, shadow_aux_weight=0.5, shadow_index=3, **kwargs):
        super().__init__(**kwargs)
        if float(shadow_aux_weight) <= 0:
            raise ValueError("shadow_aux_weight must be positive")
        if int(shadow_index) != shadow_index:
            raise ValueError("shadow_index must be an integer")
        self.shadow_aux_weight = float(shadow_aux_weight)
        self.shadow_index = int(shadow_index)

    def loss(self, inputs, data_samples):
        batch_img_metas = [sample.metainfo for sample in data_samples]
        features = self.extract_feat(inputs)
        losses = self._decode_head_forward_train(features, data_samples)
        if self.with_auxiliary_head:
            losses.update(self._auxiliary_head_forward_train(features, data_samples))

        logits = self.decode_head.predict(features, batch_img_metas, self.test_cfg)
        labels = torch.stack(
            [sample.gt_sem_seg.data for sample in data_samples], dim=0
        ).squeeze(1)
        if labels.shape[-2:] != logits.shape[-2:]:
            labels = F.interpolate(
                labels.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest"
            ).squeeze(1).long()
        shadow = logits[:, self.shadow_index]
        non_shadow = torch.logsumexp(
            torch.cat(
                [logits[:, :self.shadow_index], logits[:, self.shadow_index + 1:]],
                dim=1,
            ),
            dim=1,
        )
        losses["shadow_aux.loss_bce"] = (
            balanced_shadow_bce(
                shadow - non_shadow,
                labels,
                ignore_index=self.decode_head.ignore_index,
                shadow_index=self.shadow_index,
            )
            * self.shadow_aux_weight
        )
        return losses
