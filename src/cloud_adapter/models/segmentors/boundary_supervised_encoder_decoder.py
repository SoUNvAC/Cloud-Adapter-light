import torch
import torch.nn.functional as F
from mmseg.models.segmentors import EncoderDecoder
from mmseg.registry import MODELS

from .distill_losses import label_boundaries


@MODELS.register_module()
class BoundarySupervisedEncoderDecoder(EncoderDecoder):
    """Add a training-only class/boundary-weighted semantic CE objective."""

    def __init__(
        self,
        auxiliary_class_weights=(1.0, 1.0, 1.5, 1.25),
        auxiliary_boundary_weight=1.5,
        auxiliary_boundary_radius=1,
        auxiliary_loss_weight=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if len(auxiliary_class_weights) != int(self.decode_head.num_classes):
            raise ValueError("auxiliary_class_weights must match num_classes")
        if any(float(value) <= 0 for value in auxiliary_class_weights):
            raise ValueError("auxiliary_class_weights must be positive")
        if auxiliary_boundary_weight < 0 or auxiliary_loss_weight < 0:
            raise ValueError("auxiliary loss weights must be non-negative")
        if int(auxiliary_boundary_radius) != auxiliary_boundary_radius:
            raise ValueError("auxiliary_boundary_radius must be an integer")
        if auxiliary_boundary_radius < 0:
            raise ValueError("auxiliary_boundary_radius must be non-negative")
        self.register_buffer(
            "auxiliary_class_weights",
            torch.tensor(auxiliary_class_weights, dtype=torch.float32),
            persistent=False,
        )
        self.auxiliary_boundary_weight = float(auxiliary_boundary_weight)
        self.auxiliary_boundary_radius = int(auxiliary_boundary_radius)
        self.auxiliary_loss_weight = float(auxiliary_loss_weight)

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
                labels.unsqueeze(1).float(),
                size=logits.shape[-2:],
                mode="nearest",
            ).squeeze(1).long()
        valid = labels != self.decode_head.ignore_index
        safe_labels = labels.clamp(0, self.decode_head.num_classes - 1)
        pixel_ce = F.cross_entropy(
            logits.float(),
            labels,
            ignore_index=self.decode_head.ignore_index,
            reduction="none",
        )
        weights = self.auxiliary_class_weights[safe_labels].to(pixel_ce.dtype)
        boundaries = label_boundaries(
            labels, valid, self.auxiliary_boundary_radius
        )
        weights = weights * (
            1.0 + self.auxiliary_boundary_weight * boundaries.to(weights.dtype)
        )
        weights = weights * valid.to(weights.dtype)
        denominator = weights.sum().clamp_min(1.0)
        losses["aux_boundary.loss_ce"] = (
            (pixel_ce * weights).sum() / denominator * self.auxiliary_loss_weight
        )
        return losses
