import torch
from torch import nn
import torch.nn.functional as F
from mmseg.registry import MODELS

from mmseg.models.segmentors import EncoderDecoder
from .shadow_aux_encoder_decoder import balanced_shadow_bce


class ShadowResidualBranch(nn.Module):
    """Small zero-initialized correction for shadow versus non-shadow only."""

    def __init__(self, feature_channels=384, hidden_channels=32, dilation=3):
        super().__init__()
        input_channels = int(feature_channels) + 4 + 1
        hidden_channels = int(hidden_channels)
        self.project = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(
                hidden_channels, hidden_channels, kernel_size=3,
                padding=int(dilation), dilation=int(dilation),
                groups=hidden_channels,
            ),
            nn.GELU(),
        )
        self.output = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, feature, base_probabilities, luminance):
        size = feature.shape[-2:]
        probabilities = F.interpolate(
            base_probabilities, size=size, mode="bilinear", align_corners=False
        )
        luminance = F.interpolate(
            luminance, size=size, mode="bilinear", align_corners=False
        )
        hidden = self.project(torch.cat([feature, probabilities, luminance], dim=1))
        hidden = hidden + self.context(hidden)
        return self.output(hidden)


def shadow_boundary_band(labels, shadow_index=3, radius=1):
    shadow = (labels == int(shadow_index)).float().unsqueeze(1)
    kernel = 2 * int(radius) + 1
    dilated = F.max_pool2d(shadow, kernel, stride=1, padding=int(radius))
    eroded = -F.max_pool2d(-shadow, kernel, stride=1, padding=int(radius))
    return (dilated != eroded).squeeze(1)


def shadow_tversky(probability, target, valid, alpha=0.3, beta=0.7):
    probability = probability[valid]
    target = target[valid].to(probability.dtype)
    true_positive = (probability * target).sum()
    false_positive = (probability * (1.0 - target)).sum()
    false_negative = ((1.0 - probability) * target).sum()
    score = (true_positive + 1.0) / (
        true_positive + alpha * false_positive + beta * false_negative + 1.0
    )
    return 1.0 - score


@MODELS.register_module()
class FrozenPhase52ShadowResidualEncoderDecoder(EncoderDecoder):
    """Freeze Phase 52A and learn only a shadow/non-shadow residual."""

    def __init__(
        self,
        shadow_feature_index=2,
        shadow_index=3,
        shadow_hidden_channels=32,
        shadow_dilation=3,
        shadow_bce_weight=1.0,
        shadow_tversky_weight=1.0,
        shadow_boundary_weight=0.5,
        residual_enabled=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.shadow_feature_index = int(shadow_feature_index)
        self.shadow_index = int(shadow_index)
        self.shadow_bce_weight = float(shadow_bce_weight)
        self.shadow_tversky_weight = float(shadow_tversky_weight)
        self.shadow_boundary_weight = float(shadow_boundary_weight)
        self.residual_enabled = bool(residual_enabled)
        self.shadow_residual = ShadowResidualBranch(
            feature_channels=self.backbone.embed_dim,
            hidden_channels=shadow_hidden_channels,
            dilation=shadow_dilation,
        )
        self.register_buffer(
            "input_mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "input_std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1),
            persistent=False,
        )

    def set_target_enabled(self, enabled):
        self.residual_enabled = bool(enabled)
        self.backbone.set_target_enabled(bool(enabled))

    def train(self, mode=True):
        super().train(mode)
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name.startswith("shadow_residual.")
        self.backbone.eval()
        if self.with_neck:
            self.neck.eval()
        self.decode_head.eval()
        self.shadow_residual.train(mode)
        return self

    def _luminance(self, normalized_inputs):
        rgb = normalized_inputs.float() * self.input_std + self.input_mean
        return (
            0.2126 * rgb[:, 0:1]
            + 0.7152 * rgb[:, 1:2]
            + 0.0722 * rgb[:, 2:3]
        ).clamp(0.0, 255.0) / 255.0

    def _frozen_base(self, inputs, batch_img_metas):
        with torch.no_grad():
            features = self.extract_feat(inputs)
            logits = self.decode_head.predict(features, batch_img_metas, self.test_cfg)
            probabilities = logits.float().clamp_min(1e-8)
            probabilities = probabilities / probabilities.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
        return features, probabilities

    def _adapt(self, inputs, features, probabilities):
        if not self.residual_enabled:
            return probabilities.clamp_min(1e-7).log(), None
        shadow_base = probabilities[:, self.shadow_index].clamp(1e-6, 1.0 - 1e-6)
        base_logit = torch.logit(shadow_base)
        residual = self.shadow_residual(
            features[self.shadow_feature_index].detach(),
            probabilities.detach(), self._luminance(inputs),
        )
        residual = F.interpolate(
            residual, size=probabilities.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)
        shadow_logit = base_logit + residual
        shadow_probability = shadow_logit.sigmoid()
        non_shadow = probabilities.clone()
        non_shadow[:, self.shadow_index] = 0.0
        non_shadow = non_shadow / non_shadow.sum(dim=1, keepdim=True).clamp_min(1e-7)
        adapted = non_shadow * (1.0 - shadow_probability).unsqueeze(1)
        adapted[:, self.shadow_index] = shadow_probability
        return adapted.clamp_min(1e-7).log(), shadow_logit

    def encode_decode(self, inputs, batch_img_metas):
        features, probabilities = self._frozen_base(inputs, batch_img_metas)
        logits, _ = self._adapt(inputs, features, probabilities)
        return logits

    def loss(self, inputs, data_samples):
        batch_img_metas = [sample.metainfo for sample in data_samples]
        features, probabilities = self._frozen_base(inputs, batch_img_metas)
        adapted, shadow_logit = self._adapt(inputs, features, probabilities)
        labels = torch.stack(
            [sample.gt_sem_seg.data for sample in data_samples], dim=0
        ).squeeze(1)
        if labels.shape[-2:] != adapted.shape[-2:]:
            labels = F.interpolate(
                labels.unsqueeze(1).float(), size=adapted.shape[-2:], mode="nearest"
            ).squeeze(1).long()
        valid = labels != self.decode_head.ignore_index
        target = labels == self.shadow_index
        probability = shadow_logit.sigmoid()
        bce = balanced_shadow_bce(
            shadow_logit, labels, self.decode_head.ignore_index, self.shadow_index
        )
        tversky = shadow_tversky(probability, target, valid)
        boundary = shadow_boundary_band(labels, self.shadow_index, radius=1) & valid
        boundary_loss = F.binary_cross_entropy_with_logits(
            shadow_logit[boundary], target[boundary].to(shadow_logit.dtype)
        ) if boundary.any() else shadow_logit.sum() * 0.0
        return {
            "shadow.loss_bce": bce * self.shadow_bce_weight,
            "shadow.loss_tversky": tversky * self.shadow_tversky_weight,
            "shadow.loss_boundary": boundary_loss * self.shadow_boundary_weight,
        }
