import torch
import torch.nn.functional as F
from mmseg.models.segmentors import EncoderDecoder
from mmseg.registry import MODELS


@MODELS.register_module()
class FactorizedEncoderDecoder(EncoderDecoder):
    """Calibrate flat cloud logits through three physically meaningful factors.

    The source label order is clear, thick cloud, thin cloud, cloud shadow.
    With a zero residual adapter the factorization reconstructs the original
    four-way softmax exactly.  The adapter can therefore be trained without an
    initialization-time change to the frozen source model.
    """

    def __init__(
        self,
        factor_loss_weight=1.0,
        reconstruction_loss_weight=1.0,
        freeze_base=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if int(self.decode_head.num_classes) != 4:
            raise ValueError("FactorizedEncoderDecoder requires four classes")
        self.factor_loss_weight = float(factor_loss_weight)
        self.reconstruction_loss_weight = float(reconstruction_loss_weight)
        self.freeze_base = bool(freeze_base)
        self.factor_residual = torch.nn.Conv2d(4, 3, kernel_size=1, bias=True)
        torch.nn.init.zeros_(self.factor_residual.weight)
        torch.nn.init.zeros_(self.factor_residual.bias)
        if self.freeze_base:
            for name, parameter in self.named_parameters():
                parameter.requires_grad = name.startswith("factor_residual.")

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_base:
            self.backbone.eval()
            if self.with_neck:
                self.neck.eval()
            self.decode_head.eval()
            self.factor_residual.train(mode)
        return self

    @staticmethod
    def _base_factor_logits(logits):
        clear, thick, thin, shadow = logits.unbind(dim=1)
        non_shadow = torch.logsumexp(logits[:, :3], dim=1)
        cloudy = torch.logsumexp(torch.stack((thick, thin), dim=1), dim=1)
        return torch.stack(
            (shadow - non_shadow, cloudy - clear, thick - thin), dim=1
        )

    def _factor_logits(self, logits):
        return self._base_factor_logits(logits) + self.factor_residual(logits)

    @staticmethod
    def _reconstruct(factors):
        shadow = factors[:, 0].sigmoid()
        cloudy = factors[:, 1].sigmoid()
        thick_given_cloud = factors[:, 2].sigmoid()
        non_shadow = 1.0 - shadow
        probabilities = torch.stack(
            (
                non_shadow * (1.0 - cloudy),
                non_shadow * cloudy * thick_given_cloud,
                non_shadow * cloudy * (1.0 - thick_given_cloud),
                shadow,
            ),
            dim=1,
        )
        return probabilities.clamp_min(1e-7).log()

    def _base_logits(self, inputs, batch_img_metas):
        with torch.no_grad() if self.freeze_base else torch.enable_grad():
            features = self.extract_feat(inputs)
            return self.decode_head.predict(features, batch_img_metas, self.test_cfg)

    def encode_decode(self, inputs, batch_img_metas):
        logits = self._base_logits(inputs, batch_img_metas)
        return self._reconstruct(self._factor_logits(logits))

    def loss(self, inputs, data_samples):
        batch_img_metas = [sample.metainfo for sample in data_samples]
        base_logits = self._base_logits(inputs, batch_img_metas)
        factors = self._factor_logits(base_logits)
        reconstructed = self._reconstruct(factors)
        labels = torch.stack(
            [sample.gt_sem_seg.data for sample in data_samples], dim=0
        ).squeeze(1)
        if labels.shape[-2:] != reconstructed.shape[-2:]:
            labels = F.interpolate(
                labels.unsqueeze(1).float(),
                size=reconstructed.shape[-2:],
                mode="nearest",
            ).squeeze(1).long()
        valid = labels != self.decode_head.ignore_index

        shadow_target = (labels == 3).to(factors.dtype)
        cloudy_target = ((labels == 1) | (labels == 2)).to(factors.dtype)
        thick_target = (labels == 1).to(factors.dtype)
        non_shadow = valid & (labels != 3)
        cloud_only = valid & ((labels == 1) | (labels == 2))

        def masked_bce(logit, target, mask):
            loss = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
            weights = mask.to(loss.dtype)
            return (loss * weights).sum() / weights.sum().clamp_min(1.0)

        factor_loss = (
            masked_bce(factors[:, 0], shadow_target, valid)
            + masked_bce(factors[:, 1], cloudy_target, non_shadow)
            + masked_bce(factors[:, 2], thick_target, cloud_only)
        ) / 3.0
        reconstruction_loss = F.nll_loss(
            reconstructed,
            labels,
            ignore_index=self.decode_head.ignore_index,
        )
        return {
            "factor.loss_bce": factor_loss * self.factor_loss_weight,
            "factor.loss_reconstruction": (
                reconstruction_loss * self.reconstruction_loss_weight
            ),
        }
