import torch
from mmseg.registry import MODELS

from .distill_losses import weighted_pixel_kl
from .logit_distill_encoder_decoder import LogitDistillEncoderDecoder


@MODELS.register_module()
class WeakClassBoundaryDistillEncoderDecoder(LogitDistillEncoderDecoder):
    """Semantic KD weighted toward weak classes and label boundaries.

    The teacher and these losses exist only during training. Student inference
    remains unchanged and therefore keeps the static standard-operator graph.
    """

    def __init__(
        self,
        distill_class_weights=(1.0, 1.0, 2.0, 2.0),
        boundary_weight=2.0,
        boundary_radius=1,
        **kwargs,
    ):
        if any(float(weight) <= 0 for weight in distill_class_weights):
            raise ValueError("All distill_class_weights must be positive")
        if boundary_weight < 0:
            raise ValueError("boundary_weight must be non-negative")
        if int(boundary_radius) != boundary_radius or boundary_radius < 0:
            raise ValueError("boundary_radius must be a non-negative integer")
        super().__init__(**kwargs)
        num_classes = int(self.decode_head.num_classes)
        if len(distill_class_weights) != num_classes:
            raise ValueError(
                "distill_class_weights length must match num_classes: "
                f"{len(distill_class_weights)} versus {num_classes}"
            )
        self.register_buffer(
            "distill_class_weights",
            torch.tensor(distill_class_weights, dtype=torch.float32),
            persistent=False,
        )
        self.boundary_weight = float(boundary_weight)
        self.boundary_radius = int(boundary_radius)

    def _reduce_pixel_distill_loss(self, pixel_kl, labels, valid):
        return weighted_pixel_kl(
            pixel_kl,
            labels,
            valid,
            self.distill_class_weights,
            self.boundary_weight,
            self.boundary_radius,
        )
