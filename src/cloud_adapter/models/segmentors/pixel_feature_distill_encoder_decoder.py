from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from mmseg.registry import MODELS
from torch import Tensor

from .logit_distill_encoder_decoder import LogitDistillEncoderDecoder


@MODELS.register_module()
class PixelFeatureDistillEncoderDecoder(LogitDistillEncoderDecoder):
    """Distill a deformable pixel decoder into an exportable FPN student.

    The teacher only runs through its backbone and pixel decoder. Student pixel
    features are captured from the normal supervised Mask2Former forward pass,
    avoiding a duplicate student pixel-decoder execution. As with the parent
    class, the frozen teacher is excluded from saved state dictionaries.
    """

    def __init__(
        self,
        feature_distill_weight: float = 0.5,
        mask_feature_weight: float = 1.0,
        pyramid_feature_weight: float = 1.0,
        **kwargs,
    ):
        if feature_distill_weight < 0:
            raise ValueError("feature_distill_weight must be non-negative")
        if mask_feature_weight < 0 or pyramid_feature_weight < 0:
            raise ValueError("feature component weights must be non-negative")
        # Semantic-logit KD from the parent is intentionally disabled; this
        # subclass overrides loss() and only uses the teacher utilities.
        kwargs["distill_weight"] = 0.0
        super().__init__(**kwargs)
        self.feature_distill_weight = float(feature_distill_weight)
        self.mask_feature_weight = float(mask_feature_weight)
        self.pyramid_feature_weight = float(pyramid_feature_weight)

    @staticmethod
    def _align_teacher(student: Tensor, teacher: Tensor) -> Tensor:
        if student.ndim != 4 or teacher.ndim != 4:
            raise ValueError("Pixel-decoder features must be BCHW tensors")
        if student.shape[1] != teacher.shape[1]:
            raise ValueError(
                "Student and teacher feature channels differ: "
                f"{student.shape[1]} versus {teacher.shape[1]}"
            )
        if student.shape[-2:] != teacher.shape[-2:]:
            teacher = F.interpolate(
                teacher,
                size=student.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return teacher

    @classmethod
    def _cosine_feature_loss(cls, student: Tensor, teacher: Tensor) -> Tensor:
        student = student.float()
        teacher = cls._align_teacher(student, teacher.float())
        return 1.0 - F.cosine_similarity(
            student, teacher, dim=1, eps=1e-6
        ).mean()

    def _pixel_distill_losses(
        self,
        student_output: Tuple[Tensor, Sequence[Tensor]],
        teacher_output: Tuple[Tensor, Sequence[Tensor]],
    ):
        student_mask, student_pyramid = student_output
        teacher_mask, teacher_pyramid = teacher_output
        if len(student_pyramid) != len(teacher_pyramid):
            raise ValueError(
                "Student and teacher pyramid lengths differ: "
                f"{len(student_pyramid)} versus {len(teacher_pyramid)}"
            )

        mask_loss = self._cosine_feature_loss(student_mask, teacher_mask)
        pyramid_losses = [
            self._cosine_feature_loss(student, teacher)
            for student, teacher in zip(student_pyramid, teacher_pyramid)
        ]
        pyramid_loss = torch.stack(pyramid_losses).mean()
        common_weight = self.feature_distill_weight
        return {
            "distill.loss_mask_feature": (
                mask_loss * self.mask_feature_weight * common_weight
            ),
            "distill.loss_pyramid_feature": (
                pyramid_loss * self.pyramid_feature_weight * common_weight
            ),
        }

    def loss(self, inputs, data_samples):
        with torch.no_grad(), torch.autocast(
            device_type=inputs.device.type,
            dtype=torch.float16,
            enabled=self.teacher_fp16 and inputs.is_cuda,
        ):
            teacher_backbone_features = self.teacher.extract_feat(inputs)
            teacher_pixel_output = self.teacher.decode_head.pixel_decoder(
                teacher_backbone_features
            )
        del teacher_backbone_features

        captured_outputs = []

        def capture_pixel_output(module, module_inputs, module_output):
            del module, module_inputs
            captured_outputs.append(module_output)

        handle = self.decode_head.pixel_decoder.register_forward_hook(
            capture_pixel_output
        )
        try:
            student_features = self.extract_feat(inputs)
            losses = self._decode_head_forward_train(student_features, data_samples)
            if self.with_auxiliary_head:
                losses.update(
                    self._auxiliary_head_forward_train(
                        student_features, data_samples
                    )
                )
        finally:
            handle.remove()

        if len(captured_outputs) != 1:
            raise RuntimeError(
                "Expected one student pixel-decoder forward during loss, "
                f"captured {len(captured_outputs)}"
            )
        losses.update(
            self._pixel_distill_losses(
                captured_outputs[0], teacher_pixel_output
            )
        )
        return losses
