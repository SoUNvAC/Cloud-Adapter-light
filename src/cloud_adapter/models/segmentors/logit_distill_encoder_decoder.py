from glob import glob
from pathlib import Path

import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmseg.models.segmentors import EncoderDecoder
from mmseg.registry import MODELS


@MODELS.register_module()
class LogitDistillEncoderDecoder(EncoderDecoder):
    """EncoderDecoder student with a frozen online segmentation teacher.

    Only semantic logits are distilled. The teacher is used in ``loss`` mode,
    excluded from optimizer updates, and removed from saved state dictionaries
    so the resulting checkpoint remains a standalone student checkpoint.
    """

    def __init__(
        self,
        teacher_config: str,
        teacher_checkpoint: str,
        distill_temperature: float = 2.0,
        distill_weight: float = 2.0,
        teacher_fp16: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if distill_temperature <= 0:
            raise ValueError("distill_temperature must be positive")
        if distill_weight < 0:
            raise ValueError("distill_weight must be non-negative")

        teacher_cfg = Config.fromfile(teacher_config)
        self.teacher = MODELS.build(teacher_cfg.model)
        self.teacher_checkpoint = teacher_checkpoint
        self.distill_temperature = float(distill_temperature)
        self.distill_weight = float(distill_weight)
        self.teacher_fp16 = bool(teacher_fp16)
        self._teacher_checkpoint_loaded = False
        self._freeze_teacher()

    def _resolve_teacher_checkpoint(self) -> str:
        matches = sorted(glob(self.teacher_checkpoint))
        if not matches and Path(self.teacher_checkpoint).is_file():
            matches = [self.teacher_checkpoint]
        if len(matches) != 1:
            raise FileNotFoundError(
                "Expected exactly one teacher checkpoint matching "
                f"{self.teacher_checkpoint!r}, found {len(matches)}"
            )
        return matches[0]

    def _freeze_teacher(self) -> None:
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def init_weights(self) -> None:
        super().init_weights()
        if not self._teacher_checkpoint_loaded:
            checkpoint = self._resolve_teacher_checkpoint()
            load_checkpoint(self.teacher, checkpoint, map_location="cpu", strict=False)
            self._teacher_checkpoint_loaded = True
        self._freeze_teacher()

    def train(self, mode: bool = True):
        result = super().train(mode)
        self._freeze_teacher()
        return result

    def _semantic_logits(self, model, features, batch_img_metas):
        return model.decode_head.predict(features, batch_img_metas, model.test_cfg)

    def loss(self, inputs, data_samples):
        batch_img_metas = [sample.metainfo for sample in data_samples]

        # no_grad, rather than inference_mode, produces ordinary tensors that
        # can safely participate as constants in the student's backward graph.
        with torch.no_grad(), torch.autocast(
            device_type=inputs.device.type,
            dtype=torch.float16,
            enabled=self.teacher_fp16 and inputs.is_cuda,
        ):
            teacher_features = self.teacher.extract_feat(inputs)
            teacher_logits = self._semantic_logits(
                self.teacher, teacher_features, batch_img_metas
            )
        del teacher_features
        teacher_logits = teacher_logits.float()

        student_features = self.extract_feat(inputs)
        losses = self._decode_head_forward_train(student_features, data_samples)
        if self.with_auxiliary_head:
            losses.update(
                self._auxiliary_head_forward_train(student_features, data_samples)
            )

        student_logits = self._semantic_logits(
            self, student_features, batch_img_metas
        ).float()
        if teacher_logits.shape[-2:] != student_logits.shape[-2:]:
            teacher_logits = F.interpolate(
                teacher_logits,
                size=student_logits.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )

        temperature = self.distill_temperature
        teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
        student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
        pixel_kl = F.kl_div(
            student_log_prob,
            teacher_prob,
            reduction="none",
        ).sum(dim=1)

        labels = torch.stack(
            [sample.gt_sem_seg.data for sample in data_samples], dim=0
        ).squeeze(1)
        if labels.shape[-2:] != pixel_kl.shape[-2:]:
            labels = F.interpolate(
                labels.unsqueeze(1).float(),
                size=pixel_kl.shape[-2:],
                mode="nearest",
            ).squeeze(1).long()
        valid = labels != self.decode_head.ignore_index
        if valid.any():
            loss_kd = pixel_kl[valid].mean()
        else:
            loss_kd = pixel_kl.mean() * 0.0

        losses["distill.loss_kd"] = (
            loss_kd * temperature * temperature * self.distill_weight
        )
        return losses

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )
        teacher_prefix = f"{prefix}teacher."
        for key in [key for key in state if key.startswith(teacher_prefix)]:
            state.pop(key)
        return state
