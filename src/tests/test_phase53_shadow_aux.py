import torch

from cloud_adapter.models.segmentors.shadow_aux_encoder_decoder import (
    balanced_shadow_bce,
)


def test_balanced_shadow_bce_equalizes_class_mass():
    logits = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])
    labels = torch.tensor([[[3, 0, 0, 0]]])
    loss = balanced_shadow_bce(logits, labels)
    assert torch.allclose(loss, torch.tensor(0.6931472), atol=1e-6)


def test_balanced_shadow_bce_ignores_void_pixels():
    logits = torch.tensor([[[2.0, -2.0]]], requires_grad=True)
    labels = torch.tensor([[[255, 255]]])
    loss = balanced_shadow_bce(logits, labels)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
