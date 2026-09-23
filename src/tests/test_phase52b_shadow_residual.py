import torch

from cloud_adapter.models.segmentors.shadow_residual_encoder_decoder import (
    ShadowResidualBranch,
    shadow_boundary_band,
    shadow_tversky,
)


def test_zero_initialized_branch_is_zero_and_under_budget():
    branch = ShadowResidualBranch(feature_channels=384, hidden_channels=32, dilation=3)
    feature = torch.randn(2, 384, 8, 8)
    probability = torch.softmax(torch.randn(2, 4, 32, 32), dim=1)
    luminance = torch.rand(2, 1, 32, 32)
    output = branch(feature, probability, luminance)
    assert torch.equal(output, torch.zeros_like(output))
    assert sum(parameter.numel() for parameter in branch.parameters()) == 12833


def test_shadow_boundary_band_marks_both_sides():
    labels = torch.zeros(1, 5, 5, dtype=torch.long)
    labels[:, 2, 2] = 3
    boundary = shadow_boundary_band(labels, shadow_index=3, radius=1)
    assert boundary.sum().item() == 9
    assert boundary[0, 2, 2]


def test_tversky_is_better_for_correct_prediction():
    target = torch.tensor([[[True, False]]])
    valid = torch.ones_like(target)
    correct = shadow_tversky(torch.tensor([[[0.9, 0.1]]]), target, valid)
    wrong = shadow_tversky(torch.tensor([[[0.1, 0.9]]]), target, valid)
    assert correct < wrong
