import torch

from cloud_adapter.models.backbones.target_msre_dinov2 import SparseMsREResidual


def test_sparse_msre_disabled_is_exact_identity():
    module = SparseMsREResidual(
        embed_dims=24, injection_indices=(2, 5), token_length=4, enabled=False
    )
    features = torch.randn(2, 17, 24)
    output = module(features, 2, 4, 4)
    assert output is features


def test_sparse_msre_only_changes_registered_blocks():
    module = SparseMsREResidual(
        embed_dims=24, injection_indices=(2, 5), token_length=4
    )
    features = torch.randn(2, 17, 24)
    assert module(features, 3, 4, 4) is features
    output = module(features, 2, 4, 4)
    assert output.shape == features.shape
    assert torch.equal(output[:, :1], features[:, :1])


def test_phase52_parameter_budget_for_dinov2_small():
    module = SparseMsREResidual(
        embed_dims=384, injection_indices=(2, 5, 8, 11), token_length=16
    )
    trainable = sum(parameter.numel() for parameter in module.parameters())
    assert trainable == 354433
    head_delta = 4 * ((384 * 8 + 8) + (8 * 384 + 384))
    assert trainable + head_delta == 380577
    assert trainable + head_delta <= 500000
