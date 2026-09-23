import torch

from cloud_adapter.models.phase54_shadow_probe import (
    CONDITIONS,
    FixedShadowProbe,
    apply_shadow_residual,
    condition_mask,
    neighbouring_cloud_probability,
)


def test_all_conditions_keep_identical_parameter_count():
    counts = []
    for _condition in CONDITIONS:
        model = FixedShadowProbe()
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
    assert len(set(counts)) == 1
    assert counts[0] == 12929


def test_condition_masks_are_nested():
    masks = [condition_mask(condition) for condition in CONDITIONS]
    assert [int(mask.sum()) for mask in masks] == [0, 1, 3, 7, 8]
    for left, right in zip(masks, masks[1:]):
        assert torch.all(left <= right)


def test_zero_initialization_exactly_reproduces_base_probabilities():
    torch.manual_seed(4)
    model = FixedShadowProbe()
    feature = torch.randn(2, 384, 8, 8)
    auxiliary = torch.randn(2, 8, 8, 8)
    probabilities = torch.softmax(torch.randn(2, 4, 8, 8), dim=1)
    residual = model(feature, auxiliary, CONDITIONS[-1])
    result = apply_shadow_residual(probabilities, residual)
    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.allclose(result, probabilities, atol=2e-7, rtol=0)


def test_nonshadow_conditional_ratios_are_preserved():
    probabilities = torch.tensor([0.4, 0.2, 0.3, 0.1]).view(1, 4, 1, 1)
    result = apply_shadow_residual(probabilities, torch.ones(1, 1, 1, 1))
    old = probabilities[:, :3] / probabilities[:, :3].sum(dim=1, keepdim=True)
    new = result[:, :3] / result[:, :3].sum(dim=1, keepdim=True)
    assert torch.allclose(old, new)
    assert torch.allclose(result.sum(dim=1), torch.ones(1, 1, 1))


def test_neighbouring_cloud_probability_excludes_centre():
    cloud = torch.zeros(1, 1, 9, 9)
    cloud[:, :, 4, 4] = 1
    context = neighbouring_cloud_probability(cloud, outer=5, inner=3)
    assert context[0, 0, 4, 4] == 0

