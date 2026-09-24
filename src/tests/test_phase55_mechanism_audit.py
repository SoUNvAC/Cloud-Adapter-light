import numpy as np
import torch

from tools.run_phase55_mechanism_audit import (
    binary_auc,
    fit_balanced_ridge,
    apply_probe,
    principal_subspace_metrics,
)


def test_binary_auc_handles_ties_and_perfect_order():
    assert binary_auc([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0]) == 1.0
    assert binary_auc([0, 1], [0.5, 0.5]) == 0.5


def test_balanced_ridge_separates_a_linear_problem():
    features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
    labels = np.asarray([0, 0, 1, 1])
    probe = fit_balanced_ridge(features, labels, ridge=0.01)
    scores = apply_probe(probe, features)
    assert np.all(scores[:2] < 0.0)
    assert np.all(scores[2:] > 0.0)


def test_principal_angles_detect_shared_and_orthogonal_directions():
    shared = principal_subspace_metrics(
        [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])],
        [torch.tensor([-1.0, 0.0]), torch.tensor([-1.0, 0.0])],
    )
    assert shared["smallest_principal_angle_degrees"] < 1e-6
    assert shared["shadow_gradient_energy_in_thin_subspace"] > 0.999
    assert shared["aggregate_signed_gradient_cosine"] < -0.999

    orthogonal = principal_subspace_metrics(
        [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])],
        [torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])],
    )
    assert orthogonal["smallest_principal_angle_degrees"] > 89.9
    assert orthogonal["shadow_gradient_energy_in_thin_subspace"] < 1e-9
