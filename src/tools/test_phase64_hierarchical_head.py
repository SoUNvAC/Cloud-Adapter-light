#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "cloud_adapter" / "models" / "phase64_hierarchical_label_space.py"
)
SPEC = importlib.util.spec_from_file_location("phase64_hierarchical_label_space", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

PHASE64_HIERARCHIES = MODULE.PHASE64_HIERARCHIES
HierarchicalLabelSpaceHead = MODULE.HierarchicalLabelSpaceHead
grouped_native_probabilities = MODULE.grouped_native_probabilities
hierarchy_consistency_error = MODULE.hierarchy_consistency_error
native_nll_loss = MODULE.native_nll_loss
parent_cross_entropy_loss = MODULE.parent_cross_entropy_loss
parent_distillation_loss = MODULE.parent_distillation_loss


class Phase64HierarchicalHeadTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(64)
        self.head = HierarchicalLabelSpaceHead(8, PHASE64_HIERARCHIES)
        self.features = torch.randn(2, 8, 5, 7, requires_grad=True)

    def test_all_datasets_are_consistent_by_construction(self) -> None:
        for dataset_id, spec in PHASE64_HIERARCHIES.items():
            output = self.head(self.features, dataset_id)
            self.assertEqual(output.native_log_probs.shape[1], len(spec.native_classes))
            grouped = grouped_native_probabilities(output.native_log_probs, spec)
            self.assertTrue(torch.allclose(grouped.sum(dim=1), torch.ones_like(grouped[:, 0])))
            self.assertLess(float(hierarchy_consistency_error(output, spec)), 1e-6)

    def test_native_and_parent_losses_are_finite_and_trainable(self) -> None:
        output = self.head(self.features, "sparcs")
        native_target = torch.randint(0, 7, (2, 5, 7))
        parent_target = torch.randint(0, 3, (2, 5, 7))
        loss = native_nll_loss(output, native_target) + parent_cross_entropy_loss(
            output, parent_target
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(self.features.grad).all())

    def test_source_distillation_accepts_normalized_or_unnormalized_mass(self) -> None:
        output = self.head(self.features, "cloudsen12_high")
        teacher = torch.rand_like(output.parent_log_probs).exp()
        loss = parent_distillation_loss(output, teacher)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
