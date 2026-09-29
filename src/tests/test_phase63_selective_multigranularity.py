from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "cloud_adapter"
    / "models"
    / "phase63_selective_multigranularity.py"
)
SPEC = importlib.util.spec_from_file_location("phase63_selective_multigranularity", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

ROUTE_ABSTAIN = MODULE.ROUTE_ABSTAIN
ROUTE_COARSE = MODULE.ROUTE_COARSE
ROUTE_FINE = MODULE.ROUTE_FINE
FINE_CLASSES = MODULE.FINE_CLASSES
SelectiveMultiGranularityHead = MODULE.SelectiveMultiGranularityHead
fine_logits_to_hierarchy = MODULE.fine_logits_to_hierarchy
hierarchy_conflict_score = MODULE.hierarchy_conflict_score
route_from_calibrated_risk = MODULE.route_from_calibrated_risk
routing_coverage = MODULE.routing_coverage


class Phase63SelectiveMultiGranularityTest(unittest.TestCase):
    def test_frozen_model_output_order(self):
        self.assertEqual(
            FINE_CLASSES,
            ("clear", "thick_cloud", "thin_cloud", "cloud_shadow"),
        )

    def test_four_class_logits_map_to_normalized_consistent_hierarchies(self):
        # Frozen model order is clear, thick-cloud, thin-cloud, cloud-shadow.
        logits = torch.tensor(
            [[[[2.0, -1.0]], [[1.0, 0.0]], [[0.0, 1.0]], [[-1.0, 2.0]]]]
        )
        probabilities = fine_logits_to_hierarchy(logits)

        self.assertEqual(probabilities.fine.shape, (1, 4, 1, 2))
        self.assertEqual(probabilities.three_class.shape, (1, 3, 1, 2))
        self.assertEqual(probabilities.binary.shape, (1, 2, 1, 2))
        for output in (
            probabilities.fine,
            probabilities.three_class,
            probabilities.binary,
        ):
            self.assertTrue(torch.allclose(output.sum(dim=1), torch.ones((1, 1, 2))))

        cloud = probabilities.fine[:, 1] + probabilities.fine[:, 2]
        non_cloud = probabilities.fine[:, 0] + probabilities.fine[:, 3]
        self.assertTrue(torch.allclose(probabilities.three_class[:, 1], cloud))
        self.assertTrue(torch.allclose(probabilities.binary[:, 1], cloud))
        self.assertTrue(torch.allclose(probabilities.binary[:, 0], non_cloud))

    def test_exact_aggregation_has_zero_hierarchy_conflict(self):
        probabilities = fine_logits_to_hierarchy(torch.randn(2, 4, 3, 5))
        conflict = hierarchy_conflict_score(probabilities)
        self.assertEqual(conflict.shape, (2, 3, 5))
        self.assertTrue(torch.equal(conflict, torch.zeros_like(conflict)))

    def test_independent_coarse_head_disagreement_is_exposed(self):
        fine_logits = torch.tensor([[[[8.0]], [[-8.0]], [[-8.0]], [[-8.0]]]])
        probabilities = fine_logits_to_hierarchy(fine_logits)
        binary_logits = torch.tensor([[[[-8.0]], [[8.0]]]])
        conflict = hierarchy_conflict_score(probabilities, binary_logits=binary_logits)
        self.assertEqual(conflict.shape, (1, 1, 1))
        self.assertGreater(conflict.item(), 0.99)

    def test_external_calibrated_risk_routes_fine_coarse_and_abstain(self):
        risk = torch.tensor([0.0, 0.20, 0.21, 0.60, 0.61, float("nan")])
        route = route_from_calibrated_risk(
            risk, fine_max_risk=0.20, coarse_max_risk=0.60
        )
        self.assertEqual(
            route.tolist(),
            [ROUTE_FINE, ROUTE_FINE, ROUTE_COARSE, ROUTE_COARSE, ROUTE_ABSTAIN, ROUTE_ABSTAIN],
        )

    def test_routing_coverage_respects_valid_mask(self):
        route = torch.tensor(
            [ROUTE_FINE, ROUTE_FINE, ROUTE_COARSE, ROUTE_ABSTAIN, ROUTE_ABSTAIN]
        )
        valid = torch.tensor([True, True, True, True, False])
        coverage = routing_coverage(route, valid)
        self.assertAlmostEqual(coverage["fine"].item(), 0.50)
        self.assertAlmostEqual(coverage["coarse"].item(), 0.25)
        self.assertAlmostEqual(coverage["accepted"].item(), 0.75)
        self.assertAlmostEqual(coverage["abstain"].item(), 0.25)

    def test_head_is_parameter_free_and_keeps_output_shapes(self):
        head = SelectiveMultiGranularityHead(fine_max_risk=0.25, coarse_max_risk=0.75)
        logits = torch.randn(2, 4, 4, 5)
        risk = torch.rand(2, 1, 4, 5)
        output = head(logits, risk)
        self.assertEqual(sum(parameter.numel() for parameter in head.parameters()), 0)
        self.assertEqual(output.route.shape, (2, 4, 5))
        self.assertEqual(output.hierarchy_conflict.shape, (2, 4, 5))
        self.assertEqual(output.fine_prediction.shape, (2, 4, 5))
        self.assertEqual(output.coarse_prediction.shape, (2, 4, 5))

    def test_invalid_risk_or_thresholds_are_rejected(self):
        cases = (
            (torch.tensor([-0.1]), 0.2, 0.8),
            (torch.tensor([1.1]), 0.2, 0.8),
            (torch.tensor([0.5]), 0.9, 0.8),
        )
        for risk, fine_threshold, coarse_threshold in cases:
            with self.subTest(risk=risk, fine=fine_threshold, coarse=coarse_threshold):
                with self.assertRaises(ValueError):
                    route_from_calibrated_risk(
                        risk,
                        fine_max_risk=fine_threshold,
                        coarse_max_risk=coarse_threshold,
                    )


if __name__ == "__main__":
    unittest.main()
