import importlib.util
from pathlib import Path
import unittest

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "cloud_adapter"
    / "models"
    / "segmentors"
    / "distill_losses.py"
)
SPEC = importlib.util.spec_from_file_location("distill_losses", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DistillLossTests(unittest.TestCase):
    def test_boundary_map_and_dilation(self):
        labels = torch.tensor([[[0, 0, 0], [0, 2, 0], [0, 0, 0]]])
        valid = torch.ones_like(labels, dtype=torch.bool)
        raw = MODULE.label_boundaries(labels, valid, radius=0)
        expected_raw = torch.tensor(
            [[[False, True, False], [True, True, True], [False, True, False]]]
        )
        self.assertTrue(torch.equal(raw, expected_raw))
        dilated = MODULE.label_boundaries(labels, valid, radius=1)
        self.assertTrue(dilated.all())

    def test_invalid_neighbors_do_not_create_boundaries(self):
        labels = torch.tensor([[[0, 255]]])
        valid = labels != 255
        boundary = MODULE.label_boundaries(labels, valid, radius=0)
        self.assertFalse(boundary.any())

    def test_weighted_kl_and_ignored_gradient(self):
        pixel_kl = torch.tensor([[[1.0, 3.0, 9.0]]], requires_grad=True)
        labels = torch.tensor([[[0, 2, 255]]])
        valid = labels != 255
        class_weights = torch.tensor([1.0, 1.0, 2.0, 2.0])
        loss = MODULE.weighted_pixel_kl(
            pixel_kl,
            labels,
            valid,
            class_weights,
            boundary_weight=2.0,
            boundary_radius=0,
        )
        self.assertAlmostEqual(loss.item(), 7.0 / 3.0, places=6)
        loss.backward()
        self.assertEqual(pixel_kl.grad[0, 0, 2].item(), 0.0)

    def test_all_invalid_returns_differentiable_zero(self):
        pixel_kl = torch.tensor([[[2.0]]], requires_grad=True)
        labels = torch.tensor([[[255]]])
        valid = torch.zeros_like(labels, dtype=torch.bool)
        loss = MODULE.weighted_pixel_kl(
            pixel_kl,
            labels,
            valid,
            torch.ones(4),
            boundary_weight=2.0,
            boundary_radius=1,
        )
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertEqual(pixel_kl.grad.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
