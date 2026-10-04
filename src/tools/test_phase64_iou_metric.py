#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import unittest


HAS_MMSEG = importlib.util.find_spec("mmseg") is not None


@unittest.skipUnless(HAS_MMSEG, "mmseg is only available in the training environment")
class Phase64IoUMetricTest(unittest.TestCase):
    def test_parent_iou_is_returned_by_name(self) -> None:
        import torch

        from cloud_adapter.dg_metrics import Phase64IoUMetric

        metric = Phase64IoUMetric(iou_metrics=["mIoU"])
        metric.dataset_meta = {
            "classes": ("surface_visible", "cloud", "shadow")
        }
        # intersection, union, prediction area, target area
        row = (
            torch.tensor([8.0, 3.0, 1.0]),
            torch.tensor([10.0, 6.0, 4.0]),
            torch.tensor([9.0, 4.0, 3.0]),
            torch.tensor([9.0, 5.0, 2.0]),
        )
        result = metric.compute_metrics([row])
        self.assertEqual(result["IoU/surface_visible"], 80.0)
        self.assertEqual(result["IoU/cloud"], 50.0)
        self.assertEqual(result["IoU/shadow"], 25.0)


if __name__ == "__main__":
    unittest.main()
