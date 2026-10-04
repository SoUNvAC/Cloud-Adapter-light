#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


TOOLS = Path(__file__).resolve().parent
MODULE_PATH = TOOLS / "evaluate_phase64_source_only_parent.py"
SPEC = importlib.util.spec_from_file_location("evaluate_phase64_source_only_parent", MODULE_PATH)


@unittest.skipUnless(
    importlib.util.find_spec("mmseg") is not None and importlib.util.find_spec("cv2") is not None,
    "evaluation dependencies are only available in the training environment",
)
class Phase64SourceOnlyParentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert SPEC is not None and SPEC.loader is not None
        cls.module = importlib.util.module_from_spec(SPEC)
        sys.modules[SPEC.name] = cls.module
        SPEC.loader.exec_module(cls.module)

    def test_metrics_and_bootstrap_are_finite(self) -> None:
        confusion = np.asarray([[8, 1, 0], [1, 5, 1], [0, 1, 4]], dtype=np.int64)
        metrics = self.module.metrics_from_confusion(confusion)
        self.assertTrue(np.isfinite(metrics["mIoU"]))
        bootstrap = self.module.bootstrap_scene_confusions(
            {"a": confusion, "b": confusion * 2}, draws=100, seed=64
        )
        self.assertEqual(bootstrap["scene_count"], 2)
        self.assertLessEqual(
            bootstrap["mIoU"]["ci95_low"], bootstrap["mIoU"]["ci95_high"]
        )


if __name__ == "__main__":
    unittest.main()
