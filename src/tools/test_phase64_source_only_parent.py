#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
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

    def test_palette_mask_keeps_class_indices(self) -> None:
        labels = np.asarray([[0, 1], [2, 3]], dtype=np.uint8)
        palette = [79, 253, 199, 255, 255, 255, 170, 170, 170, 85, 85, 85]
        palette.extend([0] * (768 - len(palette)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            image = Image.fromarray(labels, mode="P")
            image.putpalette(palette)
            image.save(path)
            target = self.module.load_parent_target(
                path, self.module.SOURCE_TO_PARENT, labels.shape
            )
        expected = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
        np.testing.assert_array_equal(target, expected)


if __name__ == "__main__":
    unittest.main()
