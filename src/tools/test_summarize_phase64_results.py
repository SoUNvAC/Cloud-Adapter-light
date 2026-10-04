#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parent
MODULE_PATH = TOOLS / "summarize_phase64_results.py"
SPEC = importlib.util.spec_from_file_location("summarize_phase64_results", MODULE_PATH)


class Phase64SummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert SPEC is not None and SPEC.loader is not None
        cls.module = importlib.util.module_from_spec(SPEC)
        sys.modules[SPEC.name] = cls.module
        SPEC.loader.exec_module(cls.module)

    def test_training_parser_selects_best_validation(self) -> None:
        text = "\n".join(
            (
                "Iter(train) [ 500/4000] loss: 2.0 decode.loss_cls: 1.0",
                "Iter(val) [10/10] mIoU: 55.0 IoU/surface_visible: 70.0 "
                "IoU/cloud: 60.0 IoU/shadow: 35.0",
                "Iter(train) [1000/4000] loss: 1.5 decode.loss_cls: 0.8",
                "Iter(val) [10/10] mIoU: 57.0 IoU/surface_visible: 72.0 "
                "IoU/cloud: 62.0 IoU/shadow: 37.0",
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text(text, encoding="utf-8")
            result = self.module.parse_training_log(path)
        self.assertEqual(result["best"]["iteration"], 1000)
        self.assertEqual(result["best"]["per_class_iou"]["shadow"], 37.0)
        self.assertTrue(result["losses_finite"])

    def test_evaluation_parser_requires_one_complete_record(self) -> None:
        text = (
            "Iter(test) [535/535] mIoU: 76.5 "
            "IoU/surface_visible: 88.0 IoU/cloud: 83.0 IoU/shadow: 58.5\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.log"
            path.write_text(text, encoding="utf-8")
            result = self.module.parse_evaluation_log(path)
        self.assertEqual(result["mIoU"], 76.5)
        self.assertEqual(result["per_class_iou"]["cloud"], 83.0)


if __name__ == "__main__":
    unittest.main()
