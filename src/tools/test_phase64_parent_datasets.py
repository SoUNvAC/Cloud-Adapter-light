#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import unittest


HAS_MMSEG = importlib.util.find_spec("mmseg") is not None


@unittest.skipUnless(HAS_MMSEG, "mmseg is only available in the training environment")
class Phase64ParentDatasetsTest(unittest.TestCase):
    def test_frozen_maps(self) -> None:
        from cloud_adapter.datasets.phase64_parent import (
            CLOUDSEN_TO_PARENT,
            L8_TO_PARENT,
            PARENT_CLASSES,
        )

        self.assertEqual(PARENT_CLASSES, ("surface_visible", "cloud", "shadow"))
        self.assertEqual(CLOUDSEN_TO_PARENT, {0: 0, 1: 1, 2: 1, 3: 2})
        self.assertEqual(L8_TO_PARENT, {0: 0, 1: 2, 2: 1, 3: 1})


if __name__ == "__main__":
    unittest.main()
