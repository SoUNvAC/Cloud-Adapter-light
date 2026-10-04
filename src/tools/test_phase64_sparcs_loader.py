#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


HAS_MMSEG = importlib.util.find_spec("mmseg") is not None


@unittest.skipUnless(HAS_MMSEG, "mmseg is only available in the training environment")
class Phase64SparcsLoaderTest(unittest.TestCase):
    def test_manifest_split_and_parent_map(self) -> None:
        from cloud_adapter.datasets.sparcs_manifest import (
            Phase64SparcsManifestDataset,
            SPARCS_NATIVE_TO_PARENT,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            fields = (
                "new_split", "scene", "name", "image_path", "mask_path",
                "multispectral_path", "tiff_candidates", "non_qa_tiff_candidates",
            )
            rows = [
                {
                    "new_split": split,
                    "scene": f"SCENE-{index}",
                    "name": f"sample-{index}",
                    "image_path": str(root / f"sample-{index}_photo.png"),
                    "mask_path": str(root / f"sample-{index}_mask.png"),
                    "multispectral_path": "",
                    "tiff_candidates": "0",
                    "non_qa_tiff_candidates": "0",
                }
                for index, split in enumerate(("target_train", "target_train", "target_val"))
            ]
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            dataset = Phase64SparcsManifestDataset(
                manifest_path=str(manifest),
                split="target_train",
                granularity="parent",
                pipeline=[],
            )
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.get_data_info(0)["label_map"], SPARCS_NATIVE_TO_PARENT)
            self.assertEqual(
                tuple(dataset.metainfo["classes"]),
                ("surface_visible", "cloud", "shadow"),
            )


if __name__ == "__main__":
    unittest.main()
