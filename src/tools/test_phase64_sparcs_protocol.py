#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from prepare_phase64_sparcs import audit_dataset, safe_extract  # noqa: E402


class Phase64SparcsProtocolTest(unittest.TestCase):
    def _make_release(self, root: Path, *, unsafe: bool = False) -> Path:
        staging = root / "staging" / "sending"
        staging.mkdir(parents=True)
        for index in range(80):
            scene = f"LC8{index:06d}2013{index % 365 + 1:03d}LGN00"
            key = f"{scene}_{index:02d}"
            photo = np.full((8, 8, 3), index, dtype=np.uint8)
            # Every split sees all three parents and all seven native IDs.
            mask = np.tile(np.arange(8, dtype=np.uint8) % 7, (8, 1))
            Image.fromarray(photo).save(staging / f"{key}_photo.png")
            Image.fromarray(mask).save(staging / f"{key}_mask.png")
        archive_path = root / "l8cloudmasks.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root / "staging").as_posix())
            if unsafe:
                archive.writestr("../escape.txt", "forbidden")
        return archive_path

    def test_safe_release_is_scene_disjoint_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self._make_release(root)
            extracted = root / "extracted"
            safe_extract(archive, extracted)
            rows, summary = audit_dataset(
                archive, extracted, expected_samples=80, expected_size=8
            )
            self.assertEqual(len(rows), 80)
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["split_samples"], {
                "target_train": 60,
                "target_val": 10,
                "target_test": 10,
            })
            self.assertTrue(summary["gates"]["scene_disjoint_split"])

    def test_zip_slip_member_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self._make_release(root, unsafe=True)
            with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
                safe_extract(archive, root / "extracted")


if __name__ == "__main__":
    unittest.main()
