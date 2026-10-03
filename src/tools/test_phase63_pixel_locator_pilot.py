from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from prepare_phase63_pixel_locator_pilot import (  # noqa: E402
    BIOMES,
    LOCAL_WINDOW,
    deterministic_center,
    mark_exact_pixel,
    select_pilot_records,
)
from score_phase63_pixel_locator_pilot import score  # noqa: E402


class Phase63PixelLocatorPilotTest(unittest.TestCase):
    def fixture(self):
        rows, selected = [], []
        for biome_index, biome in enumerate(BIOMES):
            scene = f"TRAIN-{biome}"
            selected.append({"scene": scene, "name": f"{scene}-selected.png"})
            for patch in range(5):
                rows.append(
                    {
                        "new_split": "target_train",
                        "biome": biome,
                        "scene": scene,
                        "name": f"{scene}-{patch}.png",
                        "x": str(512 * patch),
                        "y": str(512 * biome_index),
                    }
                )
        rows.append(
            {
                "new_split": "target_test", "biome": "water", "scene": "SEALED",
                "name": "sealed.png", "x": "0", "y": "0",
            }
        )
        return rows, selected

    def test_selection_is_balanced_new_and_target_test_free(self):
        rows, selected = self.fixture()
        records, audit = select_pilot_records(rows, selected, {"OLD-REVIEW-SCENE"})
        self.assertEqual(len(records), 16)
        self.assertEqual(audit["scenes"], 8)
        self.assertEqual(audit["biome_counts"], {biome: 2 for biome in sorted(BIOMES)})
        self.assertEqual(audit["target_test_units"], 0)
        self.assertTrue(all(row["new_split"] == "target_train" for row in records))
        self.assertTrue(all(96 <= row["center_x"] < 416 for row in records))

    def test_exact_pixel_marker_keeps_center_interior_visible(self):
        scale = 11
        source = Image.new("RGB", (LOCAL_WINDOW * scale, LOCAL_WINDOW * scale), "white")
        marked = mark_exact_pixel(source, scale)
        center = LOCAL_WINDOW * scale // 2
        self.assertEqual(marked.getpixel((center, center)), (255, 255, 255))
        self.assertNotEqual(marked.getpixel((center - scale // 2 - 3, center)), (255, 255, 255))

    def test_center_is_deterministic(self):
        self.assertEqual(deterministic_center("a.png"), deterministic_center("a.png"))

    def test_scoring_separates_identifiability_from_semantics(self):
        ids = [f"item-{index}" for index in range(8)]
        a, b = {}, {}
        for index, item in enumerate(ids):
            ident = "fine_classifiable" if index < 6 else "mixed_boundary"
            semantic = "thin_cloud" if index < 6 else ""
            a[item] = {
                "locatable": True, "identifiability": ident,
                "semantic": semantic, "elapsed_seconds": 10.0,
            }
            b[item] = dict(a[item])
        result = score(a, b, ids)
        self.assertEqual(result["semantic_when_both_fine"]["units"], 6)
        self.assertEqual(result["identifiability"]["exact_agreement"], 1.0)
        self.assertTrue(result["ui_pilot_pass"])


if __name__ == "__main__":
    unittest.main()
