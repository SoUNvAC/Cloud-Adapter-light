from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from prepare_phase63_human_confirmation import (  # noqa: E402
    deterministic_center,
    select_confirmation_records,
)


class Phase63HumanConfirmationTest(unittest.TestCase):
    def fixture(self):
        rows = []
        status = {}
        for scene_index in range(8):
            scene = f"VAL-NO-{scene_index}"
            status[scene] = "no"
            for patch in range(10):
                rows.append(
                    {
                        "new_split": "target_val",
                        "biome": "water" if scene_index % 2 else "forest",
                        "scene": scene,
                        "name": f"{scene}-patch-{patch}.png",
                        "x": str(patch * 512),
                        "y": "0",
                    }
                )
        for scene_index in range(3):
            scene = f"SNOW-{scene_index}"
            status[scene] = "no"
            for patch in range(10):
                rows.append(
                    {
                        "new_split": "target_train",
                        "biome": "snow_ice",
                        "scene": scene,
                        "name": f"{scene}-patch-{patch}.png",
                        "x": str(patch * 512),
                        "y": "512",
                    }
                )
        for patch in range(5):
            rows.append(
                {
                    "new_split": "target_test",
                    "biome": "snow_ice",
                    "scene": "SEALED-TEST",
                    "name": f"sealed-{patch}.png",
                    "x": "0",
                    "y": "0",
                }
            )
        return rows, status

    def test_selection_is_scene_disjoint_label_blind_and_snow_preserving(self):
        rows, status = self.fixture()
        selected, audit = select_confirmation_records(
            rows,
            status,
            selected_training_scenes={"TRAINED-SCENE"},
            development_scenes={"DEV-SCENE"},
        )
        self.assertEqual(len(selected), 50)
        self.assertEqual(audit["scenes"], 11)
        self.assertEqual(audit["split_counts"], {"target_train": 18, "target_val": 32})
        self.assertEqual(audit["biome_counts"]["snow_ice"], 18)
        self.assertEqual(audit["target_test_units"], 0)
        self.assertFalse(audit["selection_uses_pixel_labels"])
        self.assertFalse(audit["selection_uses_model_predictions"])
        self.assertTrue(audit["selection_uses_protocol_metadata_strata"])
        self.assertTrue(all(row["scene"] != "SEALED-TEST" for row in selected))
        self.assertEqual(len({row["name"] for row in selected}), 50)

    def test_centers_are_deterministic_and_inside_fixed_margin(self):
        first = deterministic_center("example.png")
        second = deterministic_center("example.png")
        self.assertEqual(first, second)
        self.assertTrue(all(96 <= value < 416 for value in first))

    def test_training_scene_overlap_fails_closed(self):
        rows, status = self.fixture()
        with self.assertRaisesRegex(RuntimeError, "Expected 3"):
            select_confirmation_records(
                rows,
                status,
                selected_training_scenes={"SNOW-0"},
                development_scenes=set(),
            )


if __name__ == "__main__":
    unittest.main()
