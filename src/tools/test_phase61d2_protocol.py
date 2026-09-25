from __future__ import annotations

import sys
import unittest
from pathlib import Path

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from prepare_phase61d2_calibrated_review import split_legacy_records
from score_phase61d2_calibrated_review import (
    canonical,
    cohen_kappa,
    gate_decision,
    gwet_ac1,
    membership_iou,
    parse_label_set,
)


class Phase61D2ProtocolTest(unittest.TestCase):
    def test_set_label_parser(self):
        self.assertEqual(canonical(parse_label_set("thin_cloud|haze_cirrus")), "haze_cirrus|thin_cloud")
        self.assertEqual(parse_label_set("uncertain"), frozenset({"uncertain"}))
        for invalid in ("", "thin_cloud|boundary_mixed", "clear|thin_cloud|thick_cloud", "made_up"):
            with self.assertRaises(ValueError):
                parse_label_set(invalid)

    def test_agreement_metrics(self):
        left = ["clear", "thin_cloud", "cloud_shadow", "thin_cloud|thick_cloud"]
        self.assertAlmostEqual(cohen_kappa(left, left), 1.0)
        self.assertAlmostEqual(gwet_ac1(left, left), 1.0)
        values = [parse_label_set(item) for item in left]
        iou = membership_iou(values, values, ("thin_cloud", "cloud_shadow"))
        self.assertAlmostEqual(iou["thin_cloud"]["iou"], 1.0)
        self.assertAlmostEqual(iou["cloud_shadow"]["iou"], 1.0)

    def test_calibration_is_exactly_excluded(self):
        labels = ("definite thin", "definite cloud shadow")
        records = []
        for index in range(200):
            records.append({
                "name": f"image-{index:03d}.png",
                "center_y": index % 512,
                "center_x": (index * 7) % 512,
                "source_stratum": f"source-{index % 2}",
                "biome": f"biome-{index % 5}",
                "confidence_stratum": f"confidence-{index % 2}",
                "region": f"region-{index % 2}",
                "original_label_name": labels[index % 2],
            })
        calibration, independent = split_legacy_records(records)
        self.assertEqual(len(calibration), 40)
        self.assertEqual(len(independent), 160)
        calibration_keys = {(r["name"], r["center_y"], r["center_x"]) for r in calibration}
        independent_keys = {(r["name"], r["center_y"], r["center_x"]) for r in independent}
        self.assertFalse(calibration_keys & independent_keys)
        self.assertEqual({r["original_label_name"] for r in calibration + independent}, {"thin_cloud", "cloud_shadow"})

    def test_hard_gates_are_conservative(self):
        agreement = {
            "cohen_kappa_exact_set_categories": 0.55,
            "gwet_ac1_exact_set_categories": 0.65,
            "reviewer_membership_iou": {
                "thin_cloud": {"iou": 0.70},
                "cloud_shadow": {"iou": 0.72},
            },
        }
        original = {
            "thin_cloud": {"iou": 0.35},
            "cloud_shadow": {"iou": 0.50},
        }
        result = gate_decision(agreement, original)
        self.assertTrue(result["supports_annotation_process_shift"])
        self.assertEqual(result["conclusion"], "supports_annotation_process_shift")

        agreement["reviewer_membership_iou"]["thin_cloud"]["iou"] = 0.29
        result = gate_decision(agreement, original)
        self.assertTrue(result["terminate_four_class_hard_label_route"])
        self.assertEqual(result["conclusion"], "terminate_four_class_hard_label_route")


if __name__ == "__main__":
    unittest.main()
