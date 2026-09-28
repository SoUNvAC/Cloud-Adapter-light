from __future__ import annotations

import sys
import csv
import tempfile
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
    read_adjudication,
)
from audit_phase61d2_single_review import descriptive_subset
from canonicalize_phase61d2_review import canonicalize_rows


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

    def test_single_review_is_descriptive_only(self):
        records = [
            {"tile_id": "one", "original_label_name": "thin_cloud"},
            {"tile_id": "two", "original_label_name": "cloud_shadow"},
            {"tile_id": "three", "original_label_name": "thin_cloud"},
        ]
        review = {
            "one": parse_label_set("thin_cloud"),
            "two": parse_label_set("cloud_shadow|terrain_water_shadow"),
            "three": parse_label_set("boundary_mixed"),
        }
        summary = descriptive_subset(records, review)
        self.assertEqual(summary["units"], 3)
        self.assertEqual(summary["set_valued_units"], 1)
        self.assertEqual(summary["boundary_mixed_units"], 1)
        self.assertEqual(summary["explicit_ambiguity_or_unassessable_units"], 2)
        self.assertAlmostEqual(
            summary["original_vs_single_reviewer_membership_iou_exploratory"]["thin_cloud"]["iou"],
            0.5,
        )

    def test_blank_review_mapping_preserves_nonblank_labels(self):
        rows = [
            {"tile_id": "one", "label_set": "", "notes": ""},
            {"tile_id": "two", "label_set": "thin_cloud", "notes": "keep"},
        ]
        result, mapped = canonicalize_rows(rows, "unobservable_nodata")
        self.assertEqual(mapped, ["one"])
        self.assertEqual(result[0]["label_set"], "unobservable_nodata")
        self.assertEqual(result[1]["label_set"], "thin_cloud")
        self.assertEqual(result[1]["notes"], "keep")

    def test_missing_adjudication_rationale_requires_explicit_downgrade(self):
        review_a = {"one": parse_label_set("thin_cloud")}
        review_b = {"one": parse_label_set("cloud_shadow")}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adjudication.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=(
                    "tile_id", "reviewer_A_label_set", "reviewer_B_label_set",
                    "final_label_set", "rationale",
                ))
                writer.writeheader()
                writer.writerow({
                    "tile_id": "one", "reviewer_A_label_set": "thin_cloud",
                    "reviewer_B_label_set": "cloud_shadow", "final_label_set": "thin_cloud",
                    "rationale": "",
                })
            with self.assertRaises(RuntimeError):
                read_adjudication(path, ["one"], review_a, review_b)
            final, diagnostic = read_adjudication(
                path, ["one"], review_a, review_b, require_rationale=False
            )
            self.assertEqual(final["one"], parse_label_set("thin_cloud"))
            self.assertEqual(diagnostic["missing_rationale_units"], 1)
            self.assertEqual(diagnostic["rationale_complete_units"], 0)


if __name__ == "__main__":
    unittest.main()
