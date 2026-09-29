from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from evaluate_phase63_selective_risk import (  # noqa: E402
    BASE_FEATURES,
    binary_auc,
    feature_schema,
    fit_risk_model,
    predict_risk,
    scene_bootstrap,
    selection_metrics,
)
from export_phase63_unreliability import (  # noqa: E402
    CORE4,
    assemble_frozen_labels,
    label_outcomes,
    local_observability,
    normalize_semantic_scores,
    point_evidence,
    validate_scene_split,
    verify_fingerprint,
)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def synthetic_row(index: int, split: str, failure: int) -> dict[str, str]:
    high = 0.90 if failure else 0.10
    row = {
        "tile_id": f"tile-{index:03d}",
        "split": split,
        "scene": f"{'dev' if split == 'development' else 'confirm'}-{(index // 2) % 2}",
        "biome": "snow_ice" if split == "confirmation" and index % 3 == 0 else "forest",
        "final_label_set": "thin_cloud" if index % 2 else "cloud_shadow",
        "fine_failure": str(failure),
        "fine_evaluable_core4": "1",
        "coarse_evaluable": "1",
        "coarse_correct": str(1 - failure),
        "binary_evaluable": "1",
        "binary_correct": str(1 - failure),
        "thin_membership": str(index % 2),
        "shadow_membership": str(1 - index % 2),
        "hierarchy_feature_status": "unavailable_structurally_identical_without_independent_coarse_head",
        "hierarchy_conflict": "",
        "nir_mean": "",
        "swir1_mean": "",
        "swir2_mean": "",
    }
    for feature_index, key in enumerate(BASE_FEATURES):
        row[key] = str(high + feature_index * 1e-4)
    return row


class Phase63RiskProtocolTest(unittest.TestCase):
    def test_frozen_model_output_order_is_source_order(self):
        self.assertEqual(
            CORE4,
            ("clear", "thick_cloud", "thin_cloud", "cloud_shadow"),
        )

    def test_fingerprint_accepts_only_matching_full_or_abbreviated_sha(self):
        actual = "d33a81337544" + "0" * 48 + "8012"
        self.assertEqual(len(actual), 64)
        verify_fingerprint(actual, "d33a81337544...8012", "source")
        verify_fingerprint(actual, actual, "source")
        with self.assertRaises(RuntimeError):
            verify_fingerprint(actual, "d33a81337544...ffff", "source")

    def test_scene_split_is_exact_and_leakage_fails_closed(self):
        records = []
        split_rows = []
        for index in range(210):
            split = "development" if index < 160 else "confirmation"
            scene = f"dev-{index % 4}" if split == "development" else f"confirm-{index % 2}"
            records.append({"tile_id": f"t{index}", "scene": scene})
            split_rows.append({"tile_id": f"t{index}", "split": split})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "split.csv"
            write_csv(path, ["tile_id", "split"], split_rows)
            mapping = validate_scene_split(records, path)
            self.assertEqual(CounterLike(mapping.values()), {"development": 160, "confirmation": 50})
            records[-1]["scene"] = "dev-0"
            with self.assertRaisesRegex(RuntimeError, "Scene leakage"):
                validate_scene_split(records, path)

    def test_final_labels_preserve_79_blank_rationales(self):
        final_values = (
            ["clear"] * 39 + ["thin_cloud"] * 42 + ["thick_cloud"] * 10
            + ["cloud_shadow"] * 71 + ["haze_cirrus"] * 19
            + ["terrain_water_shadow"] * 12 + ["unobservable_nodata"] * 12
            + ["boundary_mixed"] * 4 + ["uncertain"]
        )
        records = [{"tile_id": f"t{index:03d}"} for index in range(210)]
        review_a, review_b, adjudication = [], [], []
        for index, record in enumerate(records):
            tile_id = record["tile_id"]
            target = final_values[index]
            a = "clear" if index < 79 else target
            b = "thin_cloud" if index < 79 else target
            review_a.append({"tile_id": tile_id, "label_set": a})
            review_b.append({"tile_id": tile_id, "label_set": b})
            if index < 79:
                adjudication.append({
                    "tile_id": tile_id,
                    "reviewer_A_label_set": a,
                    "reviewer_B_label_set": b,
                    "final_label_set": target,
                    "rationale": "",
                })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_csv(root / "a.csv", ["tile_id", "label_set"], review_a)
            write_csv(root / "b.csv", ["tile_id", "label_set"], review_b)
            write_csv(
                root / "c.csv",
                ["tile_id", "reviewer_A_label_set", "reviewer_B_label_set", "final_label_set", "rationale"],
                adjudication,
            )
            final, diagnostic = assemble_frozen_labels(records, root / "a.csv", root / "b.csv", root / "c.csv")
        self.assertEqual(len(final), 210)
        self.assertEqual(diagnostic["missing_rationales"], 79)
        self.assertFalse(diagnostic["rationales_imputed"])
        self.assertEqual(final["t000"], frozenset({"clear"}))

    def test_noncore_labels_are_fine_failures_not_clear(self):
        clear = label_outcomes(frozenset({"clear"}), "clear")
        self.assertEqual(clear["coarse_truth"], "clear")
        self.assertEqual(clear["binary_truth"], "non_cloud")
        haze = label_outcomes(frozenset({"haze_cirrus"}), "clear")
        self.assertEqual(haze["fine_failure"], 1)
        self.assertEqual(haze["fine_evaluable_core4"], 0)
        self.assertEqual(haze["coarse_evaluable"], 0)
        self.assertEqual(haze["binary_evaluable"], 0)
        mixed_cloud = label_outcomes(frozenset({"thin_cloud", "thick_cloud"}), "thick_cloud")
        self.assertEqual(mixed_cloud["fine_correct"], 1)
        self.assertEqual(mixed_cloud["coarse_truth"], "cloud")

    def test_hierarchy_identity_is_unavailable_without_independent_head(self):
        adapted = np.asarray([0.1, 0.2, 0.6, 0.1])[:, None, None]
        source = np.asarray([0.2, 0.2, 0.3, 0.3])[:, None, None]
        evidence = point_evidence([adapted, adapted], source, 0, 0, None)
        self.assertEqual(evidence["hierarchy_conflict"], "")
        self.assertIn("structurally_identical", evidence["hierarchy_feature_status"])
        independent = point_evidence([adapted, adapted], source, 0, 0, 0.25)
        self.assertAlmostEqual(independent["hierarchy_conflict"], 0.55)

    def test_probability_and_observability_features_are_finite(self):
        scores = np.ones((4, 3, 3), dtype=np.float64)
        probabilities = normalize_semantic_scores(scores)
        self.assertTrue(np.allclose(probabilities.sum(axis=0), 1.0))
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[4:12, 4:12] = 128
        spectral = np.ones((3, 4, 4), dtype=np.float32)
        observation = local_observability(rgb, 8, 8, 2, spectral)
        self.assertEqual(observation["spectral_available_channels"], 6)
        self.assertTrue(all(
            math.isfinite(float(value)) for value in observation.values()
        ))

    def test_shallow_risk_models_and_scene_bootstrap(self):
        development = [synthetic_row(index, "development", index % 2) for index in range(160)]
        confirmation = [synthetic_row(160 + index, "confirmation", index % 2) for index in range(50)]
        all_rows = development + confirmation
        schema = feature_schema(development, all_rows)
        for kind in ("logistic", "gam", "isotonic_entropy"):
            fitted = fit_risk_model(kind, development, schema)
            self.assertEqual(fitted["fit_units"], 160)
            risk = predict_risk(fitted, confirmation, schema)
            labels = np.asarray([int(row["fine_failure"]) for row in confirmation])
            self.assertGreater(binary_auc(labels, risk), 0.99)
            selected = selection_metrics(confirmation, risk, 0.60)
            self.assertGreater(selected["fine_error_relative_reduction"], 0.25)
        bootstrap = scene_bootstrap(confirmation, risk, draws=200, seed=63)
        self.assertEqual(bootstrap["draws_requested"], 200)
        self.assertGreaterEqual(bootstrap["auroc"]["valid_draws"], 190)
        with self.assertRaisesRegex(RuntimeError, "forbidden"):
            fit_risk_model("deep_mlp", development, schema)


def CounterLike(values):
    result = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


if __name__ == "__main__":
    unittest.main()
