from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from freeze_phase63_scene_split import (
    InputValidationError,
    SceneLeakageError,
    audit_and_build_lock,
    canonical_sha256,
    sha256_file,
    write_lock_atomic,
)


class Phase63SceneSplitTest(unittest.TestCase):
    def _write_fixture(self, root: Path, *, leak: bool = False, missing_rationales: int = 79):
        records = []
        for index in range(40):
            records.append(
                {
                    "tile_id": f"CAL-{index:03d}",
                    "scene": f"cal-scene-{index % 2}",
                    "biome": "snow_ice" if index == 0 else "forest",
                    "cohort": "calibration",
                }
            )
        for index in range(160):
            records.append(
                {
                    "tile_id": f"DEV-{index:03d}",
                    "scene": "shared-scene" if leak and index == 0 else f"dev-scene-{index % 4}",
                    "biome": "snow_ice" if index < 8 else "forest",
                    "cohort": "independent",
                }
            )
        for index in range(50):
            records.append(
                {
                    "tile_id": f"CONF-{index:03d}",
                    "scene": "shared-scene" if leak and index == 0 else f"conf-scene-{index % 3}",
                    "biome": "snow_ice" if index < 5 else "wetlands",
                    "cohort": "confirmation",
                }
            )

        sealed = {
            "phase": "61D2",
            "status": "awaiting_calibration_and_human_review",
            "calibration_units_excluded_from_statistics": 40,
            "main_units": 210,
            "independent_units": 160,
            "confirmation_units": 50,
            "manual_sha256": "manual-hash",
            "records": records,
        }
        calibration = {
            "phase": "61D2-calibration-lock",
            "status": "locked_before_independent_review",
            "calibration_units": 40,
            "calibration_excluded_from_final_statistics": True,
            "manual_sha256": "manual-hash",
        }
        sealed_path = root / "sealed_manifest.json"
        calibration_path = root / "calibration_lock.json"
        metrics_path = root / "review_metrics.json"
        sealed_path.write_text(json.dumps(sealed), encoding="utf-8")
        calibration_path.write_text(json.dumps(calibration), encoding="utf-8")

        sealed_sha = sha256_file(sealed_path)
        calibration_sha = sha256_file(calibration_path)
        missing_ids = [f"DEV-{index:03d}" for index in range(missing_rationales)]
        metrics = {
            "phase": "61D2",
            "status": "complete_adjudicated",
            "main_units": 210,
            "calibration_units_excluded_from_all_reported_statistics": 40,
            "sealed_manifest": {"sha256": sealed_sha},
            "calibration": {"sha256": calibration_sha},
            "adjudication": {
                "units": 79,
                "rationale_required": False,
                "rationale_complete_units": 0,
                "missing_rationale_units": missing_rationales,
                "missing_rationale_tile_ids": missing_ids,
            },
            "disagreements": {"units": 79},
            "final_label_provenance": {
                "exact_A_B_agreement_units": 131,
                "third_expert_disagreement_only_units": 79,
                "raw_A_B_reviews_preserved": True,
            },
        }
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        return sealed_path, calibration_path, metrics_path, sealed_sha

    def test_valid_scene_disjoint_split_is_frozen_deterministically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sealed, calibration, metrics, sealed_sha = self._write_fixture(root)
            lock = audit_and_build_lock(
                sealed,
                calibration,
                metrics,
                expected_sealed_sha256=sealed_sha,
            )
            self.assertEqual(lock["phase"], "63A-scene-freeze")
            self.assertEqual(lock["status"], "locked_scene_disjoint")
            self.assertEqual(len(lock["records"]), 210)
            self.assertEqual(lock["development_units"], 160)
            self.assertEqual(lock["confirmation_units"], 50)
            self.assertEqual(lock["counts"]["snow_ice_units"], 13)
            self.assertEqual(lock["review_integrity"]["missing_rationale_units"], 79)
            self.assertTrue(
                lock["review_integrity"]["missing_rationales_preserved_without_imputation"]
            )
            confirmation = [
                row for row in lock["records"] if row["cohort"] == "confirmation"
            ]
            self.assertTrue(
                all(row["phase63_role"] == "locked_final_validation_only" for row in confirmation)
            )
            self.assertEqual(
                lock["hashes"]["frozen_records_sha256"],
                canonical_sha256(lock["records"]),
            )

            output = root / "phase63_split_lock.json"
            first_sha = write_lock_atomic(output, lock)
            second_sha = write_lock_atomic(output, lock)
            self.assertEqual(first_sha, second_sha)
            self.assertEqual(first_sha, sha256_file(output))

    def test_scene_leakage_fails_closed_with_exact_tiles_and_no_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sealed, calibration, metrics, sealed_sha = self._write_fixture(root, leak=True)
            output = root / "phase63_split_lock.json"
            with self.assertRaises(SceneLeakageError) as context:
                audit_and_build_lock(
                    sealed,
                    calibration,
                    metrics,
                    expected_sealed_sha256=sealed_sha,
                )
            report = context.exception.report
            self.assertFalse(report["passed"])
            self.assertFalse(report["lock_written"])
            self.assertEqual(report["conflict_scene_count"], 1)
            self.assertEqual(report["conflict_scenes"][0]["scene"], "shared-scene")
            self.assertEqual(
                report["conflict_scenes"][0]["development_tile_ids"], ["DEV-000"]
            )
            self.assertEqual(
                report["conflict_scenes"][0]["confirmation_tile_ids"], ["CONF-000"]
            )
            self.assertFalse(output.exists())

    def test_wrong_sealed_hash_is_rejected_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sealed, calibration, metrics, _sealed_sha = self._write_fixture(root)
            with self.assertRaisesRegex(InputValidationError, "SHA256 mismatch"):
                audit_and_build_lock(
                    sealed,
                    calibration,
                    metrics,
                    expected_sealed_sha256="0" * 64,
                )

    def test_missing_rationales_must_remain_exactly_79(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sealed, calibration, metrics, sealed_sha = self._write_fixture(
                root, missing_rationales=78
            )
            with self.assertRaisesRegex(
                InputValidationError, "missing_rationale_units must equal 79"
            ):
                audit_and_build_lock(
                    sealed,
                    calibration,
                    metrics,
                    expected_sealed_sha256=sealed_sha,
                )


if __name__ == "__main__":
    unittest.main()
