#!/usr/bin/env python3
"""Regression tests for the Phase 64 Day-1 dataset/ontology contract."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from audit_phase64_dataset_registry import validate_registry


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "research_plans" / "protocol_data" / "phase64_dataset_registry.json"


class Phase64DatasetRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    def test_frozen_registry_passes_and_reports_execution_blocker(self) -> None:
        summary = validate_registry(self.registry)
        self.assertEqual(summary["status"], "registry_valid")
        self.assertGreaterEqual(summary["public_target_domains_with_shadow"], 2)
        self.assertFalse(summary["day2_two_target_execution_ready"])
        self.assertIn("deep_fmask", summary["unimplemented_datasets"])

    def test_every_supervised_native_label_has_exactly_one_parent(self) -> None:
        for dataset in self.registry["datasets"]:
            for label in dataset["native_labels"]:
                self.assertIn(label["parent"], ("surface_visible", "cloud", "shadow"))

    def test_non_land_surface_labels_cannot_be_collapsed_into_cloud(self) -> None:
        broken = copy.deepcopy(self.registry)
        deep = next(row for row in broken["datasets"] if row["id"] == "deep_fmask")
        next(label for label in deep["native_labels"] if label["name"] == "snow")["parent"] = "cloud"
        with self.assertRaisesRegex(ValueError, "snow must remain"):
            validate_registry(broken)

    def test_l8_shadow_completeness_filter_is_mandatory(self) -> None:
        broken = copy.deepcopy(self.registry)
        l8 = next(row for row in broken["datasets"] if row["id"] == "l8_biome")
        l8["full_parent_supervision_filter"] = "all scenes"
        with self.assertRaisesRegex(ValueError, "Shadows\\?=yes"):
            validate_registry(broken)


if __name__ == "__main__":
    unittest.main()
