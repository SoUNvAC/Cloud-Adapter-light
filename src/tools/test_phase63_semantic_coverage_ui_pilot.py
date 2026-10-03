from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from prepare_phase63_semantic_coverage_ui_pilot import (  # noqa: E402
    QUOTAS,
    canonical_csv_sha256,
    extract_patch_candidates,
    solve_assignment,
)
from score_phase63_semantic_coverage_ui_pilot import score  # noqa: E402


class Phase63SemanticCoverageUIPilotTest(unittest.TestCase):
    @staticmethod
    def striped_mask():
        mask = np.zeros((512, 512), dtype=np.int16)
        mask[:, 128:256] = 3  # clear | shadow
        mask[:, 256:384] = 1  # shadow | thick
        mask[:, 384:] = 2     # thick | thin
        return mask

    def test_candidate_distances_and_fixed_boundary_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "patch.npz"
            np.savez_compressed(cache, source_label=self.striped_mask())
            row = {
                "new_split": "target_train", "scene": "SCENE", "biome": "forest",
                "name": "patch.png", "x": "0", "y": "0",
            }
            candidates = extract_patch_candidates(row, self.striped_mask(), cache)
        by = {item["sampling_stratum"]: item for item in candidates}
        self.assertEqual(set(by), set(QUOTAS))
        for stratum, item in by.items():
            if stratum.startswith("interior_"):
                self.assertGreater(item["distance_to_any_label_boundary_px"], 3.0)
            else:
                self.assertLessEqual(item["pair_boundary_distance_px"], 1.0)

    def test_metadata_hash_is_independent_of_platform_line_endings(self):
        with tempfile.TemporaryDirectory() as directory:
            lf = Path(directory) / "lf.csv"
            crlf = Path(directory) / "crlf.csv"
            content = "scene,usgs_shadows,source_url\nS1,yes,url-1\nS2,no,url-2\n"
            lf.write_bytes(content.encode("utf-8"))
            crlf.write_bytes(content.replace("\n", "\r\n").encode("utf-8"))
            self.assertEqual(canonical_csv_sha256(lf), canonical_csv_sha256(crlf))

    def test_assignment_prefers_twenty_distinct_scenes_and_one_point_per_patch(self):
        candidates = []
        with tempfile.TemporaryDirectory() as directory:
            for index in range(20):
                cache = Path(directory) / f"patch-{index}.npz"
                np.savez_compressed(cache, source_label=self.striped_mask())
                row = {
                    "new_split": "target_train", "scene": f"SCENE-{index:02d}",
                    "biome": "forest", "name": f"patch-{index}.png", "x": "0", "y": "0",
                }
                candidates.extend(extract_patch_candidates(row, self.striped_mask(), cache))
        selected, audit = solve_assignment(candidates)
        self.assertEqual(len(selected), 20)
        self.assertEqual(audit["distinct_scenes"], 20)
        self.assertEqual(audit["max_points_per_scene"], 1)
        self.assertEqual(audit["max_points_per_patch"], 1)
        self.assertEqual(audit["stratum_counts"], dict(sorted(QUOTAS.items())))
        self.assertGreaterEqual(min(audit["interior_class_scene_counts"].values()), 3)

    def test_assignment_falls_back_to_two_points_per_scene_without_impossible_search(self):
        candidates = []
        for stratum in QUOTAS:
            for scene_index in range(10):
                candidates.append(
                    {
                        "sampling_stratum": stratum,
                        "scene": f"SCENE-{scene_index:02d}",
                        "name": f"{stratum}-patch-{scene_index:02d}.png",
                        "center_y": 100,
                        "center_x": 100,
                    }
                )
        selected, audit = solve_assignment(candidates)
        self.assertEqual(len(selected), 20)
        self.assertEqual(audit["available_candidate_scenes"], 10)
        self.assertEqual(audit["distinct_scenes"], 10)
        self.assertEqual(audit["max_points_per_scene"], 2)

    def test_missing_boundary_pair_fails_closed(self):
        candidates = []
        for stratum, quota in QUOTAS.items():
            if stratum == "boundary_clear__cloud_shadow":
                continue
            for index in range(quota):
                candidates.append(
                    {
                        "sampling_stratum": stratum, "scene": f"S-{stratum}-{index}",
                        "name": f"P-{stratum}-{index}", "center_y": 100, "center_x": 100,
                    }
                )
        with self.assertRaisesRegex(RuntimeError, "Insufficient"):
            solve_assignment(candidates)

    def test_scoring_uses_interior_only_for_semantic_gate(self):
        records, review_a, review_b = [], {}, {}
        for index in range(20):
            interior = index < 16
            tile_id = f"item-{index:02d}"
            records.append(
                {
                    "tile_id": tile_id,
                    "sampling_kind": "interior" if interior else "boundary",
                    "sampling_stratum": "interior_clear" if interior else "boundary_clear__cloud_shadow",
                    "nominal_label_name": "clear",
                }
            )
            value = {
                "locatable": True,
                "identifiability": "fine_classifiable" if interior else "mixed_boundary",
                "semantic": "clear" if interior else "",
                "elapsed_seconds": 10.0,
            }
            review_a[tile_id] = dict(value)
            review_b[tile_id] = dict(value)
        sealed = {
            "phase": "63-semantic-coverage-exact-pixel-ui-pilot",
            "records": records,
            "artifact_validation": {
                "coordinates_valid": True, "pages_valid": True,
                "review_templates_blank": True,
                "reviewer_packet_excludes_nominal_fields": True,
            },
        }
        result = score(sealed, review_a, review_b)
        self.assertEqual(result["semantic_when_both_fine_interior"]["units"], 16)
        self.assertTrue(result["ui_pilot_pass"])
        self.assertEqual(
            result["reviewer_nominal_label_concordance"]["reviewer_A"]["name"],
            "reviewer–nominal-label concordance",
        )


if __name__ == "__main__":
    unittest.main()
