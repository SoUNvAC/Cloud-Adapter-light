import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "cloud_adapter"
    / "models"
    / "backbones"
    / "structured_pruning.py"
)
SPEC = importlib.util.spec_from_file_location("structured_pruning", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
resolve_active_block_indices = MODULE.resolve_active_block_indices


class StructuredPruningTests(unittest.TestCase):
    def test_none_keeps_every_block(self):
        self.assertEqual(
            resolve_active_block_indices(12, None, [2, 5, 8, 11], [2, 5, 8, 11]),
            tuple(range(12)),
        )

    def test_valid_static_subset(self):
        self.assertEqual(
            resolve_active_block_indices(
                12,
                [0, 2, 4, 5, 7, 8, 10, 11],
                [2, 5, 8, 11],
                [2, 5, 8, 11],
            ),
            (0, 2, 4, 5, 7, 8, 10, 11),
        )

    def test_missing_feature_tap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing indices: \[8\]"):
            resolve_active_block_indices(
                12,
                [0, 2, 4, 5, 7, 10, 11],
                [2, 5, 8, 11],
                [2, 5, 8, 11],
            )

    def test_duplicates_and_out_of_range_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            resolve_active_block_indices(4, [0, 1, 1, 3], [1, 3], [1, 3])
        with self.assertRaisesRegex(ValueError, "outside"):
            resolve_active_block_indices(4, [0, 1, 3, 4], [1, 3], [1, 3])


if __name__ == "__main__":
    unittest.main()
