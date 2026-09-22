import json
from pathlib import Path
import tempfile
import unittest

from cloud_adapter.datasets.weak_prevalence_sampler import (
    WeakPrevalenceInfiniteSampler,
)


class FakeDataset:
    def __init__(self, size=8):
        self.size = size

    def __len__(self):
        return self.size

    def get_data_info(self, index):
        return {"seg_map_path": f"/data/{index}.png"}


class WeakPrevalenceSamplerTest(unittest.TestCase):
    def manifest(self, directory, ordered=None):
        path = Path(directory) / "manifest.json"
        path.write_text(
            json.dumps(
                {
                    "sample_count": 8,
                    "ordered_paths": ordered or [f"{index}.png" for index in range(8)],
                    "rich_indices": [6, 7],
                }
            ),
            encoding="utf-8",
        )
        return str(path)

    def test_deterministic_mixture(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.manifest(directory)
            first = WeakPrevalenceInfiniteSampler(
                FakeDataset(), manifest, rich_probability=0.5, seed=42
            )
            second = WeakPrevalenceInfiniteSampler(
                FakeDataset(), manifest, rich_probability=0.5, seed=42
            )
            values = [next(iter(first)) for _ in range(100)]
            expected = [next(iter(second)) for _ in range(100)]
            self.assertEqual(values, expected)
            self.assertTrue(all(0 <= value < 8 for value in values))
            self.assertGreater(sum(value in {6, 7} for value in values), 50)

    def test_rejects_wrong_order(self):
        with tempfile.TemporaryDirectory() as directory:
            ordered = [f"{index}.png" for index in range(8)]
            ordered[0], ordered[1] = ordered[1], ordered[0]
            with self.assertRaises(ValueError):
                WeakPrevalenceInfiniteSampler(
                    FakeDataset(), self.manifest(directory, ordered), seed=42
                )


if __name__ == "__main__":
    unittest.main()
