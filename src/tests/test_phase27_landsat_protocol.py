import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from phase27_landsat_protocol import (  # noqa: E402
    binary_metrics,
    decode_cfmask_cloud,
    landsat_dn_to_rgb_scale,
    map_cloud_adapter_binary,
    map_cloud_adapter_three_class,
    map_truth_binary,
    map_truth_three_class,
    paired_scene_bootstrap,
    scene_cloud_iou,
    update_binary_confusion,
)
from audit_phase27_landsat_c2 import parse_mtl  # noqa: E402


class Phase27LandsatProtocolTests(unittest.TestCase):
    def test_frozen_class_harmonization(self):
        prediction = np.array([[0, 1, 2, 3]], dtype=np.uint8)
        self.assertEqual(
            map_cloud_adapter_binary(prediction).tolist(), [[0, 1, 1, 0]]
        )
        self.assertEqual(
            map_cloud_adapter_three_class(prediction).tolist(), [[0, 2, 1, 0]]
        )
        truth = np.array([[0, 128, 192, 255]], dtype=np.uint8)
        binary, valid = map_truth_binary(truth)
        self.assertEqual(binary.tolist(), [[0, 0, 1, 1]])
        self.assertEqual(valid.tolist(), [[False, True, True, True]])
        self.assertEqual(map_truth_three_class(truth).tolist(), [[0, 0, 1, 2]])

    def test_cfmask_bits_are_exactly_1_2_3(self):
        qa = np.array([[0, 1, 2, 4, 8, 16, 32]], dtype=np.uint16)
        self.assertEqual(
            decode_cfmask_cloud(qa).tolist(), [[0, 0, 1, 1, 1, 0, 0]]
        )

    def test_binary_metrics_and_empty_scene_policy(self):
        confusion = np.zeros((2, 2), dtype=np.int64)
        prediction = np.array([[0, 1, 1, 0]], dtype=np.uint8)
        target = np.array([[0, 0, 1, 1]], dtype=np.uint8)
        update_binary_confusion(
            confusion, prediction, target, np.ones_like(target, dtype=bool)
        )
        metrics = binary_metrics(confusion)
        self.assertAlmostEqual(metrics["cloud_iou"], 100.0 / 3.0)
        self.assertAlmostEqual(metrics["cloud_f1"], 50.0)
        self.assertEqual(scene_cloud_iou(np.zeros((2, 2), dtype=np.int64)), 100.0)

    def test_toa_conversion_uses_sun_angle_and_preserves_fill(self):
        digital_number = np.array([0, 10000, 50000], dtype=np.float32)
        converted = landsat_dn_to_rgb_scale(
            digital_number, multiplier=0.00002, additive=-0.1, sun_elevation=30.0
        )
        self.assertEqual(converted[0], 0.0)
        self.assertAlmostEqual(converted[1], 51.0, places=4)
        self.assertEqual(converted[2], 255.0)

    def test_paired_bootstrap_is_deterministic_and_paired(self):
        model = [50.0, 60.0, 70.0, 80.0]
        baseline = [55.0, 65.0, 75.0, 85.0]
        first = paired_scene_bootstrap(model, baseline, replicates=500, seed=7)
        second = paired_scene_bootstrap(model, baseline, replicates=500, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["paired_difference_mean"], -5.0)
        self.assertEqual(first["paired_difference_95ci"], [-5.0, -5.0])

    def test_mtl_parser_requires_all_reflectance_fields(self):
        content = "\n".join(
            [
                "REFLECTANCE_MULT_BAND_2 = 0.00002",
                "REFLECTANCE_ADD_BAND_2 = -0.1",
                "REFLECTANCE_MULT_BAND_3 = 0.00002",
                "REFLECTANCE_ADD_BAND_3 = -0.1",
                "REFLECTANCE_MULT_BAND_4 = 0.00002",
                "REFLECTANCE_ADD_BAND_4 = -0.1",
                "SUN_ELEVATION = 45.0",
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "MTL.txt"
            path.write_text(content, encoding="utf-8")
            result = parse_mtl(path)
        self.assertEqual(result["SUN_ELEVATION"], 45.0)

    @unittest.skipUnless(
        importlib.util.find_spec("rasterio") is not None,
        "rasterio is an optional Phase 27 dependency",
    )
    def test_audit_accepts_aligned_official_manifest(self):
        import rasterio
        from rasterio.transform import from_origin

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transform = from_origin(100.0, 200.0, 30.0, 30.0)
            profile = {
                "driver": "GTiff",
                "height": 2,
                "width": 2,
                "count": 1,
                "dtype": "uint16",
                "crs": "EPSG:32631",
                "transform": transform,
            }
            arrays = {
                "b2.tif": np.array([[0, 10], [20, 30]], dtype=np.uint16),
                "b3.tif": np.array([[0, 11], [21, 31]], dtype=np.uint16),
                "b4.tif": np.array([[0, 12], [22, 32]], dtype=np.uint16),
                "truth.tif": np.array([[0, 128], [192, 255]], dtype=np.uint16),
                "qa.tif": np.array([[0, 0], [4, 8]], dtype=np.uint16),
            }
            for name, array in arrays.items():
                with rasterio.open(root / name, "w", **profile) as dataset:
                    dataset.write(array, 1)
            mtl = root / "MTL.txt"
            mtl.write_text(
                "\n".join(
                    [
                        "REFLECTANCE_MULT_BAND_2 = 0.00002",
                        "REFLECTANCE_ADD_BAND_2 = -0.1",
                        "REFLECTANCE_MULT_BAND_3 = 0.00002",
                        "REFLECTANCE_ADD_BAND_3 = -0.1",
                        "REFLECTANCE_MULT_BAND_4 = 0.00002",
                        "REFLECTANCE_ADD_BAND_4 = -0.1",
                        "SUN_ELEVATION = 45.0",
                    ]
                ),
                encoding="utf-8",
            )
            manifest = {
                "dataset": "USGS Landsat 8 Collection 2 cloud truth mask validation set",
                "source": "https://doi.org/10.5066/P9FI4A0Y",
                "sciencebase_item_id": "61015b2fd34ef8d7055d6395",
                "license": "CC0-1.0",
                "scenes": [
                    {
                        "scene_id": "synthetic-scene",
                        "b2": "b2.tif",
                        "b3": "b3.tif",
                        "b4": "b4.tif",
                        "truth": "truth.tif",
                        "qa_pixel": "qa.tif",
                        "mtl": "MTL.txt",
                    }
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output = root / "audit.json"
            process = subprocess.run(
                [
                    sys.executable,
                    str(TOOLS_DIR / "audit_phase27_landsat_c2.py"),
                    "--root",
                    str(root),
                    "--manifest",
                    str(manifest_path),
                    "--expected-scenes",
                    "1",
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(result["passed"])
            self.assertEqual(result["truth_histogram"]["192"], 1)


if __name__ == "__main__":
    unittest.main()
