from __future__ import annotations

import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np
import torch

from tools.prepare_phase64b_six_channel_checkpoint import (
    STEM_PREFIX,
    convert,
)

HAS_MMSEG = importlib.util.find_spec("mmseg") is not None


@unittest.skipUnless(HAS_MMSEG, "mmseg is only available in the training environment")
class Phase64BProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cloud_adapter.datasets.phase64b_six_band import (
            FinalizePhase64BSpatialProtocol,
            _assert_development_split,
            _landsat_toa_reflectance,
        )

        cls.finalizer = FinalizePhase64BSpatialProtocol
        cls.assert_split = staticmethod(_assert_development_split)
        cls.convert_landsat = staticmethod(_landsat_toa_reflectance)

    def test_target_test_is_rejected(self):
        with self.assertRaises(ValueError):
            self.assert_split("target_test")

    def test_s2_area_grid_and_invalid_mask(self):
        image = np.ones((512, 512, 6), dtype=np.float32)
        valid = np.ones((512, 512), dtype=np.uint8)
        valid[:2, :2] = 0
        label = np.zeros((512, 512), dtype=np.uint8)
        result = self.finalizer().transform(
            {
                "phase64b_source": "cloudsen_l1c",
                "img": image,
                "gt_seg_map": label,
                "phase64b_valid_mask": valid,
            }
        )
        self.assertEqual(result["img"].shape, (256, 256, 6))
        self.assertEqual(result["gt_seg_map"].shape, (256, 256))
        self.assertEqual(int(result["gt_seg_map"][0, 0]), 255)

    def test_landsat_mtl_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene_mtl.txt"
            lines = ["SUN_ELEVATION = 30"]
            for band in (4, 3, 2, 5, 6, 7):
                lines.extend(
                    [
                        f"REFLECTANCE_MULT_BAND_{band} = 2.0E-05",
                        f"REFLECTANCE_ADD_BAND_{band} = -0.1",
                    ]
                )
            path.write_text("\n".join(lines), encoding="utf-8")
            values = np.full((2, 2, 6), 10000, dtype=np.uint16)
            converted = self.convert_landsat(values, str(path))
            np.testing.assert_allclose(converted, 0.2, rtol=1e-5)


class Phase64BCheckpointTest(unittest.TestCase):
    def test_checkpoint_stems_expand_to_six(self):
        prefix = STEM_PREFIX
        state = {
            "backbone.patch_embed.proj.weight": torch.ones(2, 3, 2, 2),
            f"{prefix}.conv_dw.weight": torch.ones(3, 1, 1, 1),
            f"{prefix}.conv_pw.weight": torch.ones(4, 3, 1, 1),
        }
        for suffix in ("weight", "bias", "running_mean", "running_var"):
            state[f"{prefix}.bn1.{suffix}"] = torch.ones(3)
        shapes = convert(state)
        self.assertEqual(shapes["backbone.patch_embed.proj.weight"], (2, 6, 2, 2))
        self.assertEqual(shapes[f"{prefix}.conv_dw.weight"], (6, 1, 1, 1))
        self.assertEqual(shapes[f"{prefix}.conv_pw.weight"], (4, 6, 1, 1))
        self.assertTrue(
            torch.equal(
                state["backbone.patch_embed.proj.weight"][:, :3],
                torch.full((2, 3, 2, 2), 0.5),
            )
        )


if __name__ == "__main__":
    unittest.main()
