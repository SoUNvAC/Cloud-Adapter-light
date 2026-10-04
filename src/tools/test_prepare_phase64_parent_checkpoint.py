#!/usr/bin/env python3

from __future__ import annotations

import unittest
import importlib

try:
    import torch
except ImportError:  # pragma: no cover - local workstation may omit torch
    torch = None


@unittest.skipIf(torch is None, "torch is only required in the training environment")
class ParentCheckpointConversionTest(unittest.TestCase):
    @staticmethod
    def converter():
        for module_name in (
            "tools.prepare_phase64_parent_checkpoint",
            "src.tools.prepare_phase64_parent_checkpoint",
        ):
            try:
                return importlib.import_module(module_name).convert_checkpoint
            except ModuleNotFoundError as exc:
                if exc.name != module_name.split(".")[0]:
                    raise
        raise ModuleNotFoundError("Cannot import Phase 64 checkpoint converter")

    def test_only_native_classifier_is_removed(self) -> None:
        convert_checkpoint = self.converter()
        checkpoint = {
            "state_dict": {
                "backbone.blocks.0.weight": torch.ones(2, 2),
                "decode_head.cls_embed.weight": torch.ones(5, 8),
                "decode_head.cls_embed.bias": torch.ones(5),
                "decode_head.mask_embed.layers.0.weight": torch.ones(3, 3),
            },
            "optimizer": {"unsafe_to_resume": True},
        }
        converted, dropped = convert_checkpoint(checkpoint)
        self.assertEqual(
            {row["key"] for row in dropped},
            {
                "decode_head.cls_embed.weight",
                "decode_head.cls_embed.bias",
            },
        )
        self.assertNotIn("optimizer", converted)
        self.assertEqual(len(converted["state_dict"]), 2)

    def test_wrong_source_classifier_size_stops(self) -> None:
        convert_checkpoint = self.converter()
        checkpoint = {
            "state_dict": {
                "decode_head.cls_embed.weight": torch.ones(4, 8),
                "decode_head.cls_embed.bias": torch.ones(4),
            }
        }
        with self.assertRaisesRegex(ValueError, "expected 5"):
            convert_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
