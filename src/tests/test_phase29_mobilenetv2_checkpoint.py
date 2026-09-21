import unittest

import torch

from tools.prepare_phase29_mobilenetv2_checkpoint import convert_state_dict


class Phase29MobileNetV2CheckpointTests(unittest.TestCase):
    def test_strips_backbone_prefix_and_drops_head(self):
        state = {
            "backbone.conv1.conv.weight": torch.ones(2, 2),
            "backbone.layer1.0.conv.weight": torch.zeros(2, 2),
            "head.fc.weight": torch.full((2, 2), 3.0),
        }
        converted = convert_state_dict(state)
        self.assertEqual(
            set(converted), {"conv1.conv.weight", "layer1.0.conv.weight"}
        )
        self.assertTrue(torch.equal(converted["conv1.conv.weight"], torch.ones(2, 2)))

    def test_rejects_unknown_non_model_keys(self):
        with self.assertRaises(ValueError):
            convert_state_dict(
                {
                    "backbone.conv1.conv.weight": torch.ones(1),
                    "optimizer.state": torch.ones(1),
                }
            )


if __name__ == "__main__":
    unittest.main()
