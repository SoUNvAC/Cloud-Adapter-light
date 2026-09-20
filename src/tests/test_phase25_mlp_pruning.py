import unittest

import torch

from tools.prepare_phase25_mlp_checkpoint import prune_state_dict, select_channels


class Phase25MlpPruningTests(unittest.TestCase):
    @staticmethod
    def make_state_dict():
        state = {"unrelated": torch.tensor([7.0])}
        for block in reversed(range(12)):
            prefix = f"backbone.blocks.{block}.mlp."
            fc1 = torch.arange(16, dtype=torch.float32).reshape(8, 2) + block
            fc2 = torch.arange(24, dtype=torch.float32).reshape(3, 8) + block
            state[prefix + "fc1.weight"] = fc1
            state[prefix + "fc1.bias"] = torch.arange(8, dtype=torch.float32)
            state[prefix + "fc2.weight"] = fc2
            state[prefix + "fc2.bias"] = torch.arange(3, dtype=torch.float32)
        return state

    def test_selection_is_sorted_and_uses_path_strength(self):
        fc1 = torch.tensor([[10.0, 0.0], [4.0, 0.0], [3.0, 0.0]])
        fc2 = torch.tensor([[0.1, 3.0, 2.0]])
        selected = select_channels(fc1, fc2, 2)
        self.assertEqual(selected.tolist(), [1, 2])

    def test_ratio_two_slices_all_twelve_blocks(self):
        state = self.make_state_dict()
        pruned = prune_state_dict(state, 2.0)
        self.assertEqual(pruned["backbone.blocks.0.mlp.fc1.weight"].shape, (4, 2))
        self.assertEqual(pruned["backbone.blocks.0.mlp.fc1.bias"].shape, (4,))
        self.assertEqual(pruned["backbone.blocks.0.mlp.fc2.weight"].shape, (3, 4))
        self.assertEqual(pruned["unrelated"].item(), 7.0)

    def test_ratio_four_is_exact_identity(self):
        state = self.make_state_dict()
        converted = prune_state_dict(state, 4.0)
        for key, value in state.items():
            self.assertTrue(torch.equal(converted[key], value), key)


if __name__ == "__main__":
    unittest.main()
