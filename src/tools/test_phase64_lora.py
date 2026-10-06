#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch
from torch import nn


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "cloud_adapter" / "models" / "backbones" / "phase64_lora_layers.py"
)
SPEC = importlib.util.spec_from_file_location("phase64_lora_layers", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
LoRALinear = MODULE.LoRALinear
lora_parameter_count = MODULE.lora_parameter_count


class Phase64LoRATest(unittest.TestCase):
    def test_replacement_is_exact_at_zero_b(self) -> None:
        torch.manual_seed(64)
        linear = nn.Linear(11, 17)
        lora = LoRALinear.from_linear(linear, rank=4, alpha=4.0)
        inputs = torch.randn(3, 5, 11)
        self.assertTrue(torch.equal(linear(inputs), lora(inputs)))
        self.assertFalse(lora.weight.requires_grad)
        self.assertTrue(lora.lora_a.requires_grad)
        self.assertTrue(lora.lora_b.requires_grad)
        self.assertEqual(lora_parameter_count(lora), 4 * (11 + 17))

    def test_only_low_rank_branch_updates(self) -> None:
        torch.manual_seed(65)
        lora = LoRALinear.from_linear(nn.Linear(7, 9), rank=3, alpha=6.0)
        optimizer = torch.optim.SGD(
            [parameter for parameter in lora.parameters() if parameter.requires_grad], lr=0.1
        )
        inputs = torch.randn(4, 7)
        base_before = lora.weight.detach().clone()
        loss = lora(inputs).square().mean()
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.equal(base_before, lora.weight))
        self.assertGreater(float(lora.lora_b.abs().sum()), 0.0)

    def test_disabled_path_is_exact_base_linear(self) -> None:
        torch.manual_seed(66)
        linear = nn.Linear(7, 9)
        lora = LoRALinear.from_linear(linear, rank=3, alpha=6.0)
        with torch.no_grad():
            lora.lora_b.normal_()
        inputs = torch.randn(2, 5, 7)
        lora.enabled = False
        self.assertTrue(torch.equal(lora(inputs), linear(inputs)))


if __name__ == "__main__":
    unittest.main()
