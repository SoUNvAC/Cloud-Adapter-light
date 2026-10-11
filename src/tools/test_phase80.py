"""Meaningful state-loading and paired group-bootstrap safeguards (no GPU)."""
import unittest
import numpy as np
from phase80_common import permitted_lora,state_keys,interval

class Checks(unittest.TestCase):
    def test_only_exact_lora_locations(self):
        self.assertTrue(permitted_lora('backbone.blocks.11.mlp.fc2.lora_b'))
        for n in ['decode_head.lora_a','backbone.blocks.0.attn.qkv.weight',
                  'backbone.blocks.0.mlp.qkv.lora_a','backbone.blocks.x.attn.qkv.lora_a']:
            self.assertFalse(permitted_lora(n))
    def test_missing_or_unknown_source_key_rejected(self):
        source={'head.weight':np.zeros((2,3))}
        target={**source,'backbone.blocks.0.attn.qkv.lora_b':np.zeros((2,4))}
        self.assertEqual(len(state_keys(source,target)),1)
        with self.assertRaises(ValueError):state_keys(source,{'head.other':np.zeros((2,3))})
        with self.assertRaises(ValueError):state_keys(source,{**target,'unknown':np.zeros(1)})
        with self.assertRaises(ValueError):state_keys(source,{**target,'head.weight':np.zeros((3,2))})
    def test_group_bootstrap_constant_and_deterministic(self):
        a=interval([2,2,2]);self.assertEqual(a['ci95'],[2.,2.])
        self.assertEqual(interval([-1,3,4]),interval([-1,3,4]))
        self.assertIsNone(interval([2])['ci95'])

if __name__=='__main__':unittest.main()
