"""Synthetic fixed-rule tests; never opens published point labels."""
import unittest
from phase103_reference import classify,code

class MappingTests(unittest.TestCase):
    def rule(self,c,s,u=32):
        return classify({'CLOUD_CHARACTERISTICS_ID':str(c),'SHADOW_ID':str(s),'SURFACE_ID':str(u)})
    def test_all_supported_pairs(self):
        for c in [1,7]:
            self.assertEqual(self.rule(c,3)[0],'positive_ground_cloud_shadow')
            for s in [1,2]:self.assertEqual(self.rule(c,s)[0],'negative_surface')
        for c in [2,3,4,5,6,9]:self.assertEqual(self.rule(c,1)[0],'negative_cloud')
    def test_global_exclusions(self):
        for u in [9,15,16]:self.assertEqual(self.rule(7,3,u)[1],['spatial_cloud_mixture_SURFACE_ID'])
        self.assertEqual(self.rule(7,4)[1],['shadow_above_cloud'])
        self.assertEqual(self.rule(7,4,16)[1],['spatial_cloud_mixture_SURFACE_ID','shadow_above_cloud'])
    def test_none_not_unknown_or_clear(self):
        self.assertEqual(self.rule(0,3)[0],'ignore')
        self.assertEqual(self.rule(7,0)[0],'ignore')
        self.assertEqual(self.rule(7,3,0)[0],'positive_ground_cloud_shadow') # U0 known code, not globally excluded in fixed task.
        self.assertEqual(self.rule(1,3)[0],'positive_ground_cloud_shadow')
    def test_ambiguous_and_conflicting(self):
        for c in [8,10,11,12]:self.assertEqual(self.rule(c,1)[0],'ignore')
        for c in [2,3,4,5,6,9]:
            for s in [2,3]:self.assertEqual(self.rule(c,s)[1],['cloud_surface_shadow_conflict'])
    def test_missing_unknown_and_integral_encoding(self):
        self.assertEqual(self.rule(99,3)[0],'ignore')
        self.assertEqual(self.rule('',3)[0],'ignore')
        self.assertEqual(self.rule(7,99)[0],'ignore')
        self.assertIsNone(code('1.3'));self.assertIsNone(code('NaN'));self.assertEqual(code('1.0'),1)

if __name__=='__main__':unittest.main()
