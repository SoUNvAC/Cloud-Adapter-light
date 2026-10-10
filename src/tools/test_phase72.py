"""Check tied AP, group pairing, safety bounds and gatekeeping on synthetic data."""
import unittest
from copy import deepcopy
import numpy as np
from phase72_statistics import interval,paired,decision
from phase65d_metrics import curve

class Statistics(unittest.TestCase):
    def test_ties_and_undefined(self):
        self.assertEqual(curve([.8,.8],[True,False])['ap'],.5)
        self.assertIsNone(curve([.2,.3],[False,False])['ap'])
    def test_small_support_and_fixed_seed(self):
        self.assertIsNone(interval([1])['ci95'])
        self.assertEqual(interval([1,2,3]),interval([1,2,3]))
        self.assertEqual(interval([.01,.01])['upper95_one_sided'],.01)
    def test_noninferiority_and_gatekeeping(self):
        p=dict(ap_pp=interval([2,2]),recall_pp=interval([-1,-1]),miou_pp=interval([-1,-1]))
        s=dict(ap_pp=interval([2,2]));f=interval([.01,.01])
        self.assertTrue(decision(p,s,f)['all_claims_confirmed'])
        q=deepcopy(p);q['recall_pp']=interval([-1.01,-1.01])
        d=decision(q,s,f);self.assertFalse(d['specificity_test_performed']);self.assertIsNone(d['spectral_specificity_confirmed'])
        q=deepcopy(p);q['ap_pp']=interval([0,0]);self.assertFalse(decision(q,s,f)['gates']['ap_increment'])
    def test_pairing_and_no_shadow(self):
        a=[dict(group_id='a',product='a',ranking={'ap':None},recall=None,miou_percent=80),
           dict(group_id='b',product='b',ranking={'ap':.6},recall=.3,miou_percent=70)]
        b=deepcopy(a);b[1]['ranking']['ap']=.5
        result=paired(a,b);self.assertEqual(result['ap_pp']['n_groups'],1);self.assertEqual(result['miou_pp']['n_groups'],2)
        b.reverse()
        with self.assertRaises(AssertionError):paired(a,b)

if __name__=='__main__':unittest.main()
