"""Synthetic ranking/NA/conditional-source input contracts, no research evaluation."""
import unittest
import numpy as np
from phase65d_metrics import curve
from phase89_cpu_ap import average_precision_score
from phase65d_recovery_math import logit_score
from phase105_common import repeats,key

class Contracts(unittest.TestCase):
    def test_tied_scores(self):
        y=[1,0,1,0];s=[.9,.9,.8,.2]
        self.assertAlmostEqual(curve(s,y)['ap'],7/12)
        self.assertAlmostEqual(average_precision_score(y,np.asarray(s,np.float32)),7/12)
    def test_no_positive_NA(self):
        self.assertIsNone(curve([.2,.5],[False,False])['ap'])
    def test_merge_products_before_group_AP(self):
        a=curve([.9,.1],[1,0])['ap'];b=curve([.2,.8],[1,0])['ap']
        merged=curve([.9,.1,.2,.8],[1,0,1,0])['ap']
        self.assertAlmostEqual(merged,5/6);self.assertNotEqual(merged,(a+b)/2)
    def test_source_logit_cast_order(self):
        p=np.array([0,1,1e-7,.12345679],np.float32)
        actual=logit_score(p).astype(np.float32).astype(np.float64)
        q=np.clip(p.astype(np.float64),1e-6,1-1e-6)
        np.testing.assert_array_equal(actual,(np.log(q)-np.log1p(-q)).astype(np.float32).astype(np.float64))
    def test_repeat_selection_order_independent(self):
        rows=[dict(product_id='fixed',y=i*1020,x=0) for i in range(8)]
        self.assertEqual([key(w) for w in repeats(rows)],[key(w) for w in repeats(list(reversed(rows)))])

if __name__=='__main__':unittest.main()
