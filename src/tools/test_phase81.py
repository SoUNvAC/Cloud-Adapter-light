"""Crossed-readout boundary and absent-class tests, without real data."""
import unittest
import numpy as np
from run_phase81 import cross, group_row, interval

class Tests(unittest.TestCase):
    def test_cross_at_exact_logit_threshold(self):
        scores={'M':np.array([-.5,-.4,-.6,-.5],np.float32),'L':np.array([1.,0.,2.,0.],np.float32)}
        winners={'M':np.array([0,1,0,1]),'L':np.array([1,0,1,0])}
        p=cross(scores,{'M':-.5,'L':1.},winners)
        self.assertEqual(p['LM'].tolist(),[2,1,2,1])
        self.assertEqual(p['ML'].tolist(),[2,2,1,2])
        self.assertTrue(np.array_equal(p['LM']==2,p['LL']==2))
        self.assertTrue(np.array_equal(p['ML']==2,p['MM']==2))
        with self.assertRaises(AssertionError):cross(scores,{'M':-.5,'L':1.},{'M':np.array([2]*4),'L':winners['L']})

    def test_absent_classes_not_zero_filled(self):
        r=group_row('MM',{'product':'synthetic','group_id':'synthetic'},np.zeros(4,dtype=np.uint8),np.zeros(4,dtype=np.uint8),None)
        self.assertEqual(r['defined_iou_classes'],1);self.assertEqual(r['miou_percent'],100.)
        self.assertIsNone(r['ap']);self.assertIsNone(r['iou_percent'][2])
        q=group_row('LM',{'product':'synthetic','group_id':'synthetic'},np.zeros(4,dtype=np.uint8),np.array([0,0,0,1],np.uint8),None)
        self.assertEqual(q['miou_percent'],37.5)

    def test_group_bootstrap(self):
        x=np.array([-5.,2.,8.]);r=interval(x)
        b=np.random.default_rng(81011).choice(x,size=(10000,3),replace=True).mean(1)
        np.testing.assert_allclose(r['ci95'],np.percentile(b,[2.5,97.5]),atol=1e-12)

if __name__=='__main__':unittest.main()
