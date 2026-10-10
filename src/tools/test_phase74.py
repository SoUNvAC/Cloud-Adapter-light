import unittest
import numpy as np
from phase74_math import contrast,shuffled_mask,changes,interval,continue_conditions
from phase70_math import local_contrast,checked


class Tests(unittest.TestCase):
    def test_baseline_edges_and_center(self):
        raw=np.random.default_rng(4).random((12,13,13)).astype(np.float32)
        valid=np.ones((12,13),bool);valid[0,0]=False
        x,means,old,a=contrast(raw,valid,np.zeros_like(valid),5)
        np.testing.assert_array_equal(x,checked(local_contrast(raw,valid,5)))
        excluded=np.zeros_like(valid);excluded[4,4]=True
        x,m,o,a=contrast(raw,valid,excluded,5)
        self.assertEqual(a['remaining'][np.flatnonzero(valid.ravel()).tolist().index(4*13+4)],24)
        self.assertEqual(a['remaining'][np.flatnonzero(valid.ravel()).tolist().index(4*13+5)],23)
        # Manual clipped top-left mean; <16 neighbors forces original mean.
        np.testing.assert_array_equal(m[0],o[0])

    def test_all_excluded_fallback(self):
        raw=np.ones((7,8,13),np.float32);valid=np.ones((7,8),bool)
        x,m,o,a=contrast(raw,valid,valid,5)
        self.assertTrue(a['fallback'].all());self.assertEqual(a['remaining'].sum(),0)
        np.testing.assert_array_equal(m,o);self.assertTrue(np.isfinite(x).all())

    def test_manual_common_neighbor_set(self):
        raw=np.random.default_rng(7).random((9,9,13)).astype(np.float32)
        v=np.ones((9,9),bool);e=np.zeros_like(v);e[3,3]=True;e[4,4]=True
        x,m,o,a=contrast(raw,v,e,5)
        mask=np.zeros_like(v);mask[2:7,2:7]=True;mask &= ~e;mask[4,4]=False
        np.testing.assert_allclose(m[40],raw[mask][:,[7,11,12]].astype(float).mean(0),atol=1e-12)
        self.assertEqual(a['remaining'][40],23)

    def test_shuffle_and_finite(self):
        v=np.ones((10,11),bool);v[0]=False;e=np.indices(v.shape)[0]%3==0
        r,a=shuffled_mask('product',v,e);s,b=shuffled_mask('product',v,e)
        np.testing.assert_array_equal(r,s);self.assertEqual(a,b);self.assertEqual(r.sum(),e[v].sum())
        self.assertFalse(r[~v].any())
        raw=np.ones((10,11,13),np.float32);raw[5,5,7]=np.nan
        with self.assertRaises(ValueError):contrast(raw,v,e,5)

    def test_flows_and_bootstrap(self):
        self.assertEqual(changes([0,1,0,1],[1,0,1,0],[0,0,1,1]),dict(new_fp=1,removed_fp=1,rescued_tp=1,lost_tp=1))
        self.assertEqual(interval([.1,.2]),interval([.1,.2]));self.assertEqual(interval([.1])['ci95'],[.1,.1])

if __name__=='__main__':unittest.main()
