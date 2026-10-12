"""Small contracts for fixed raw-band order, binary output and fit objective."""
import unittest
import numpy as np
import run_phase92 as r
from phase65d_recovery_math import objective, predict_logit


class Contracts(unittest.TestCase):
    def test_band_order(self):
        x=np.tile(np.arange(8,dtype=np.float64),(4,1))
        np.testing.assert_array_equal(r.raw_selected(x,'S3')[0],[2,3,4])
        np.testing.assert_array_equal(r.raw_selected(x,'R6')[0],[5,6,7,2,3,4])
        self.assertEqual(r.BANDS['R6'],['B04','B03','B02','B08','B11','B12'])

    def test_free_negative_coefficient_and_intercept(self):
        z=np.array([[.1,.2,.3],[-.3,.2,-.1],[.2,-.4,.5]])
        theta=np.array([-.4,.2,-.5,.1]);y=np.array([0.,1.,1.]);w=np.array([.2,.3,.5])
        loss,grad=objective(theta,z,y,w)
        for i in range(4):
            d=np.zeros(4);d[i]=1e-6
            numeric=(objective(theta+d,z,y,w)[0]-objective(theta-d,z,y,w)[0])/2e-6
            self.assertAlmostEqual(numeric,grad[i],places=8)
        model=dict(mean=[0,0,0],scale=[1,1,1],theta=theta.tolist())
        np.testing.assert_array_equal(predict_logit(np.asfortranarray(z),model),r.independent_logit(np.asfortranarray(z),model))

    def test_binary_no_shadow_and_cloud_negatives(self):
        out=r.row_metric(np.array([.9,.2,.7,.1]),np.array([0,1,2,2]),.5,dict(product='synthetic'))
        self.assertEqual(out['true_positive'],1);self.assertEqual(out['false_positive'],1)
        self.assertEqual(out['surface_fpr'],1);self.assertEqual(out['cloud_fpr'],0)
        self.assertAlmostEqual(out['shadow_iou'],1/3)
        self.assertNotIn('miou',out);self.assertNotIn('confusion',out)
        no=r.row_metric(np.array([.9,.2]),np.array([0,1]),.5,dict(product='none'))
        self.assertIsNone(no['ap']);self.assertIsNone(no['recall']);self.assertEqual(no['fpr'],.5)

    def test_fixed_group_bootstrap(self):
        values=[-2.,1.,4.];result=r.interval(values)
        expected=np.array(values)[np.random.default_rng(92012).integers(3,size=(10000,3))].mean(1)
        self.assertEqual(result['mean'],1.)
        np.testing.assert_array_equal(result['ci95'],np.quantile(expected,[.025,.975]))


if __name__=='__main__':unittest.main()
