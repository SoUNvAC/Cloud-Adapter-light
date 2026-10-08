import unittest
import numpy as np
from phase65d_followup_metrics import loo_ridge

class FoldTests(unittest.TestCase):
    def test_fold_target_does_not_leak(self):
        x=np.arange(12).reshape(6,2);y=np.arange(6,dtype=float)
        a=loo_ridge(x,y);y[0]=1000;b=loo_ridge(x,y)
        self.assertEqual(a['predictions'][0],b['predictions'][0])
        self.assertEqual(a['baseline_predictions'][0],b['baseline_predictions'][0])
    def test_constant_features_equal_fold_mean(self):
        x=np.ones((5,3));y=np.arange(5,dtype=float);a=loo_ridge(x,y)
        np.testing.assert_allclose(a['predictions'],a['baseline_predictions'])
    def test_affine_feature_units_do_not_change_oof(self):
        x=np.arange(12).reshape(6,2);y=np.sin(np.arange(6));a=loo_ridge(x,y);b=loo_ridge(x*7+100,y)
        np.testing.assert_allclose(a['predictions'],b['predictions'])

if __name__=='__main__':unittest.main()
