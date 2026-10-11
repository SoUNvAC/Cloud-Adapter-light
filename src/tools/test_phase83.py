"""Independent numeric checks of fixed conditional readout rules."""
import unittest
import numpy as np
from run_phase83 import conditional_u,predict_cloud,interval
from phase65d_recovery_math import objective,standardize

class Tests(unittest.TestCase):
    def test_conditional_odds_and_floor(self):
        p=np.array([[.2,0],[.6,.1],[.2,.9]],np.float32)
        np.testing.assert_allclose(conditional_u(p),np.log(np.maximum(p[1].astype(float),1e-6)/np.maximum(p[0].astype(float),1e-6)))
        q=p.copy();q[2]=.01;np.testing.assert_array_equal(conditional_u(p),conditional_u(q))
    def test_weighting_and_gradient(self):
        x=np.array([[0.,1],[2,1],[10,1]]);w=np.array([.25,.25,.5]);y=np.array([0,1,1]);mu,s=standardize(x,w)
        np.testing.assert_allclose(mu,[5.5,1]);self.assertEqual(s[1],1)
        theta=np.array([.1,-.2,.3]);loss,g=objective(theta,x,y,w)
        for i in range(3):
            e=np.zeros(3);e[i]=1e-6
            self.assertAlmostEqual(g[i],(objective(theta+e,x,y,w)[0]-objective(theta-e,x,y,w)[0])/2e-6,places=7)
    def test_threshold_tie_and_bootstrap(self):
        m=dict(mean=[0],scale=[1],theta=[1,0]);np.testing.assert_array_equal(predict_cloud(np.array([[-1.],[0.],[1.]]),m),[0,1,1])
        x=np.array([-2.,0.,3.]);b=np.random.default_rng(83011).choice(x,(10000,3)).mean(1)
        np.testing.assert_allclose(interval(x)['ci95'],np.percentile(b,[2.5,97.5]))
if __name__=='__main__':unittest.main()
