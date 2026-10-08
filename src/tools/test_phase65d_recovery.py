import unittest
import numpy as np
from phase65d_recovery_math import objective,standardize,group_threshold,logit_score,predict_logit

class RecoveryTest(unittest.TestCase):
    def test_gradient(self):
        rng=np.random.default_rng(8);z=rng.normal(size=(20,5));y=rng.integers(0,2,20);w=np.ones(20)/20;t=rng.normal(size=6)
        loss,g=objective(t,z,y,w);numeric=[]
        for i in range(6):
            e=np.eye(6)[i]*1e-5;numeric.append((objective(t+e,z,y,w)[0]-objective(t-e,z,y,w)[0])/2e-5)
        np.testing.assert_allclose(g,numeric,atol=1e-8)
    def test_group_ties_budget(self):
        arrays=[np.array([1.,1.,0.]),np.array([2.,0.,0.,0.,0.,0.])]
        t=group_threshold(arrays,.1)
        rate=np.mean([(a>=t).mean() for a in arrays]);self.assertLessEqual(rate,.1)
        self.assertAlmostEqual(rate,1/12)
        self.assertGreater(group_threshold([np.ones(5)],.01),1)
    def test_equal_group_weight_duplicate_invariance(self):
        x=np.array([[1.,5.],[3.,5.],[20.,5.]]);w=np.array([.25,.25,.5]);m,s=standardize(x,w)
        xx=np.r_[x[:2],x[2:],x[2:]];mm,ss=standardize(xx,np.ones(4)*.25)
        np.testing.assert_allclose(m,mm);np.testing.assert_allclose(s,ss);self.assertEqual(s[1],1)
    def test_monotone_baseline(self):
        x=logit_score(np.array([.1,.2,.5,.9]))[:,None]
        model=dict(mean=[0],scale=[1],theta=[2,-1]);p=predict_logit(x,model)
        self.assertTrue((np.diff(p)>0).all())

if __name__=='__main__':unittest.main()
