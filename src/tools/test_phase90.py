import unittest
import numpy as np
from run_phase90 import inputs, independent_prob, control_decision, interval
from phase65d_recovery_math import predict_logit
from phase89_math import sigmoid, area


class FrozenContracts(unittest.TestCase):
    def test_RGB_TOA_and_exact_order(self):
        spec=np.full((3,3,6),[.02,.03,.04,.08,.11,.12])
        x=inputs(np.full((3,3),.5),np.full((3,3),.8),spec,'L2_RGB')
        np.testing.assert_allclose(x[0],[0,np.log(4),.04,.03,.02],rtol=1e-7)
        for name,first in [('Source_spectral',np.log(4)),('MsRE_spectral',0)]:
            xx=inputs(np.full((3,3),.5),np.full((3,3),.8),spec,name)
            np.testing.assert_allclose(xx[0],[first,.08,.11,.12],rtol=1e-7)

    def test_clip_and_nonlinear_before_area(self):
        spec=np.zeros((3,3,6));ms=np.array([[0,0,1.]]*3)
        x=inputs(ms,np.ones((3,3)),spec,'MsRE_spectral')
        self.assertTrue(np.isfinite(x).all())
        m=dict(mean=[0]*4,scale=[1]*4,theta=[1,0,0,0,1])
        cpu=independent_prob(x,m,(3,3))
        np.testing.assert_allclose(cpu,area(sigmoid(predict_logit(x,m)).reshape(3,3),3),rtol=0,atol=2e-15)
        self.assertGreater(abs(cpu.item()-sigmoid(np.log(ms.mean()/(1-ms.mean()))+1)),.1)

    def test_source_nonshadow_and_ge_tie(self):
        p={'Source':np.array([[[.8,.1,.1],[.8,.1,.1]]]),
           'MsRE':np.array([[[.1,.8,.1],[.1,.8,.1]]])}
        scores=np.array([[.4,.5]])
        for ns,expected in [('source',[[0,2]]),('msre',[[1,2]])]:
            np.testing.assert_array_equal(control_decision(scores,p,dict(nonshadow=ns,threshold=0)),expected)

    def test_fixed_group_bootstrap(self):
        a=interval([-2,1,4]);self.assertEqual(a,interval([-2,1,4]))
        self.assertEqual(a['seed'],90012);self.assertEqual(a['n_groups'],3)
        x=np.array([-2,1,4]);rng=np.random.default_rng(90012)
        b=np.array([sum(x[i])/3 for i in rng.integers(3,size=(10000,3))])
        np.testing.assert_array_equal(np.percentile(b,[2.5,97.5]),a['ci95'])


if __name__=='__main__':unittest.main()
